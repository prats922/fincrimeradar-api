"""
Content indexing pipeline for the guide chatbot.

Parses guide HTML files from the fincrimeradar-repo site, strips nav/
footer/cheat-sheet/decision-scenario/quiz/promo markup, extracts clean
article prose, chunks it at paragraph level, embeds each chunk, and
writes chunks + embeddings to a flat JSON file.

Session 1 proved the parser on 4 pages, embedding locally via
sentence-transformers since the script only needed to run on a dev
machine. Session 2 extended parsing to the full site: every remaining
guide, screen.html's tool explainer, and scenario-lab.html's
tool-context copy. Session 3 (live /api/chat endpoint) embeds live
questions via Voyage instead of a local model, to avoid deploying torch
to Render's free tier, so this script now embeds the corpus via Voyage
too, corpus and query embeddings must share one vector space or cosine
similarity is meaningless (and, since Voyage's dimension differs from
the old local model's, a shape mismatch that would crash outright
rather than silently misbehave). See BACKLOG.md in fincrimeradar-repo,
"Guide chatbot" entry, for all sessions' findings.
"""

import json
import os
import re
import time

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

GUIDES_DIR = os.environ.get(
    "GUIDES_DIR",
    os.path.join(os.path.dirname(__file__), "..", "fincrimeradar-repo"),
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "guide_index.json")
# Must match the model routes_guide_chat.py uses to embed live questions.
EMBEDDING_MODEL = "voyage-4-lite"
VOYAGE_BATCH_SIZE = 100

# Classes that mark interactive/non-prose widgets (quizzes, self-assessment
# tools, decision-scenario diagrams, cheat-sheet SVGs, nav toggles). These
# are removed wholesale before prose extraction, they are not article
# content and would pollute retrieval with UI copy and quiz option text.
STRIP_CLASSES = [
    "quiz-q", "quiz-opts", "quiz-feedback", "quiz-explain",
    # dilemma-card wrapper is kept: it holds dilemma-scenario, the actual
    # case narrative prose. Only the interactive picker UI and the
    # JS-populated verdict reveal are stripped out of it.
    "dilemma-options", "dilemma-verdict", "dilemma-header",
    "scene-flow", "scene-node", "scene-edge",
    "toc-scenario",
    "self-assess",
    "firm-toggle",
    "cheat-sheet-card", "cheat-sheet-scroll",
    "pull-quote",
    "inline-tool-promo",
    # Page-level subtitle/tagline directly under a doc-header, e.g.
    # methodology.html's "Where the data comes from, and how the
    # screening tool actually works." It's framing copy for the page,
    # not a retrievable fact, and it has no heading context since it
    # sits above the first real h2. Logged from session 1 retrieval
    # testing, where it surfaced as a low-value top-3 result.
    "highlight",
    # "Try our screening tool" promo CTA box. Must be stripped wholesale,
    # not just have its own text excluded: it reuses the tl-title class
    # from the genuine tl-item timeline pattern below, so if this wrapper
    # survives, its promo tl-title ("See how MLR screening obligations
    # work in practice") gets picked up by the heading pass and mislabels
    # whatever real content follows it.
    "try-tool",
    # End-of-guide "you've completed this series" CTA box, same shape as
    # try-tool (badge, title, blurb, action buttons), promotional rather
    # than reference content.
    "series-complete",
    # Prev/next part navigation chrome ("Part 3 of 3").
    "part-footer",
    # Scored interactive alert-triage game (tm-guide.html). Contains
    # realistic-looking case text, but it's gamified UI (score display,
    # click-to-resolve buttons) rather than reference prose, same
    # rationale as self-assess above.
    "simulator",
    # Pseudocode rule-engine diagram (tm-guide.html), not natural
    # language even once extracted.
    "rule-anatomy",
    # UI instruction ("Click an item to mark it covered") plus a
    # disclaimer that's also stated in full by checklist-disclaimer
    # later in the same box.
    "checklist-intro",
    # The "+" toggle icon nested inside every faq-q. Without stripping
    # this, faq-q's extracted text ends in a stray "+".
    "faq-icon",
]
STRIP_TAGS = ["script", "style", "nav", "footer", "button", "svg"]

# Prose that lives outside the standard h2-h4/p/li tags, in a <div> or
# <span> instead (a repeated "icon + title + description" card pattern
# shows up across nearly every guide, under a different class-name
# prefix per widget: timelines, red-flag cards, case studies, FAQ pairs,
# checklists, and more). Heading-role elements update the running
# "current heading" the same way an h3 would; text-role elements are
# treated as paragraph chunks. Matched regardless of tag name (div or
# span), see the find_all predicate in extract_chunks.
DIV_HEADING_CLASSES = {
    "scenario-name", "step-title",
    "tl-title",  # genuine timeline items only, try-tool's promo variant is stripped first
    "offence-name", "oc-title",
    "me-step-title", "reason-title", "delisting-step-title",
    "pep-cat-title", "rc-title",
    "sk-title", "sk-item-title",
    "rf-title", "mc-title", "sol-title", "ps-title",
    "rec-title",
    "type-name",  # <span>, sanctions-compliance-guide.html type cards
    "sow-title",
    "case-name",
    "milestone-title",
    "flash-term",  # nested inside flash-face.flash-front, precedes flash-face.flash-back
    "tm-label", "tc-title",
    "fw-word",
    "compare-label", "p-label",
    "checklist-label", "cl-section-title",
    "archetype-name",
    "ng-title",
    "memory-title",
    "ec-firm",  # <span>, enforcement case-study header e.g. "Major UK bank — AML systems failure"
    "faq-q",
}
DIV_TEXT_CLASSES = {
    "step-body", "callout-text",
    "tl-desc", "tl-body",
    "offence-desc", "oc-desc",
    "me-step-desc", "reason-desc", "delisting-step-desc",
    "pep-cat-desc", "rc-desc",
    "sk-item-desc",
    "rf-desc", "mc-desc", "sol-desc", "ps-desc",
    "rec-tagline",
    "type-detail",
    "sow-desc", "sow-eg",
    "case-body",
    "milestone-desc",
    "flash-back",
    "tm-desc", "tc-desc",
    "fw-desc",
    "compare-text", "p-desc",
    "checklist-text", "cl-text", "checklist-disclaimer",
    "cs-lesson",
    "memory-text",
    "humour-text",
    "peel-chain-caption",
    "ec-failures", "ec-lesson",
    "faq-a",
}

# Each source defines which container(s) hold article prose, and a label
# used in chunk metadata for citation back to the source page. All guide
# pages use "article-section" as their prose container (confirmed across
# every file below), only methodology (policy-body), screen.html
# (toolContextBody), and scenario-lab.html (tool-context) differ.
ARTICLE_SECTION_GUIDES = [
    "mlro-handbook-part1", "mlro-handbook-part2",
    "fraud-red-flags-guide",
    "aml-guide-part1", "aml-guide-part2", "aml-guide-part3",
    "crypto-guide-part1", "crypto-guide-part2", "crypto-guide-part3",
    "crypto-guide-part4", "crypto-guide-part5", "crypto-guide-part6",
    "fatf-guide-part1", "fatf-guide-part2",
    "false-positive-playbook", "kyc-onboarding-dilemma",
    "pep-guide-part1", "pep-guide-part2", "pep-guide-part3",
    "sanctions-compliance-guide",
    "sar-guide-part1", "sar-guide-part2", "sar-guide-part3",
    "screening-alerts-guide", "tm-guide", "ubo-investigation-handbook",
]
SOURCES = [
    {
        "id": guide_id,
        "file": f"{guide_id}.html",
        "url": f"/{guide_id}.html",
        "select": (True, {"class": "article-section"}),
    }
    for guide_id in ARTICLE_SECTION_GUIDES
] + [
    {
        "id": "methodology",
        "file": "methodology.html",
        "url": "/methodology.html",
        "select": (True, {"class": "policy-body"}),
    },
    {
        "id": "screen-tool-explainer",
        "file": "screen.html",
        "url": "/screen.html#toolContext",
        "select": (True, {"id": "toolContextBody"}),
    },
    {
        "id": "scenario-lab-tool-context",
        "file": "scenario-lab.html",
        "url": "/scenario-lab.html",
        "select": (True, {"class": "tool-context"}),
    },
]


def clean_containers(containers):
    for container in containers:
        for tag_name in STRIP_TAGS:
            for el in container.find_all(tag_name):
                el.decompose()
        for cls in STRIP_CLASSES:
            for el in container.find_all(class_=cls):
                el.decompose()
    return containers


def extract_chunks(source):
    path = os.path.join(GUIDES_DIR, source["file"])
    with open(path, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    tag_name, attrs = source["select"]
    containers = soup.find_all(tag_name, attrs=attrs)
    if not containers:
        raise ValueError(f"No containers matched {source['select']} in {source['file']}")
    clean_containers(containers)

    chunks = []
    current_heading = None
    for container in containers:
        # Some prose lives outside p/h2-h4 entirely: fraud-red-flags-guide's
        # per-scenario headers are a scenario-num + scenario-name div pair,
        # mlro-handbook's approval-process steps use step-title/step-body
        # divs, and sanctions-compliance-guide's type cards use a <span
        # class="type-name"> heading. Missing these would silently drop
        # real article content rather than just mislabeling it, so the
        # match is by class, not tag name.
        # A single find_all pass keeps everything in document order, two
        # separate passes concatenated would put all class-matched elements
        # after all p/li matches, breaking heading tracking below.
        matches = container.find_all(lambda t: t.name in ("h2", "h3", "h4", "p", "li")
                                      or bool(set(t.get("class") or [])
                                              & (DIV_HEADING_CLASSES | DIV_TEXT_CLASSES)))
        for el in matches:
            classes = set(el.get("class") or [])
            # Some text-role classes (e.g. callout-text) are reused for a
            # link-list "see also" box in some guides rather than prose,
            # e.g. crypto-guide-part6's series-navigation callout. A
            # nested ul/ol is never real prose regardless of which class
            # wraps it, get_text() would otherwise concatenate the link
            # labels into one meaningless chunk.
            if classes & DIV_TEXT_CLASSES and el.find(["ul", "ol"]):
                continue
            # <li> is always matched (real bullet lists are common
            # article content), but a link-list's individual <li>
            # elements survive that guard independently, each one is
            # just an <a>, not prose. crypto-guide-part6's series-nav
            # list leaked 6 separate one-line chunks this way before
            # this check existed.
            if el.name == "li":
                link = el.find("a")
                if link and normalize_text(el.get_text(" ", strip=True)) == normalize_text(link.get_text(" ", strip=True)):
                    continue
            text = normalize_text(el.get_text(" ", strip=True))
            if not text:
                continue
            if el.name in ("h2", "h3", "h4") or classes & DIV_HEADING_CLASSES:
                if "scenario-name" in classes:
                    num_el = el.parent.find(class_="scenario-num")
                    num_text = normalize_text(num_el.get_text(" ", strip=True)) if num_el else None
                    current_heading = f"{num_text}: {text}" if num_text else text
                else:
                    current_heading = text
                continue
            # Skip short fragments (button labels, stray UI copy that
            # survived strip rules) that aren't real prose paragraphs.
            if len(text.split()) < 8:
                continue
            chunks.append({
                "source_id": source["id"],
                "url": source["url"],
                "heading": current_heading,
                "text": text,
            })
    return chunks


def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()


def embed_texts(texts, input_type):
    """Embed via Voyage, input_type is "document" for corpus chunks or
    "query" for a live question, Voyage prepends a different internal
    instruction for each, using the wrong one for either side degrades
    retrieval quality even though both still return a same-shaped vector."""
    api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "VOYAGE_API_KEY is not set (or empty) in .env. "
            "Add the real key to fincrimeradar-api/.env and try again."
        )

    embeddings = []
    for i in range(0, len(texts), VOYAGE_BATCH_SIZE):
        batch = texts[i:i + VOYAGE_BATCH_SIZE]
        for attempt in range(5):
            response = requests.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={"input": batch, "model": EMBEDDING_MODEL, "input_type": input_type},
                timeout=60,
            )
            if response.status_code == 429:
                # Free-tier rate limit, hit reliably at this corpus size
                # with VOYAGE_BATCH_SIZE=100. Honor Retry-After when
                # Voyage sends one, otherwise back off.
                wait = int(response.headers.get("retry-after", 20 * (attempt + 1)))
                print(f"  rate limited, waiting {wait}s (attempt {attempt + 1}/5)...")
                time.sleep(wait)
                continue
            response.raise_for_status()
            data = response.json()
            embeddings.extend(item["embedding"] for item in data["data"])
            break
        else:
            raise SystemExit("Voyage rate limit persisted after 5 retries, aborting.")
        print(f"  embedded {min(i + VOYAGE_BATCH_SIZE, len(texts))}/{len(texts)}")
    return embeddings


def main():
    print(f"Reading guides from: {os.path.abspath(GUIDES_DIR)}")
    all_chunks = []
    for source in SOURCES:
        chunks = extract_chunks(source)
        print(f"  {source['file']}: {len(chunks)} paragraph chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No chunks extracted, aborting before embedding.")

    print(f"\nEmbedding {len(all_chunks)} chunks via Voyage ({EMBEDDING_MODEL})...")
    texts = [f"{c['heading']}. {c['text']}" if c["heading"] else c["text"] for c in all_chunks]
    embeddings = embed_texts(texts, input_type="document")

    for chunk, embedding, chunk_id in zip(all_chunks, embeddings, range(len(all_chunks))):
        chunk["id"] = chunk_id
        chunk["embedding"] = embedding

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": EMBEDDING_MODEL,
            "dim": len(embeddings[0]),
            "chunks": all_chunks,
        }, f, ensure_ascii=False, indent=None)

    print(f"\nWrote {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
