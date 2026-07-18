"""
Content indexing pipeline, proof of concept (session 1).

Parses a small, deliberately mixed set of guide HTML files from the
fincrimeradar-repo site, strips nav/footer/cheat-sheet/decision-scenario
markup, extracts clean article prose, chunks it at paragraph level,
embeds each chunk, and writes chunks + embeddings to a flat JSON file.

Scope: 4 source pages only, to prove the parser handles different HTML
structures (scenario-based guide, reference page, tool page) before
indexing the full site. See BACKLOG.md in fincrimeradar-repo, "Guide
chatbot" entry, for the full-site plan (session 2+).
"""

import json
import os
import re

from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

GUIDES_DIR = os.environ.get(
    "GUIDES_DIR",
    os.path.join(os.path.dirname(__file__), "..", "fincrimeradar-repo"),
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "guide_index.json")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Classes that mark interactive/non-prose widgets (quizzes, self-assessment
# tools, decision-scenario diagrams, cheat-sheet SVGs, nav toggles). These
# are removed wholesale before prose extraction, they are not article
# content and would pollute retrieval with UI copy and quiz option text.
STRIP_CLASSES = [
    "quiz-q", "quiz-opts", "quiz-feedback",
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
]
STRIP_TAGS = ["script", "style", "nav", "footer", "button", "svg"]

# Prose that lives in <div> elements rather than the standard h2-h4/p/li
# tags. Heading-role divs update the running "current heading" the same
# way an h3 would; text-role divs are treated as paragraph chunks.
DIV_HEADING_CLASSES = {"scenario-name", "step-title"}
DIV_TEXT_CLASSES = {"step-body", "callout-text", "highlight"}

# Each source defines which container(s) hold article prose, and a label
# used in chunk metadata for citation back to the source page.
SOURCES = [
    {
        "id": "mlro-handbook-part1",
        "file": "mlro-handbook-part1.html",
        "url": "/mlro-handbook-part1.html",
        "select": (True, {"class": "article-section"}),
    },
    {
        "id": "fraud-red-flags-guide",
        "file": "fraud-red-flags-guide.html",
        "url": "/fraud-red-flags-guide.html",
        "select": (True, {"class": "article-section"}),
    },
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
        # Some prose lives in divs, not p/h2-h4: fraud-red-flags-guide's
        # per-scenario headers are a scenario-num + scenario-name div pair,
        # and mlro-handbook's approval-process steps use step-title/
        # step-body divs instead of h3/p. Missing these would silently drop
        # real article content (the 5-step FCA approval process, in
        # step-body's case) rather than just mislabeling it.
        # A single find_all pass keeps everything in document order, two
        # separate passes concatenated would put all div matches after all
        # p/li matches, breaking heading tracking below.
        matches = container.find_all(lambda t: t.name in ("h2", "h3", "h4", "p", "li")
                                      or (t.name == "div" and set(t.get("class") or [])
                                          & (DIV_HEADING_CLASSES | DIV_TEXT_CLASSES)))
        for el in matches:
            text = normalize_text(el.get_text(" ", strip=True))
            if not text:
                continue
            classes = set(el.get("class") or [])
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


def main():
    print(f"Reading guides from: {os.path.abspath(GUIDES_DIR)}")
    all_chunks = []
    for source in SOURCES:
        chunks = extract_chunks(source)
        print(f"  {source['file']}: {len(chunks)} paragraph chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        raise SystemExit("No chunks extracted, aborting before embedding.")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Embedding {len(all_chunks)} chunks...")
    texts = [f"{c['heading']}. {c['text']}" if c["heading"] else c["text"] for c in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, embedding, chunk_id in zip(all_chunks, embeddings, range(len(all_chunks))):
        chunk["id"] = chunk_id
        chunk["embedding"] = embedding.tolist()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": EMBEDDING_MODEL,
            "dim": len(embeddings[0]),
            "chunks": all_chunks,
        }, f, ensure_ascii=False, indent=None)

    print(f"\nWrote {len(all_chunks)} chunks to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
