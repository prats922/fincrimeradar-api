"""
Guide chatbot route for the fincrimeradar-api service.

Loads guide_index.json (built by index_guides.py) once at import time,
embeds incoming questions via Voyage (same model as the corpus, see
index_guides.py's docstring for why that has to match), retrieves the
top matching chunks, and asks Claude Haiku 4.5 to answer strictly from
that retrieved context. Hard constraint: never answer from the model's
own general knowledge, only from what was actually retrieved, with a
visible fallback when nothing relevant was found.

Wire into main.py with:

    from routes_guide_chat import router as guide_chat_router
    app.include_router(guide_chat_router)

CORS: reuses the same origin allowlist already configured in main.py's
CORSMiddleware, this file adds no origins of its own. main.py's
allow_methods needed POST added (it was GET-only, every other route
here is GET).
"""

import json
import os
import time
from collections import deque
from pathlib import Path

import numpy as np
import requests
from anthropic import Anthropic
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

load_dotenv()

router = APIRouter()

INDEX_PATH = Path(__file__).parent / "guide_index.json"
EMBEDDING_MODEL = "voyage-4-lite"
GENERATION_MODEL = "claude-haiku-4-5"

# Top 6, not top 3: session 2's retrieval testing found a case
# (AML-penalties question) where a chunk that just literally named
# "Money Laundering Regulations" outranked the actual penalty-scale
# content, literal keyword overlap beating semantic relevance. Logged
# in BACKLOG.md under "Guide chatbot" as a Step 3 design requirement.
# Pulling more candidates and letting the model synthesize across all of
# them, rather than leaning on the single top match, is the mitigation.
TOP_K = 6

# Confirmed-relevant results scored 0.5-0.76 across session 1 and 2's
# testing (all real questions against the 4-page and then full-site
# index). Starting the cutoff meaningfully below that range, weak or
# off-topic matches should sit well under 0.4, not just slightly under
# a confirmed-relevant score.
SIMILARITY_THRESHOLD = 0.4

# Starting number, not a permanent decision. Based on the site's actual
# traffic at the time this was built (roughly 600-700 sessions per
# 28-day period per the last GA4 pull), not a load-tested ceiling.
# Revisit once this endpoint has real usage data of its own, in-memory
# storage also means this resets on every Render restart/redeploy and
# is per-instance only, both accepted at this scale, same tradeoff
# routes_scenario_lab.py already makes for its own rate limiter.
RATE_LIMIT_MAX_MESSAGES = 20
RATE_LIMIT_WINDOW_SECONDS = 24 * 60 * 60

KNOWLEDGE_HUB_LINK = "/knowledge.html"
CONTACT_LINK = "/contact.html"

SUMMARY_PATTERNS = (
    "summar", "tl;dr", "tldr", "give me the gist", "the gist",
    "recap", "in short", "brief overview", "overview of this page",
    "what is this page about", "what's this page about",
    "explain this page",
)

SYSTEM_PROMPT = """You are FinCrimeRadar's guide assistant. You answer questions using ONLY the context excerpts provided in the user message, drawn from FinCrimeRadar's own published guides.

Hard rule, the one this entire feature depends on: never use general knowledge, training data, or anything not explicitly stated in the provided context, even if you are confident it is correct. If the context is thin, contradictory, or simply doesn't cover the question, that is a valid outcome, say so, do not fill the gap from what you already know.

For every question:
1. Read the provided context excerpts.
2. If they actually answer the question, set answered_from_context to true and write a clear, direct answer using only facts stated in that context. Do not add examples, caveats, or elaboration the source material doesn't contain.
3. If they don't answer the question, or only tangentially relate to it, set answered_from_context to false and leave response empty or write one sentence noting the guides don't cover this.
4. Never mention "the context" or "the provided excerpts" to the user, write as if you simply know the answer from FinCrimeRadar's guides.
5. Keep answers concise, a few sentences unless the question genuinely requires more."""

SUMMARY_SYSTEM_PROMPT = """You are FinCrimeRadar's guide assistant. The user wants a summary of the page they're currently reading. The full indexed text of that page is provided in the user message.

Hard rule: summarize only what is in the provided text, never add outside knowledge or general commentary not grounded in that text.

Write a concise summary, a short paragraph or a few bullet points, covering the main points of the page. Set answered_from_context to true, this endpoint always has content to summarize when this branch runs."""


class GuideAnswer(BaseModel):
    answered_from_context: bool
    response: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    page_context: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    response: str
    suggested_link: str | None = None


def _load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"guide_index.json not found at {INDEX_PATH}. Run index_guides.py first."
        )
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if data["model"] != EMBEDDING_MODEL:
        raise ValueError(
            f"guide_index.json was built with model {data['model']!r}, "
            f"this route embeds queries with {EMBEDDING_MODEL!r}. They must "
            f"match, mismatched embedding spaces produce meaningless "
            f"similarity scores (or a shape-mismatch crash if dimensions differ)."
        )
    embeddings = np.array([c["embedding"] for c in data["chunks"]], dtype=np.float32)
    return data["chunks"], embeddings


# Load once at import time, not per-request, matching routes_scenario_lab.py's
# pattern. 1148 chunks plus their 1024-dim embeddings is small enough to
# keep resident in memory for the process lifetime.
try:
    _CHUNKS, _EMBEDDINGS = _load_index()
    _LOAD_ERROR = None
except (FileNotFoundError, ValueError) as exc:
    _CHUNKS, _EMBEDDINGS = [], None
    _LOAD_ERROR = str(exc)

_anthropic_client = Anthropic()

# Per session_id, a deque of request timestamps within the rolling
# window. In-memory, single-process, resets on restart, see
# RATE_LIMIT_MAX_MESSAGES comment above for why that's accepted here.
_request_log = {}


def _rate_limited(session_id: str) -> bool:
    now = time.monotonic()
    log = _request_log.setdefault(session_id, deque())

    while log and now - log[0] > RATE_LIMIT_WINDOW_SECONDS:
        log.popleft()

    if len(log) >= RATE_LIMIT_MAX_MESSAGES:
        return True

    log.append(now)
    return False


def _normalize_url(url: str) -> str:
    return url.split("#")[0].rstrip("/")


def _looks_like_summary_request(message: str) -> bool:
    lowered = message.lower()
    return any(pattern in lowered for pattern in SUMMARY_PATTERNS)


def _page_chunks(page_context: str):
    target = _normalize_url(page_context)
    return [c for c in _CHUNKS if _normalize_url(c["url"]) == target]


def _embed_query(text: str) -> np.ndarray:
    api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set.")
    response = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        # input_type="query" pairs with the corpus's input_type="document"
        # (index_guides.py), Voyage prepends a different internal
        # instruction to each side, using the wrong one degrades retrieval.
        json={"input": [text], "model": EMBEDDING_MODEL, "input_type": "query"},
        timeout=15,
    )
    response.raise_for_status()
    return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)


def _top_k(query_embedding: np.ndarray, k: int):
    # Voyage embeddings are normalized to length 1, so a plain dot product
    # is cosine similarity.
    scores = _EMBEDDINGS @ query_embedding
    top_idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def _format_chunks_as_context(chunks) -> str:
    parts = []
    for c in chunks:
        label = f"[{c['heading']}]" if c.get("heading") else ""
        parts.append(f"{label} {c['text']}".strip())
    return "\n\n".join(parts)


def _ask_claude(system_prompt: str, question: str, context: str) -> GuideAnswer:
    response = _anthropic_client.messages.create(
        model=GENERATION_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"Context:\n\n{context}\n\nQuestion: {question}",
        }],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {
                        "answered_from_context": {"type": "boolean"},
                        "response": {"type": "string"},
                    },
                    "required": ["answered_from_context", "response"],
                    "additionalProperties": False,
                },
            }
        },
    )
    text = next(b.text for b in response.content if b.type == "text")
    parsed = json.loads(text)
    return GuideAnswer(**parsed)


def _fallback_response() -> ChatResponse:
    return ChatResponse(
        response=(
            "That's not something covered in FinCrimeRadar's guides. Try the "
            f"Knowledge Hub search for related content ({KNOWLEDGE_HUB_LINK}), "
            f"or reach out via the contact page ({CONTACT_LINK}) if you think "
            "this should be covered."
        ),
        suggested_link=KNOWLEDGE_HUB_LINK,
    )


def _service_error_response() -> JSONResponse:
    # Deliberately distinct from _fallback_response(): a Claude call
    # failing (auth, rate limit, malformed output) is a real backend
    # problem, not "this isn't covered in the guides", the "not covered"
    # text would misrepresent a service outage as a content gap. Same
    # ChatResponse shape (response + suggested_link) as every other
    # response, just wrapped in a 503 JSONResponse since returning
    # ChatResponse normally would carry the route's default 200 status.
    return JSONResponse(
        status_code=503,
        content=ChatResponse(
            response=(
                "Something went wrong answering that on our end, this isn't "
                "about whether the guides cover it. Please try again in a "
                "moment."
            ),
            suggested_link=None,
        ).model_dump(),
    )


@router.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if _EMBEDDINGS is None:
        print(f"Guide chat route error: {_LOAD_ERROR}")
        raise HTTPException(status_code=503, detail="Guide chat is temporarily unavailable.")

    if _rate_limited(req.session_id):
        raise HTTPException(
            status_code=429,
            detail=f"Message limit reached ({RATE_LIMIT_MAX_MESSAGES} per 24 hours). Try again later.",
        )

    message = req.message.strip()

    if _looks_like_summary_request(message):
        page_chunks = _page_chunks(req.page_context)
        if not page_chunks:
            return _fallback_response()
        context = _format_chunks_as_context(page_chunks)
        try:
            answer = _ask_claude(SUMMARY_SYSTEM_PROMPT, message, context)
        except Exception as exc:
            print(f"Guide chat generation error (summary): {exc}")
            return _service_error_response()
        return ChatResponse(response=answer.response, suggested_link=None)

    try:
        query_embedding = _embed_query(message)
    except (requests.RequestException, RuntimeError) as exc:
        print(f"Guide chat embedding error: {exc}")
        raise HTTPException(status_code=503, detail="Guide chat is temporarily unavailable.")

    top = _top_k(query_embedding, TOP_K)
    top_score = top[0][1] if top else 0.0

    if top_score < SIMILARITY_THRESHOLD:
        return _fallback_response()

    retrieved = [_CHUNKS[idx] for idx, _score in top]
    context = _format_chunks_as_context(retrieved)
    try:
        answer = _ask_claude(SYSTEM_PROMPT, message, context)
    except Exception as exc:
        print(f"Guide chat generation error: {exc}")
        return _service_error_response()

    if not answer.answered_from_context:
        return _fallback_response()

    return ChatResponse(response=answer.response, suggested_link=None)
