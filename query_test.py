"""
Retrieval quality test for the guide chatbot's indexed content.

Loads guide_index.json (built by index_guides.py), embeds a set of sample
visitor questions via Voyage (matching the corpus's embedding model, see
index_guides.py's docstring), and returns the top 3 most similar chunks
by cosine similarity. Prints the actual retrieved text, not just a
pass/fail, so retrieval quality can be judged directly.
"""

import json
import os
import time

import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

INDEX_PATH = os.path.join(os.path.dirname(__file__), "guide_index.json")

SAMPLE_QUESTIONS = [
    # UBO Investigation Handbook
    "How do I identify the ultimate beneficial owner behind layered corporate structures?",
    # False Positive Playbook
    "Why do sanctions screening tools generate so many false positives?",
    # Crypto Guide (mixers/layering, part 3)
    "How do criminals use crypto mixers and tumblers to launder funds?",
    # AML Guide (part 1, penalties)
    "What are the penalties for failing to comply with the UK Money Laundering Regulations?",
    # Fraud Red Flags, re-confirm post-reindex (tested in session 1)
    "What are the warning signs of an authorised push payment scam?",
    # scenario-lab.html tool-context, new source this session
    "What does Scenario Lab actually train you to do?",
    # Collision test: UK has no fixed SAR threshold for structuring, unlike
    # the US CTR's $10,000 line. Fraud Red Flags, AML guide, and SAR guide
    # all touch structuring, this checks retrieval doesn't blur them or
    # import a US-style fixed-threshold answer that isn't true for the UK.
    "Is there a fixed pound threshold that triggers a Suspicious Activity Report for structuring in the UK?",
    # PEP Guide part 2, new sow-title/sow-desc/sow-eg extraction this session
    "What's the difference between source of funds and source of wealth for a PEP?",
    # TM Guide, new source this session (also had simulator/rule-anatomy stripped)
    "How many false positives does a typical transaction monitoring system generate?",
]


def load_index():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    embeddings = np.array([c["embedding"] for c in data["chunks"]], dtype=np.float32)
    return data["model"], data["chunks"], embeddings


def embed_query(text, model_name):
    api_key = os.environ.get("VOYAGE_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("VOYAGE_API_KEY is not set (or empty) in .env.")
    for attempt in range(5):
        response = requests.post(
            "https://api.voyageai.com/v1/embeddings",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            # input_type="query" pairs with the corpus's
            # input_type="document", Voyage prepends a different internal
            # instruction to each side.
            json={"input": [text], "model": model_name, "input_type": "query"},
            timeout=30,
        )
        if response.status_code == 429:
            wait = int(response.headers.get("retry-after", 20 * (attempt + 1)))
            print(f"  rate limited, waiting {wait}s (attempt {attempt + 1}/5)...")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"], dtype=np.float32)
    raise SystemExit("Voyage rate limit persisted after 5 retries, aborting.")


def top_k(query_embedding, embeddings, k=3):
    # Voyage embeddings are normalized to length 1, so a plain dot product
    # is cosine similarity.
    scores = embeddings @ query_embedding
    top_idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def main():
    model_name, chunks, embeddings = load_index()
    print(f"Loaded {len(chunks)} chunks, model={model_name}\n")

    for question in SAMPLE_QUESTIONS:
        query_embedding = embed_query(question, model_name)
        results = top_k(query_embedding, embeddings, k=3)

        print("=" * 100)
        print(f"Q: {question}")
        print("-" * 100)
        for rank, (idx, score) in enumerate(results, start=1):
            c = chunks[idx]
            print(f"  #{rank} score={score:.3f} source={c['source_id']} heading=({c['heading']})")
            print(f"      {c['text']}")
            print()


if __name__ == "__main__":
    main()
