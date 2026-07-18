"""
Retrieval quality test for the guide-chatbot indexing proof of concept.

Loads guide_index.json (built by index_guides.py), embeds a set of sample
visitor questions, and returns the top 3 most similar chunks by cosine
similarity. Prints the actual retrieved text, not just a pass/fail, so
retrieval quality can be judged directly.
"""

import json
import os

import numpy as np
from sentence_transformers import SentenceTransformer

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


def top_k(query_embedding, embeddings, k=3):
    # Both sides are already L2-normalized (index_guides.py encodes with
    # normalize_embeddings=True), so a plain dot product is cosine similarity.
    scores = embeddings @ query_embedding
    top_idx = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in top_idx]


def main():
    model_name, chunks, embeddings = load_index()
    print(f"Loaded {len(chunks)} chunks, model={model_name}\n")
    model = SentenceTransformer(model_name)

    for question in SAMPLE_QUESTIONS:
        query_embedding = model.encode(question, normalize_embeddings=True)
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
