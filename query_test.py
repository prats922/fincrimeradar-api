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
    "Can one person hold both the SMF16 and SMF17 roles at a small firm?",
    "What happens after I submit my Form A to the FCA?",
    "How does the sanctions and PEP data actually get updated?",
    "If someone changes their password right before a big purchase, is that suspicious?",
    "Does a clean screening result mean the customer is definitely safe?",
    "What's the actual difference between a sanctions hit and a PEP hit?",
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
