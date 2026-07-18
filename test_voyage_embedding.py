"""
Standalone smoke test for the Voyage AI embeddings API, session 2 setup.

Confirms VOYAGE_API_KEY is valid and a real embedding vector comes back,
before the key is wired into the live /api/chat endpoint. Uses raw HTTP
via `requests` rather than the `voyageai` package, both `requests` and
`python-dotenv` are already in requirements.txt, no new dependency needed.

Never prints the key itself, only whether one was found.
"""

import os

import requests
from dotenv import load_dotenv

load_dotenv()

VOYAGE_API_KEY = os.environ.get("VOYAGE_API_KEY", "").strip()
MODEL = "voyage-4"
TEST_TEXT = "What happens after I submit my Form A to the FCA?"


def main():
    if not VOYAGE_API_KEY:
        raise SystemExit(
            "VOYAGE_API_KEY is not set (or empty) in .env. "
            "Add the real key to fincrimeradar-api/.env and try again."
        )

    print(f"Key found in environment: yes ({len(VOYAGE_API_KEY)} chars, not printed)")
    print(f"Model: {MODEL}")
    print(f"Test string: {TEST_TEXT!r}")
    print()

    response = requests.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {VOYAGE_API_KEY}",
        },
        json={
            "input": [TEST_TEXT],
            "model": MODEL,
            "input_type": "query",
        },
        timeout=30,
    )

    print(f"HTTP status: {response.status_code}")
    response.raise_for_status()
    data = response.json()

    embedding = data["data"][0]["embedding"]
    usage = data.get("usage", {})

    print(f"Response model: {data.get('model')}")
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 8 values: {embedding[:8]}")
    print(f"Token usage: {usage}")
    print()
    print("Voyage embedding API call succeeded.")


if __name__ == "__main__":
    main()
