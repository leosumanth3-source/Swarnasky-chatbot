#!/usr/bin/env python3
"""
Production-oriented Swarnasky embedding -> Qdrant ingestion.

Reads:
    swarnasky_chunks/chunks.jsonl

Uses:
    BAAI/bge-large-en-v1.5 via FastEmbed

Qdrant connection:
    Set QDRANT_URL
    Optional: QDRANT_API_KEY

Example local:
    export QDRANT_URL=http://localhost:6333

Example Cloud:
    export QDRANT_URL="https://YOUR-CLUSTER.cloud.qdrant.io"
    export QDRANT_API_KEY="YOUR_API_KEY"
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

CHUNKS_FILE = Path("swarnasky_chunks/chunks.jsonl")
COLLECTION = "swarnasky_chunks"
MODEL_NAME = "BAAI/bge-large-en-v1.5"
BATCH_SIZE = 32


def point_id(chunk_id: str) -> int:
    # Deterministic UUID-like 63-bit integer for Qdrant compatibility.
    digest = hashlib.sha256(chunk_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def load_chunks():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(f"Missing {CHUNKS_FILE}")

    rows = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc

            text = str(row.get("text", "")).strip()
            chunk_id = str(row.get("chunk_id", "")).strip()

            if not chunk_id or not text:
                print(f"WARNING: skipping incomplete chunk on line {line_no}")
                continue

            row["_point_id"] = point_id(chunk_id)
            rows.append(row)

    if not rows:
        raise ValueError("No valid chunks found.")

    return rows


def make_client():
    url = os.getenv("QDRANT_URL")
    api_key = os.getenv("QDRANT_API_KEY")

    if not url:
        print("ERROR: QDRANT_URL is not set.")
        print("For local Qdrant:")
        print("  export QDRANT_URL=http://localhost:6333")
        print("For Qdrant Cloud:")
        print("  export QDRANT_URL='https://YOUR-CLUSTER.cloud.qdrant.io'")
        print("  export QDRANT_API_KEY='YOUR_API_KEY'")
        sys.exit(1)

    return QdrantClient(url=url, api_key=api_key)


def main():
    rows = load_chunks()
    print(f"Loaded chunks : {len(rows)}")
    print(f"Embedding     : {MODEL_NAME}")

    model = TextEmbedding(model_name=MODEL_NAME)
    embedding_size = len(next(iter(model.embed(["dimension check"]))))

    print(f"Vector size   : {embedding_size}")

    client = make_client()

    if client.collection_exists(COLLECTION):
        info = client.get_collection(COLLECTION)
        print(f"Collection    : {COLLECTION} already exists")
        print(f"Existing pts  : {info.points_count}")
    else:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(
                size=embedding_size,
                distance=models.Distance.COSINE,
            ),
        )
        print(f"Created       : {COLLECTION}")

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [r["text"] for r in batch]
        vectors = list(model.embed(texts))

        points = []
        for row, vector in zip(batch, vectors):
            payload = {
                "chunk_id": row["chunk_id"],
                "page_id": row.get("page_id"),
                "url": row.get("url"),
                "page_title": row.get("page_title"),
                "section_id": row.get("section_id"),
                "heading": row.get("heading"),
                "level": row.get("level"),
                "section_path": row.get("section_path", []),
                "text": row["text"],
                "word_count": row.get("word_count"),
                "source_file": row.get("source_file"),
            }

            points.append(
                models.PointStruct(
                    id=row["_point_id"],
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        client.upsert(
            collection_name=COLLECTION,
            points=points,
            wait=True,
        )

        print(
            f"Uploaded {min(start + len(batch), len(rows)):>3}/{len(rows)}"
        )

    info = client.get_collection(COLLECTION)
    print("\n" + "=" * 52)
    print("EMBEDDING + QDRANT INGESTION COMPLETE")
    print("=" * 52)
    print(f"Collection    : {COLLECTION}")
    print(f"Model         : {MODEL_NAME}")
    print(f"Vector size   : {embedding_size}")
    print(f"Expected pts  : {len(rows)}")
    print(f"Qdrant points : {info.points_count}")

    if info.points_count != len(rows):
        raise RuntimeError(
            f"Point count mismatch: expected {len(rows)}, got {info.points_count}"
        )

    print("Verification   : PASS")


if __name__ == "__main__":
    main()
