from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient


QDRANT_URL = "http://localhost:6333"
COLLECTION = "swarnasky_chunks"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"

QUESTION = "How does Swarnasky help automate insurance claims?"

print("Loading embedding model...")
embedder = TextEmbedding(model_name=EMBED_MODEL)

print("Loading reranker...")
reranker = TextCrossEncoder(model_name=RERANK_MODEL)

print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)


# --------------------------------------------------
# 1. Embed question
# --------------------------------------------------

query_vector = list(
    embedder.embed([QUESTION])
)[0]


# --------------------------------------------------
# 2. Retrieve more candidates from Qdrant
# --------------------------------------------------

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=15,
    with_payload=True,
).points

print(f"\nQdrant candidates: {len(results)}")


# --------------------------------------------------
# 3. Prepare documents for reranking
# --------------------------------------------------

documents = []

for result in results:
    payload = result.payload or {}
    documents.append(payload.get("text", ""))


# --------------------------------------------------
# 4. Cross-encoder reranking
# --------------------------------------------------

reranked = list(
    reranker.rerank(
        QUESTION,
        documents,
    )
)


# --------------------------------------------------
# 5. Sort by reranker score
# --------------------------------------------------

ranked = sorted(
    zip(reranked, results),
    key=lambda x: x[0],
    reverse=True,
)


# --------------------------------------------------
# 6. Show top 5
# --------------------------------------------------

print("\n" + "=" * 70)
print("TOP 5 AFTER RERANKING")
print("=" * 70)

for rank, (rerank_score, result) in enumerate(
    ranked[:5], 1
):

    payload = result.payload or {}

    print(f"\n#{rank}")
    print(f"Rerank score : {rerank_score:.4f}")
    print(f"Qdrant score : {result.score:.4f}")
    print(f"Heading      : {payload.get('heading')}")
    print(f"Page         : {payload.get('page_title')}")
    print(f"URL          : {payload.get('url')}")
    print(f"Text         : {payload.get('text')}")
    print("-" * 70)