from fastembed import TextEmbedding
from qdrant_client import QdrantClient

QDRANT_URL = "http://localhost:6333"
COLLECTION = "swarnasky_chunks"
MODEL = "BAAI/bge-large-en-v1.5"

question = "How does Swarnasky help automate insurance claims?"

print("Loading embedding model...")
model = TextEmbedding(model_name=MODEL)

print("Connecting to Qdrant...")
client = QdrantClient(url=QDRANT_URL)

print(f"\nQuestion: {question}\n")

# Convert question into vector
query_vector = list(model.embed([question]))[0]

# Search Qdrant
results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=5,
    with_payload=True,
).points

print("=" * 70)
print("TOP 5 RETRIEVED CHUNKS")
print("=" * 70)

for i, result in enumerate(results, 1):
    payload = result.payload

    print(f"\n#{i}")
    print(f"Score   : {result.score:.4f}")
    print(f"Heading : {payload.get('heading')}")
    print(f"Page    : {payload.get('page_title')}")
    print(f"URL     : {payload.get('url')}")
    text = payload.get("text", "")

    # Remove page/section prefix only for display
    if "Section: " in text:
        text = text.split("Section: ", 1)[1]

    print(f"Text    : {text}")
    
    print("-" * 70)