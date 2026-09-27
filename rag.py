from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
import ollama


QDRANT_URL = "http://localhost:6333"
COLLECTION = "swarnasky_chunks"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-base"
LLM_MODEL = "llama3.2:3b"

RETRIEVAL_K = 15
FINAL_K = 5


embedder = TextEmbedding(model_name=EMBED_MODEL)

reranker = TextCrossEncoder(
    model_name=RERANK_MODEL
)

client = QdrantClient(
    url=QDRANT_URL
)


question = input("\nAsk Swarnasky: ").strip()


# --------------------------------------------------
# 1. Embed question
# --------------------------------------------------

query_vector = list(
    embedder.embed([question])
)[0]


# --------------------------------------------------
# 2. Retrieve candidates from Qdrant
# --------------------------------------------------

results = client.query_points(
    collection_name=COLLECTION,
    query=query_vector,
    limit=RETRIEVAL_K,
    with_payload=True,
).points


# --------------------------------------------------
# 3. Prepare documents for reranking
# --------------------------------------------------

documents = [
    (r.payload or {}).get("text", "")
    for r in results
]


# --------------------------------------------------
# 4. Rerank
# --------------------------------------------------

rerank_scores = list(
    reranker.rerank(
        question,
        documents,
    )
)


ranked = sorted(
    zip(rerank_scores, results),
    key=lambda x: x[0],
    reverse=True,
)


# --------------------------------------------------
# 5. Select final context
# --------------------------------------------------

top_results = ranked[:FINAL_K]


context_parts = []
sources = []

for rank, (score, result) in enumerate(
    top_results,
    1
):

    payload = result.payload or {}

    text = payload.get("text", "")
    heading = payload.get("heading", "")
    page_title = payload.get("page_title", "")
    url = payload.get("url", "")

    context_parts.append(
        f"""
Source {rank}
Heading: {heading}
Page: {page_title}

{text}
"""
    )

    sources.append({
        "heading": heading,
        "page_title": page_title,
        "url": url,
        "rerank_score": float(score),
    })


context = "\n\n".join(context_parts)


# --------------------------------------------------
# 6. Generate answer with Ollama
# --------------------------------------------------

system_prompt = """
You are the Swarnasky website assistant.

Answer the user's question ONLY using the provided
Swarnasky website context.

Do not use outside knowledge.

If the answer is not supported by the context, say:

"I couldn't find that information in the Swarnasky website data."

Give a concise, clear answer.

Do not repeat the same information unnecessarily.
"""


response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": f"""
Swarnasky website context:

{context}

Question:
{question}
""",
        },
    ],
)


answer = response["message"]["content"]


# --------------------------------------------------
# 7. Output
# --------------------------------------------------

print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)

print(answer)

print("\n" + "=" * 70)
print("SOURCES")
print("=" * 70)

for i, source in enumerate(sources, 1):

    print(f"\n{i}. {source['heading']}")
    print(f"   Page : {source['page_title']}")
    print(f"   URL  : {source['url']}")
    print(f"   Score: {source['rerank_score']:.4f}")