import os
import time
import uuid
import logging
from typing import Optional

import ollama

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient


# ============================================================
# CONFIGURATION
# ============================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "swarnasky_chunks"
)

EMBED_MODEL = os.getenv(
    "EMBED_MODEL",
    "BAAI/bge-large-en-v1.5"
)

RERANK_MODEL = os.getenv(
    "RERANK_MODEL",
    "BAAI/bge-reranker-base"
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "llama3.2:3b"
)

RETRIEVAL_K = int(
    os.getenv("RETRIEVAL_K", "15")
)

FINAL_K = int(
    os.getenv("FINAL_K", "3")
)

MAX_OUTPUT_TOKENS = int(
    os.getenv("MAX_OUTPUT_TOKENS", "150")
)

MAX_QUESTION_LENGTH = int(
    os.getenv("MAX_QUESTION_LENGTH", "1000")
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("swarnasky-rag")


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Swarnasky RAG API",
    description="Production RAG API for Swarnasky website data",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# MODELS / CLIENTS
# ============================================================

logger.info("Loading embedding model...")
embedder = TextEmbedding(
    model_name=EMBED_MODEL
)

logger.info("Loading reranker model...")
reranker = TextCrossEncoder(
    model_name=RERANK_MODEL
)

logger.info("Connecting to Qdrant...")
qdrant = QdrantClient(
    url=QDRANT_URL
)

logger.info("RAG components loaded successfully.")


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUESTION_LENGTH,
        description="User question"
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the Swarnasky website assistant.

Answer the user's question ONLY using the provided
Swarnasky website context.

Rules:

1. Use only information explicitly supported by the context.
2. Do not add outside knowledge.
3. Do not invent facts, numbers, features, or claims.
4. Preserve the exact meaning of the source content.
5. Do not combine separate statements if doing so could
   change their meaning.
6. Do not repeat information.
7. Give a concise answer.
8. Use short bullet points when useful.
9. Prefer 3-5 points when multiple points are needed.
10. Be especially careful with:
    minutes, weeks, faster, slower, before, after.
11. Never reverse or alter the meaning of the source.
12. If the context does not support the answer, say exactly:

"I couldn't find that information in the Swarnasky website data."
"""


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "service": "Swarnasky RAG API",
        "status": "running",
        "version": "1.0.0",

        "embedding_model": EMBED_MODEL,
        "reranker_model": RERANK_MODEL,
        "llm_model": LLM_MODEL,

        "retrieval_k": RETRIEVAL_K,
        "final_k": FINAL_K
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    try:

        collection_info = qdrant.get_collection(
            COLLECTION
        )

        return {
            "status": "healthy",
            "qdrant": "connected",
            "collection": COLLECTION,
            "points": collection_info.points_count,

            "embedding_model": EMBED_MODEL,
            "reranker_model": RERANK_MODEL,
            "llm_model": LLM_MODEL
        }

    except Exception as e:

        logger.exception(
            "Health check failed"
        )

        raise HTTPException(
            status_code=503,
            detail="RAG service is not healthy."
        )


# ============================================================
# CHAT
# ============================================================

@app.post("/chat")
def chat(request: ChatRequest):

    request_id = str(
        uuid.uuid4()
    )[:8]

    total_start = time.perf_counter()

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty."
        )

    logger.info(
        "[%s] Question received: %s",
        request_id,
        question
    )

    try:

        # ----------------------------------------------------
        # EMBEDDING
        # ----------------------------------------------------

        embedding_start = time.perf_counter()

        query_vector = list(
            embedder.embed(
                [question]
            )
        )[0]

        embedding_time = (
            time.perf_counter()
            - embedding_start
        )


        # ----------------------------------------------------
        # QDRANT
        # ----------------------------------------------------

        qdrant_start = time.perf_counter()

        results = qdrant.query_points(
            collection_name=COLLECTION,
            query=query_vector,
            limit=RETRIEVAL_K,
            with_payload=True
        ).points

        qdrant_time = (
            time.perf_counter()
            - qdrant_start
        )


        # ----------------------------------------------------
        # NO RESULTS
        # ----------------------------------------------------

        if not results:

            total_time = (
                time.perf_counter()
                - total_start
            )

            return {
                "request_id": request_id,

                "answer":
                    "I couldn't find that information "
                    "in the Swarnasky website data.",

                "sources": [],

                "timing": {
                    "embedding_seconds":
                        round(embedding_time, 3),

                    "qdrant_seconds":
                        round(qdrant_time, 3),

                    "reranker_seconds": 0,

                    "ollama_seconds": 0,

                    "total_seconds":
                        round(total_time, 3)
                },

                "retrieval": {
                    "qdrant_candidates": 0,
                    "final_context_chunks": 0
                }
            }


        # ----------------------------------------------------
        # DOCUMENTS FOR RERANKER
        # ----------------------------------------------------

        documents = []

        for result in results:

            payload = result.payload or {}

            text = payload.get(
                "text",
                ""
            )

            documents.append(text)


        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        rerank_start = time.perf_counter()

        rerank_scores = list(
            reranker.rerank(
                question,
                documents
            )
        )

        rerank_time = (
            time.perf_counter()
            - rerank_start
        )


        # ----------------------------------------------------
        # SORT RESULTS
        # ----------------------------------------------------

        ranked = sorted(
            zip(
                rerank_scores,
                results
            ),
            key=lambda item: item[0],
            reverse=True
        )


        top_results = ranked[
            :FINAL_K
        ]


        # ----------------------------------------------------
        # BUILD CONTEXT
        # ----------------------------------------------------

        context_parts = []

        sources = []

        for rank, (
            rerank_score,
            result
        ) in enumerate(
            top_results,
            1
        ):

            payload = result.payload or {}

            text = payload.get(
                "text",
                ""
            )

            heading = payload.get(
                "heading",
                ""
            )

            page_title = payload.get(
                "page_title",
                ""
            )

            url = payload.get(
                "url",
                ""
            )


            context_parts.append(
                f"""
Source {rank}

Heading:
{heading}

Page:
{page_title}

Content:
{text}
"""
            )


            sources.append({

                "rank": rank,

                "heading": heading,

                "page_title":
                    page_title,

                "url":
                    url,

                "rerank_score":
                    round(
                        float(rerank_score),
                        4
                    )
            })


        context = "\n\n".join(
            context_parts
        )


        # ----------------------------------------------------
        # OLLAMA
        # ----------------------------------------------------

        ollama_start = time.perf_counter()

        response = ollama.chat(

            model=LLM_MODEL,

            messages=[

                {
                    "role":
                        "system",

                    "content":
                        SYSTEM_PROMPT
                },

                {
                    "role":
                        "user",

                    "content":
                        f"""
Swarnasky website context:

{context}

Question:
{question}
"""
                }
            ],

            options={

                "temperature":
                    0.1,

                "num_predict":
                    MAX_OUTPUT_TOKENS
            }
        )


        ollama_time = (
            time.perf_counter()
            - ollama_start
        )


        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        answer = (
            response["message"]["content"]
            .strip()
        )


        total_time = (
            time.perf_counter()
            - total_start
        )


        # ----------------------------------------------------
        # LOG TIMING
        # ----------------------------------------------------

        logger.info(
            "[%s] Embedding: %.3fs | "
            "Qdrant: %.3fs | "
            "Reranker: %.3fs | "
            "Ollama: %.3fs | "
            "Total: %.3fs",

            request_id,

            embedding_time,

            qdrant_time,

            rerank_time,

            ollama_time,

            total_time
        )


        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        return {

            "request_id":
                request_id,

            "answer":
                answer,

            "sources":
                sources,

            "timing": {

                "embedding_seconds":
                    round(
                        embedding_time,
                        3
                    ),

                "qdrant_seconds":
                    round(
                        qdrant_time,
                        3
                    ),

                "reranker_seconds":
                    round(
                        rerank_time,
                        3
                    ),

                "ollama_seconds":
                    round(
                        ollama_time,
                        3
                    ),

                "total_seconds":
                    round(
                        total_time,
                        3
                    )
            },

            "retrieval": {

                "qdrant_candidates":
                    len(results),

                "final_context_chunks":
                    len(top_results)
            }
        }


    except Exception as e:

        logger.exception(
            "[%s] RAG request failed",
            request_id
        )

        raise HTTPException(

            status_code=500,

            detail={
                "message":
                    "RAG request failed.",

                "request_id":
                    request_id
            }
        )