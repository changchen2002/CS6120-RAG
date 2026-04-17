import os
import json
import time
import logging
from typing import List
import threading
import httpx
import asyncio

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

app = FastAPI(title="Citation RAG")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingModelSingleton:
    _instance = None
    _model = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def get_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    logger.info("Loading embedding model...")
                    start_time = time.time()
                    self._model = SentenceTransformer(EMBEDDING_MODEL)
                    load_time = time.time() - start_time
                    logger.info(f"Embedding model loaded in {load_time:.2f}s")
        return self._model


embedding_service = EmbeddingModelSingleton()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
ARXIV_COLLECTION_NAME = "arxiv_abstracts"
RETRIEVAL_TOP_K = 2
RAG_SCORE_HIGH = float(os.getenv("RAG_SCORE_HIGH", "0.48"))
RAG_SCORE_LOW = float(os.getenv("RAG_SCORE_LOW", "0.30"))
RAG_API_KEY = os.getenv("RAG_API_KEY", "").strip()

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(x_api_key: str | None = Depends(_api_key_header)) -> None:
    if not RAG_API_KEY:
        return
    if not x_api_key or x_api_key != RAG_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")


class QueryRequest(BaseModel):
    question: str


def build_sources(hits):
    best: dict = {}
    for hit in hits:
        payload = hit.payload or {}
        title = payload.get("title", "Untitled")
        url = (payload.get("link") or payload.get("source_url") or "").strip()
        passage = payload.get("text", "") or ""
        if url:
            key = ("url", url, title.strip())
        else:
            key = ("txt", title.strip(), passage[:400])
        score = getattr(hit, "score", None)
        try:
            score_f = float(score) if score is not None else None
        except (TypeError, ValueError):
            score_f = None
        row = {
            "title": title,
            "url": url,
            "passage": passage,
            "similarity_score": round(score_f, 4) if score_f is not None else None,
        }
        prev = best.get(key)
        if prev is None or (score_f is not None and (prev[0] is None or score_f > prev[0])):
            best[key] = (score_f, row)
    return [v[1] for v in best.values()]


def retrieval_confidence_from_scores(scores: List[float]) -> tuple:
    if not scores:
        return "unknown", 0.0, 0.0
    mx = max(scores)
    mean = sum(scores) / len(scores)
    if mx >= RAG_SCORE_HIGH:
        label = "high"
    elif mx >= RAG_SCORE_LOW:
        label = "medium"
    else:
        label = "low"
    return label, mx, mean


def collection_exists(collection_name: str) -> bool:
    collections = qdrant.get_collections().collections
    return collection_name in {col.name for col in collections}


@app.on_event("startup")
async def startup_event():
    logger.info("Starting up application...")
    try:
        embed_model = embedding_service.get_model()
        embed_model.encode("warmup test")
        logger.info("Embedding model warmed up successfully")
        qdrant.get_collections()
        logger.info(f"Qdrant connected")
    except Exception as e:
        logger.error(f"Startup error: {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "Citation RAG",
        "api_key_required": bool(RAG_API_KEY),
        "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}",
        "ollama": f"{OLLAMA_HOST}:{OLLAMA_PORT}",
    }


@app.post("/query_stream")
async def query_rag_stream(
    req: QueryRequest,
    _: None = Depends(require_api_key),
):
    async def generate_stream():
        question = req.question.strip()
        if not question:
            yield "data: " + json.dumps({"error": "⚠️ Please enter a valid question."}) + "\n\n"
            return

        logger.info(f"Question received: '{question[:50]}...'")

        try:
            embed_model = embedding_service.get_model()
            query_vector = embed_model.encode(question).tolist()
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            yield "data: " + json.dumps({"error": f"⚠️ Embedding error: {e}"}) + "\n\n"
            return

        if not collection_exists(ARXIV_COLLECTION_NAME):
            yield "data: " + json.dumps(
                {
                    "error": f"⚠️ Qdrant collection '{ARXIV_COLLECTION_NAME}' not found. "
                    "Run ingest_arxiv.py (see README)."
                }
            ) + "\n\n"
            return

        try:
            hits = qdrant.search(
                collection_name=ARXIV_COLLECTION_NAME,
                query_vector=query_vector,
                limit=RETRIEVAL_TOP_K,
                query_filter=None,
            )
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            yield "data: " + json.dumps({"error": f"⚠️ Search error: {e}"}) + "\n\n"
            return

        if not hits:
            yield "data: " + json.dumps({"error": "⚠️ No context found for this question."}) + "\n\n"
            return

        scores = []
        for h in hits:
            try:
                scores.append(float(h.score))
            except (TypeError, ValueError, AttributeError):
                pass
        conf_label, max_sim, mean_sim = retrieval_confidence_from_scores(scores)

        sources = build_sources(hits)
        retrieval_meta = {
            "confidence": conf_label,
            "max_similarity": round(max_sim, 4),
            "mean_similarity": round(mean_sim, 4),
            "thresholds": {"high_at_least": RAG_SCORE_HIGH, "medium_at_least": RAG_SCORE_LOW},
            "instructor_note": (
                "Low retrieval similarity: compare each sentence of the answer to the passages below; "
                "unsupported claims are likely hallucinations."
                if conf_label == "low"
                else "Verify non-obvious claims against the cited passages (similarity scores are approximate)."
            ),
        }
        yield "data: " + json.dumps({"sources": sources, "retrieval": retrieval_meta}) + "\n\n"

        context = "\n\n".join([hit.payload["text"] for hit in hits])
        abstain_rules = (
            "Rules: Use ONLY the context. If the context is insufficient, unrelated, or ambiguous, say so clearly "
            "in 2–3 sentences—do not invent citations, numbers, or details. "
            "If you are uncertain, say you are uncertain. "
        )
        if conf_label == "low":
            abstain_rules += (
                "Retrieval similarity to this question is LOW; prefer stating that you cannot answer from the given passages. "
            )
        prompt = (
            f"{abstain_rules}"
            "Respond in exactly 2 or 3 complete sentences. Do not use bullet points or headings.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer (2-3 sentences):"
        )

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
                    json={
                        "model": "phi3:3.8b-mini-128k-instruct-q4_0",
                        "prompt": prompt,
                        "stream": True,
                        "options": {"num_predict": 220},
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                part = chunk.get("response")
                                if part:
                                    yield f"data: {json.dumps({'chunk': part})}\n\n"
                                    await asyncio.sleep(0.01)
                                if chunk.get("done"):
                                    yield "data: " + json.dumps(
                                        {"done": True, "sources": sources, "retrieval": retrieval_meta}
                                    ) + "\n\n"
                                    break
                            except Exception as e:
                                logger.error(f"Chunk parse error: {e}")
                                continue
        except Exception as e:
            logger.error(f"Ollama request error: {e}")
            yield "data: " + json.dumps({"error": f"⚠️ Ollama error: {e}"}) + "\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
