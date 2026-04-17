import os
import uuid, datetime
from datetime import datetime, timezone
from typing import List
import json
import time
import logging
import threading
import httpx
import asyncio

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from sklearn.cluster import KMeans
from qdrant_client.http.models import Distance, VectorParams
from fastapi.responses import StreamingResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Singleton for embedding model to avoid reloading
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

# Initialize services
embedding_service = EmbeddingModelSingleton()
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost")
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
PDF_COLLECTION_NAME = "RAG-3.0"
ARXIV_COLLECTION_NAME = "arxiv_abstracts"
UPLOADED_FILES_COLLECTION_NAME = "uploaded_files"
# Top-k retrieved passages for the LLM (arxiv: 2 abstracts; PDF mode uses same cap)
RETRIEVAL_TOP_K = 2
# Cosine similarity from Qdrant (higher = closer match). Tune for your embedding space.
RAG_SCORE_HIGH = float(os.getenv("RAG_SCORE_HIGH", "0.48"))
RAG_SCORE_LOW = float(os.getenv("RAG_SCORE_LOW", "0.30"))
# Optional: set RAG_API_KEY in production so only clients with header X-API-Key can call protected routes.
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
    filenames: List[str]

class DeleteRequest(BaseModel):
    filenames: List[str]


def ensure_collection(collection_name: str, vector_size: int) -> None:
    collections = qdrant.get_collections().collections
    collection_names = [col.name for col in collections]
    if collection_name not in collection_names:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )


def build_sources(hits):
    """
    De-duplicate sources; keep highest Qdrant similarity score per key.
    Instructors/TAs can compare the answer to cited passages and similarity_score to spot unsupported claims.
    """
    best: dict = {}
    for hit in hits:
        payload = hit.payload or {}
        title = payload.get("title") or payload.get("filename", "Untitled")
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
    """Return (label, max_score, mean_score) for transparency."""
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
    """Warm up the services on startup"""
    logger.info("Starting up application...")
    try:
        # Warm up embedding model
        embed_model = embedding_service.get_model()
        test_embedding = embed_model.encode("warmup test")
        logger.info("Embedding model warmed up successfully")
        
        # Test Qdrant connection
        collections = qdrant.get_collections()
        logger.info(f"Qdrant connected, found {len(collections.collections)} collections")
    except Exception as e:
        logger.error(f"Startup error: {e}")

@app.get("/health")
def health():
    """For uptime checks, load balancers, and verifying deployment (no API key)."""
    return {
        "status": "ok",
        "service": "local-rag-engine",
        "api_key_required": bool(RAG_API_KEY),
        "qdrant": f"{QDRANT_HOST}:{QDRANT_PORT}",
        "ollama": f"{OLLAMA_HOST}:{OLLAMA_PORT}",
    }


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    _: None = Depends(require_api_key),
):
    start_time = time.time()
    logger.info(f"Upload started for file: {file.filename}")
    
    try:
        pdf = PdfReader(file.file)
        raw_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                raw_text += "\n" + text
        
        logger.info(f"PDF read in {time.time() - start_time:.2f}s")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=150
        )
        chunks = splitter.split_text(raw_text)
        unique_chunks = list(set(chunk.strip() for chunk in chunks if len(chunk.strip()) > 30))
        
        logger.info(f"Text splitting completed, {len(unique_chunks)} unique chunks")
        
        embed_start = time.time()
        embed_model = embedding_service.get_model()
        embeddings = embed_model.encode(unique_chunks, batch_size=16)
        logger.info(f"Embeddings generated in {time.time() - embed_start:.2f}s")
        
        # Clustering
        num_clusters = min(10, len(embeddings) // 5)
        if num_clusters > 0:
            kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(embeddings)
            cluster_ids = kmeans.predict(embeddings)
        else:
            cluster_ids = [0] * len(embeddings)
        
        points = []
        now = datetime.now(timezone.utc).isoformat()

        for chunk, embedding, cluster_id in zip(unique_chunks, embeddings, cluster_ids):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding.tolist(),
                payload={
                    "text": chunk,
                    "filename": file.filename,
                    "cluster_id": int(cluster_id),
                    "uploaded_at": now
                }
            ))
        
        # Create collections if needed
        collections = qdrant.get_collections().collections
        collection_names = [col.name for col in collections]

        ensure_collection(PDF_COLLECTION_NAME, len(embeddings[0]))
        
        qdrant.upsert(
            collection_name=PDF_COLLECTION_NAME,
            points=points
        )
        
        # Handle uploaded_files collection
        if UPLOADED_FILES_COLLECTION_NAME not in collection_names:
            qdrant.create_collection(
                collection_name=UPLOADED_FILES_COLLECTION_NAME,
                vectors_config=VectorParams(
                size=10,
                distance=Distance.COSINE
                ) 
            )
        
        qdrant.upsert(
        collection_name=UPLOADED_FILES_COLLECTION_NAME,
        points=[
            PointStruct(
                id=uuid.uuid4().int >> 64,
                vector=[0.0] * 10,
                payload={
                    "filename": file.filename,
                    "uploaded_at": datetime.now(timezone.utc).isoformat()
                }
            )
        ]
        )
        
        total_time = time.time() - start_time
        logger.info(f"Upload completed in {total_time:.2f}s")
        
        return {"status": f"✅ Uploaded {len(points)} clustered chunks for {file.filename}"}
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise
    

@app.post("/query_stream")
async def query_rag_stream(
    req: QueryRequest,
    _: None = Depends(require_api_key),
):
    async def generate_stream():
        start_time = time.time()
        logger.info(f"=== Streaming Query started ===")
        
        question = req.question.strip()
        if not question:
            yield "data: " + json.dumps({"error": "⚠️ Please enter a valid question."}) + "\n\n"
            return

        logger.info(f"Question received: '{question[:50]}...' for files: {req.filenames}")
        
        # Embedding generation
        try:
            embed_model = embedding_service.get_model()
            query_vector = embed_model.encode(question).tolist()
            logger.info("Query embedding generated")
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            yield "data: " + json.dumps({"error": f"⚠️ Embedding error: {e}"}) + "\n\n"
            return

        filters = None
        collection_name = ARXIV_COLLECTION_NAME
        if req.filenames:
            collection_name = PDF_COLLECTION_NAME
            filters = Filter(
                should=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=fname)
                    )
                    for fname in req.filenames
                ]
            )

        if not collection_exists(collection_name):
            logger.error(f"Qdrant collection not found: {collection_name}")
            yield "data: " + json.dumps({"error": f"⚠️ Qdrant collection '{collection_name}' not found. Please preload the database with ingest_arxiv.py."}) + "\n\n"
            return

        # Qdrant search
        try:
            hits = qdrant.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=RETRIEVAL_TOP_K,
                query_filter=filters
            )
            logger.info(f"Qdrant search completed, found {len(hits)} hits")
        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            yield "data: " + json.dumps({"error": f"⚠️ Search error: {e}"}) + "\n\n"
            return

        if not hits:
            yield "data: " + json.dumps({"error": "⚠️ No context found for this question."}) + "\n\n"
            return

        scores: List[float] = []
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
        
        # Ollama streaming request
        logger.info("Starting Ollama streaming request...")
        ollama_start = time.time()
        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream(
                    "POST",
                    f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/generate",
                    json={
                        "model": "phi3:3.8b-mini-128k-instruct-q4_0",
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            # Soft cap to discourage long answers (2–3 sentences ≪ this in practice)
                            "num_predict": 220,
                        },
                    }
                ) as response:
                    ollama_time = time.time() - ollama_start
                    logger.info(f"Ollama started streaming answer in {ollama_time:.2f}s")
                    async for line in response.aiter_lines():
                        if line:
                            try:
                                chunk = json.loads(line)
                                part = chunk.get("response")
                                if part:
                                    yield f"data: {json.dumps({'chunk': part})}\n\n"
                                    await asyncio.sleep(0.01)  # helps flushing

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
        media_type="text/event-stream",  # <-- set this for real streaming
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )    


@app.get("/files")
def list_files(_: None = Depends(require_api_key)):
    try:
        collections = qdrant.get_collections().collections
        collection_names = [col.name for col in collections]
        if UPLOADED_FILES_COLLECTION_NAME not in collection_names:
            return {"files": []}
        scroll = qdrant.scroll(
            collection_name=UPLOADED_FILES_COLLECTION_NAME,
            limit=1000,
            with_payload=True
        )
        points = scroll[0] if scroll else []
        filenames = list(set(
            point.payload.get("filename")
            for point in points if point.payload and "filename" in point.payload
        ))
        return {"files": filenames}
    except Exception as e:
        logger.error(f"List files error: {e}")
        return {"error": f"Qdrant scroll failed: {str(e)}"}

@app.post("/delete_file")
def delete_file(req: DeleteRequest, _: None = Depends(require_api_key)):
    try:
        # Delete file vectors
        qdrant.delete(
            collection_name=PDF_COLLECTION_NAME,
            points_selector=Filter(
                should=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=fname)
                    )
                    for fname in req.filenames
                ]
            )
        )
        # Delete file index
        qdrant.delete(
            collection_name=UPLOADED_FILES_COLLECTION_NAME,
            points_selector=Filter(
                should=[
                    FieldCondition(
                        key="filename",
                        match=MatchValue(value=fname)
                    )
                    for fname in req.filenames
                ]
            )
        )
        return {"status": f"🗑️ Deleted all vectors for {req.filenames}"}
    except Exception as e:
        logger.error(f"Delete error: {e}")
        return {"error": f"Delete failed: {str(e)}"} 
