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

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
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
qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

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
    """De-duplicate sources if the index still contains identical papers (legacy duplicates)."""
    rows = []
    seen = set()
    for hit in hits:
        payload = hit.payload or {}
        title = payload.get("title") or payload.get("filename", "Untitled")
        url = (payload.get("link") or payload.get("source_url") or "").strip()
        passage = payload.get("text", "") or ""
        if url:
            key = ("url", url, title.strip())
        else:
            key = ("txt", title.strip(), passage[:400])
        if key in seen:
            continue
        seen.add(key)
        rows.append({"title": title, "url": url, "passage": passage})
    return rows


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

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
async def query_rag_stream(req: QueryRequest):
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

        sources = build_sources(hits)
        yield "data: " + json.dumps({"sources": sources}) + "\n\n"

        context = "\n\n".join([hit.payload["text"] for hit in hits])
        prompt = (
            "Answer using only the context below. "
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
                                    yield "data: " + json.dumps({"done": True, "sources": sources}) + "\n\n"
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
def list_files():
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
def delete_file(req: DeleteRequest):
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
