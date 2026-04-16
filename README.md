# 🧠 Local RAG App with Qdrant + Ollama
<div align="center">

![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)

[![GitHub stars](https://img.shields.io/github/stars/noorjotk/local-rag-engine?style=social)](https://github.com/noorjotk/local-rag-engine/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/noorjotk/local-rag-engine?style=social)](https://github.com/noorjotk/local-rag-engine/network/members)

</div>


> **🚀 Minimal setup**: The `rag-app` container downloads raw data, samples it, and ingests it into Qdrant before starting the app.

A fully local, end-to-end Retrieval-Augmented Generation (RAG) system built with:
- 🧠 **FastAPI** – backend API server
- 🎨 **Streamlit** – simple and interactive frontend UI
- 🔍 **Qdrant** – high-performance vector database
- 🤖 **Ollama** – run LLMs like Phi3/Mistral etc locally
- 📄 **PDF Support** – upload documents and ask questions

This project is configured so that the `rag-app` service will run `python scripts/download_data.py` and then `python scripts/ingest_arxiv.py` before starting FastAPI and Streamlit.
If the `arxiv_abstracts` collection already exists, ingestion is skipped automatically.
The Qdrant volume `qdrant_data:/qdrant/storage` persists the ingested data, so subsequent restarts can query the existing dataset directly.

---

## 🚀 Features

- 📁 Upload **PDFs** and extract content
- 🧠 Embed content using `sentence-transformers`
- 🔍 Store and search with **Qdrant** vector database
- 🤖 Query documents using **local LLMs via Ollama**
- 🧩 Smart chunking and **clustering** support via `LangChain` & `scikit-learn`
- 🛡️ 100% **offline** — your data never leaves your machine
- ⚡ Fast retrieval and response generation
- 🎯 Semantic search with relevance scoring

### 🔥 What Makes This Different

|❌ **Traditional RAG Setup** |✅ **This Project** |
|---|---|
| Install Python manually | ✅ Containerized |
| Download Ollama separately | ✅ Auto-installed |
| Pull models manually | ✅ Auto-downloaded |
| Configure embeddings | ✅ Pre-configured |
| Set up vector database | ✅ Persistent Qdrant volume |
| **Result: Hours of setup** | **Result: Container-run download + ingest + startup** |

**Run `docker compose up --build`** to let the container download raw data, sample the dataset, ingest Qdrant, and start the app. 🎉

---

## 📋 Prerequisites

**Only need these 2 things:**
- **Docker** (version 20.10+)
- **Docker Compose** (version 2.0+)

**That's it!** No Python, no Ollama, no manual model downloads required.

**Recommended system specs:**
- At least **8GB RAM** (for optimal LLM performance)
- **10GB+ free disk space** (for models and containers)

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/noorjotk/local-rag-engine.git
cd local-rag-engine
```

### 2. Start the App and Load Data in the Container
```bash
docker compose up --build
```

This will run the following sequence inside the `rag-app` container:
1. download the raw ArXiv metadata file from the configured `ARXIV_RAW_URL`,
2. sample the first 10k qualifying records to `data/arxiv-dataset.json`,
3. ingest the sampled records into the `arxiv_abstracts` Qdrant collection,
4. start FastAPI and Streamlit.

The data is persisted in the volume `qdrant_data:/qdrant/storage` so Qdrant keeps the collection between restarts.

If the `arxiv_abstracts` collection already exists, ingestion is skipped and the app starts directly.

### 3. Monitor Startup Progress
```bash
docker compose logs -f rag-app
```

⏱️ **Note:** First build will automatically:
- Download and install all Python dependencies for the app
- Pre-download the embedding model (`BAAI/bge-small-en-v1.5`)
- Pull and load the LLM model (`phi3:3.8b-mini-128k-instruct-q4_0`)
- Start Qdrant and Ollama services
- This may take 10-15 minutes (depending on your internet connection)

### 3. Monitor Startup Progress
You'll see logs indicating the startup process:

```bash
rag-app  | INFO:     Started server process [7]
rag-app  | INFO:     Waiting for application startup.
rag-app  | INFO:app:Starting up application...
rag-app  | INFO:app:Loading embedding model...
rag-app  | INFO:sentence_transformers.SentenceTransformer:Load pretrained SentenceTransformer: BAAI/bge-small-en-v1.5
ollama   | time=2025-07-26T09:11:13.894Z level=INFO source=server.go:637 msg="llama runner started in 4.28 seconds"
ollama   | [GIN] 2025/07/26 - 09:11:14 | 200 |  5.105294711s |             ::1 | POST     "/api/chat"
ollama   | ✅ Model loaded into memory.
rag-app  | INFO:app:Embedding model warmed up successfully
qdrant   | 2025-07-26T09:13:05.887308Z  INFO actix_web::middleware::logger: 172.18.0.4 "GET /collections HTTP/1.1" 200 111
rag-app  | INFO:httpx:HTTP Request: GET http://qdrant:6333/collections "HTTP/1.1 200 OK"
rag-app  | INFO:app:Qdrant connected, found 2 collections
rag-app  | INFO:     Application startup complete.
rag-app  | INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

✅ **Ready to use!** Once you see `INFO: Uvicorn running on http://0.0.0.0:8000`, the application is fully started and accessible at **http://localhost:8501**

### 4. Subsequent Runs
```bash
docker compose up
```

With the Qdrant volume already containing the ingested dataset, the application starts directly and queries the existing data. The one-time ingest step will be skipped if `arxiv_abstracts` already exists.

🚀 **That's it!** No configuration files needed - everything is automated!

---

## 🌐 Access URLs

Once running, access the application through:

| Service | URL | Description |
|---------|-----|-------------|
| 🔗 **Streamlit App** | http://localhost:8501 | Main user interface |
| 📘 **FastAPI Docs** | http://localhost:8000/docs | API documentation |
| 🧠 **Qdrant UI** | http://localhost:6333/dashboard | Vector database dashboard |
| 🤖 **Ollama API** | http://localhost:11434 | LLM service endpoint |

---

## 📖 Usage Guide

### Basic Workflow
1. **Start the application** using Docker Compose
2. **Open Streamlit UI** at http://localhost:8501
3. **Upload PDF documents** through the file uploader
4. **Wait for processing**
5. **Ask questions** about your documents
6. **Get AI-powered answers**

---

## 📦 Tech Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI | RESTful API server |
| **Frontend** | Streamlit | Interactive web interface |
| **Vector DB** | Qdrant | Similarity search & storage |
| **LLM Runtime** | Ollama | Local language model inference |
| **Embeddings** | sentence-transformers | Text vectorization |
| **Document Processing** | LangChain | Text chunking & QA chains |
| **Clustering** | scikit-learn | Document similarity grouping |
| **Containerization** | Docker + Compose | Deployment & orchestration |

---

## ⚙️ Configuration

### Pre-configured Models
The application comes pre-configured with optimized models:

- **LLM Model**: `phi3:3.8b-mini-128k-instruct-q4_0` (automatically downloaded)
- **Embedding Model**: `BAAI/bge-small-en-v1.5` (pre-downloaded during build)
- **Vector Database**: Qdrant with persistent storage
- **Chunk Settings**: Optimized for document processing

### Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   RAG App       │    │     Ollama      │    │     Qdrant      │
│ FastAPI+Streamlit│◄──►│   (Phi-3 Model) │    │ (Vector Store)  │
│  Port: 8000/8501│    │  Port: 11434    │    │  Port: 6333     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

All services are automatically configured and connected via Docker Compose.

---

## 🐛 Troubleshooting

### Common Issues

**First-time build taking long:**
- This is normal! The build downloads ~2-3GB of models
- Check progress: `docker-compose logs -f`
- Models are cached for subsequent runs

**Docker build fails:**
```bash
# Clean Docker cache and rebuild
docker system prune -a
docker-compose up --build
```

**Ollama model loading issues:**
```bash
# Check Ollama container logs
docker-compose logs ollama
# The entrypoint.sh automatically handles model downloading
```

**Memory issues:**
- Ensure Docker has at least 8GB RAM allocated
- The Phi-3 model is optimized (3.8B parameters, quantized)
- Monitor usage: `docker stats`

**Port conflicts:**
- Ports used: 8000 (FastAPI), 8501 (Streamlit), 6333 (Qdrant), 11434 (Ollama)
- Modify ports in `docker-compose.yml` if needed

**Application not responding:**
```bash
# Check all services are running
docker-compose ps
# Restart if needed
docker-compose restart
```

---

## 🛡️ 100% Local & Secure

✅ **No OpenAI API key required**  
✅ **No remote API calls**  
✅ **Your files stay on your machine**  
✅ **Ideal for privacy-conscious use cases**  
✅ **Perfect for secure environments**  

---

## 📊 What Happens During First Build
1. **Python Dependencies**: Downloads from requirements.txt (~500MB)
2. **Embedding Model**: Pre-downloads `BAAI/bge-small-en-v1.5` (~500MB)
3. **LLM Model**: Pulls `phi3:3.8b-mini-128k-instruct-q4_0` (~2.2GB)
4. **Model Warm-up**: Loads model into memory for faster responses

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Thanks to the amazing open-source projects that make this possible:

- [Qdrant](https://qdrant.tech/) - Vector database
- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Sentence Transformers](https://www.sbert.net/) - Embedding models
- [LangChain](https://langchain.com/) - LLM application framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Streamlit](https://streamlit.io/) - App framework

