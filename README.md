# 🧠 Citation RAG

Fully local RAG app (FastAPI + Streamlit) backed by Qdrant and Ollama. On first run, the container downloads a small arXiv abstract sample and ingests it into Qdrant.

## Prerequisites

- Docker
- Docker Compose

## Quickstart

From the repo root:

```bash
docker compose up --build
```

First startup can take a while because models and dependencies are downloaded.

## Open the app

- **Streamlit UI**: http://localhost:8501
- **FastAPI docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health
- **Qdrant dashboard**: http://localhost:6333/dashboard
- **Ollama API**: http://localhost:11434

## What happens on startup

The `rag-app` container runs:

- `python scripts/download_data.py` (downloads a Hugging Face snapshot into `data/arxiv_raw/` and writes `data/arxiv-dataset.json`)
- `python scripts/ingest_arxiv.py data/arxiv-dataset.json` (creates/updates the Qdrant collection `arxiv_abstracts`)
- starts FastAPI on `:8000` and Streamlit on `:8501`

Qdrant storage is persisted via the Docker volume `qdrant_data`, so data remains available across restarts.

## Rerun / reset

- **Start again (no rebuild)**:

```bash
docker compose up
```

- **Stop**:

```bash
docker compose down
```

- **Reset Qdrant data (deletes your local index)**:

```bash
docker compose down -v
```

## Optional configuration

Set env vars in `docker-compose.yml` (or your shell) if needed:

- `RAG_API_KEY`: require `X-API-Key` for API + UI
- `RAG_SCORE_HIGH`, `RAG_SCORE_LOW`: retrieval-confidence thresholds
- `ARXIV_HF_REPO`, `ARXIV_SAMPLE_SIZE`, `ARXIV_DATASET_PATH`, `ARXIV_RAW_DIR`: dataset download/sample settings

## License

MIT (see `LICENSE`).
