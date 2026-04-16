import argparse
import csv
import gzip
import glob
import json
import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer


COLLECTION_NAME = "arxiv_abstracts"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_LIMIT = 10_000
DEFAULT_BATCH_SIZE = 128
DEFAULT_SAMPLE_PATH = os.getenv("ARXIV_DATASET_PATH", "data/arxiv-dataset.json")
DEFAULT_RAW_DIR = os.getenv("ARXIV_RAW_DIR", "data/arxiv_raw")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest the first 10k arXiv abstracts into Qdrant."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=os.getenv("ARXIV_DATASET_PATH", DEFAULT_SAMPLE_PATH),
        help="Path to the sampled arXiv dataset file (.json, .jsonl, .csv, optionally .gz).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Maximum number of papers to ingest.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Embedding and upsert batch size.",
    )
    parser.add_argument(
        "--raw-dir",
        default=os.getenv("ARXIV_RAW_DIR", DEFAULT_RAW_DIR),
        help="Path to the raw dataset directory (parquet/json/csv files).",
    )
    return parser.parse_args()


def open_text_file(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def normalize_record(record):
    abstract = (
        record.get("abstract")
        or record.get("summary")
        or record.get("Abstract")
        or ""
    ).strip()
    title = (
        record.get("title")
        or record.get("Title")
        or ""
    ).strip()
    source_url = (
        record.get("link")
        or record.get("source_url")
        or (f"https://arxiv.org/abs/{record.get('id')}" if record.get("id") else "")
    ).strip()

    if not abstract or not title:
        return None

    return {
        "text": abstract,
        "title": title,
        "source_url": source_url,
        "link": source_url,
    }


def iter_json_records(path):
    with open_text_file(path) as handle:
        first_char = handle.read(1)
        handle.seek(0)

        if first_char == "[":
            data = json.load(handle)
            for item in data:
                if isinstance(item, dict):
                    yield item
            return

        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def iter_csv_records(path):
    with open_text_file(path) as handle:
        reader = csv.DictReader(handle)
        yield from reader


def iter_parquet_records(path):
    import pandas as pd

    df = pd.read_parquet(path)
    for _, row in df.iterrows():
        yield row.to_dict()


def iter_dataset_records(path):
    lowered = path.lower()
    if lowered.endswith((".csv", ".csv.gz")):
        yield from iter_csv_records(path)
        return
    if lowered.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz")):
        yield from iter_json_records(path)
        return
    if lowered.endswith(".parquet"):
        yield from iter_parquet_records(path)
        return
    raise ValueError(
        "Unsupported dataset format. Use .json, .jsonl, .csv, .parquet, or gzip-compressed variants."
    )


def iter_directory_records(directory):
    supported = [".parquet", ".json", ".jsonl", ".json.gz", ".jsonl.gz", ".csv", ".csv.gz"]
    for file_path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isdir(file_path):
            continue
        lowered = file_path.lower()
        if any(lowered.endswith(ext) for ext in supported):
            yield from iter_dataset_records(file_path)


def is_data_dir_populated(directory):
    if not os.path.isdir(directory):
        return False
    supported = [".parquet", ".json", ".jsonl", ".json.gz", ".jsonl.gz", ".csv", ".csv.gz"]
    for file_path in glob.glob(os.path.join(directory, "**", "*"), recursive=True):
        if os.path.isdir(file_path):
            continue
        if any(file_path.lower().endswith(ext) for ext in supported):
            return True
    return False


def load_documents(path, limit):
    documents = []
    if os.path.isdir(path):
        record_iter = iter_directory_records(path)
    else:
        record_iter = iter_dataset_records(path)

    for raw_record in record_iter:
        record = normalize_record(raw_record)
        if record is None:
            continue
        documents.append(record)
        if len(documents) >= limit:
            break
    return documents


def collection_exists(client, collection_name: str) -> bool:
    collections = client.get_collections().collections
    return collection_name in {collection.name for collection in collections}


def reset_collection(client, vector_size):
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]
    if COLLECTION_NAME in collection_names:
        client.delete_collection(collection_name=COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )


def main():
    args = parse_args()
    dataset_path = args.dataset
    raw_dir = args.raw_dir

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    if collection_exists(client, COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists, deleting it before ingest.")
        client.delete_collection(collection_name=COLLECTION_NAME)

    if dataset_path and os.path.exists(dataset_path):
        source_path = dataset_path
        print(f"📄 Using sample dataset file: {source_path}")
    elif is_data_dir_populated(raw_dir):
        source_path = raw_dir
        print(f"📁 Using raw dataset directory: {source_path}")
    else:
        raise SystemExit(
            f"Dataset not found: {dataset_path}. "
            f"No supported files found in raw dir: {raw_dir}. "
            "Run download_data.py first or provide a dataset path."
        )

    documents = load_documents(source_path, args.limit)
    if not documents:
        raise SystemExit("No valid records found in dataset.")

    model = SentenceTransformer(EMBEDDING_MODEL)
    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    qdrant_port = int(os.getenv("QDRANT_PORT", "6333"))
    client = QdrantClient(host=qdrant_host, port=qdrant_port)

    first_vector = model.encode(documents[0]["text"]).tolist()
    reset_collection(client, len(first_vector))

    first_point = PointStruct(
        id=str(uuid.uuid4()),
        vector=first_vector,
        payload=documents[0],
    )
    client.upsert(collection_name=COLLECTION_NAME, points=[first_point])

    for start in range(1, len(documents), args.batch_size):
        batch = documents[start:start + args.batch_size]
        texts = [doc["text"] for doc in batch]
        vectors = model.encode(texts, batch_size=args.batch_size)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector.tolist(),
                payload=doc,
            )
            for doc, vector in zip(batch, vectors)
        ]
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"Ingested {min(start + len(batch), len(documents))}/{len(documents)}")

    print(f"Finished ingesting {len(documents)} arXiv abstracts into '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
