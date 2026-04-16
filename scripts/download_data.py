import json
import os
import sys
import requests

from huggingface_hub import snapshot_download

DEFAULT_RAW_URL = os.getenv("ARXIV_RAW_URL", "")
DEFAULT_RAW_PATH = os.getenv("ARXIV_RAW_PATH", "data/arxiv-metadata-oai-snapshot.json")
DEFAULT_SAMPLE_PATH = os.getenv("ARXIV_SAMPLE_PATH", "data/arxiv-dataset.json")
DEFAULT_SAMPLE_SIZE = int(os.getenv("ARXIV_SAMPLE_SIZE", "10000"))
DEFAULT_SNAPSHOT_REPO = os.getenv("ARXIV_HF_REPO", "Just-Curieous/arxiv-cs-paper-metadata-embedding")


def download_file(url: str, output_path: str):
    if not url:
        raise ValueError(
            "ARXIV_RAW_URL is not set. Set this environment variable to a raw ArXiv metadata URL."
        )

    if os.path.exists(output_path):
        print(f"✅ Raw data already exists at {output_path}, skipping download.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"🔽 Downloading raw ArXiv metadata from {url}...")

    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with open(output_path, "wb") as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)

    print(f"✅ Download complete: {output_path}")


def download_snapshot(output_path: str):
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    print(
        f"🔽 Downloading Hugging Face snapshot repo {DEFAULT_SNAPSHOT_REPO} into {output_dir}..."
    )
    snapshot_download(
        repo_id=DEFAULT_SNAPSHOT_REPO,
        repo_type="dataset",
        local_dir=output_dir,
    )

    if os.path.exists(output_path):
        print(f"✅ Snapshot data already present at {output_path}.")
        return

    candidates = [
        f for f in os.listdir(output_dir)
        if f.endswith((".json", ".jsonl", ".json.gz", ".jsonl.gz"))
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No dataset file found in {output_dir} after snapshot_download."
        )

    first_file = os.path.join(output_dir, candidates[0])
    if first_file != output_path:
        os.replace(first_file, output_path)
    print(f"✅ Snapshot data saved to {output_path}")


def sample_arxiv_data(input_file, output_file, n=100):
    """
    Read the first N lines from the raw Arxiv dataset and save them.
    """
    if os.path.exists(output_file):
        print(f"✅ Sample file already exists at {output_file}, skipping sampling.")
        return

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Raw file not found: {input_file}")

    print(f"🔄 Extracting the first {n} records from {input_file}...")
    sampled_data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if len(sampled_data) >= n:
                break
            line = line.strip()
            if not line:
                continue

            item = json.loads(line)
            categories = item.get("categories", "")
            if not any(cat.startswith("cs.") for cat in categories.split()):
                continue

            paper_id = item.get("id")
            relevant_data = {
                "id": paper_id,
                "title": item.get("title"),
                "abstract": item.get("abstract"),
                "link": f"https://arxiv.org/abs/{paper_id}" if paper_id else None,
            }
            sampled_data.append(relevant_data)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Sampled data saved to: {output_file}")


if __name__ == "__main__":
    raw_path = DEFAULT_RAW_PATH
    sample_path = DEFAULT_SAMPLE_PATH
    sample_size = DEFAULT_SAMPLE_SIZE

    try:
        if DEFAULT_RAW_URL.strip():
            print(f"🚀 Detected ARXIV_RAW_URL. Prioritizing direct download...")
            download_file(DEFAULT_RAW_URL, raw_path)
        else:
            print(f"🚀 No direct URL provided. Using Hugging Face repo: {DEFAULT_SNAPSHOT_REPO}")
            download_snapshot(raw_path)
            
        sample_arxiv_data(raw_path, sample_path, n=sample_size)
        
    except Exception as exc:
        print(f"❌ Failed: {exc}")
        sys.exit(1)