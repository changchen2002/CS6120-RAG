import json
import os
import sys
import glob

from huggingface_hub import snapshot_download


DEFAULT_SAMPLE_PATH = os.getenv("ARXIV_SAMPLE_PATH", "data/arxiv-dataset.json")
DEFAULT_SAMPLE_SIZE = int(os.getenv("ARXIV_SAMPLE_SIZE", "10000"))
DEFAULT_SNAPSHOT_REPO = os.getenv(
    "ARXIV_HF_REPO",
    "permutans/arxiv-papers-by-subject"
)
DEFAULT_RAW_DIR = os.getenv("ARXIV_RAW_DIR", "data/arxiv_raw")


# -----------------------------
# Step 1: Download dataset (CS only)
# -----------------------------
def download_snapshot(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔽 Downloading dataset: {DEFAULT_SNAPSHOT_REPO}")

    snapshot_download(
        repo_id=DEFAULT_SNAPSHOT_REPO,
        repo_type="dataset",
        allow_patterns=[
            "data/cs.AI/2025/*/train-00000*.parquet"
        ],
        local_dir=output_dir,
        token=False,
    )

    print(f"✅ Download complete: {output_dir}")


# -----------------------------
# Step 2: Read parquet files
# -----------------------------
def iter_parquet_records(file_path):
    import pandas as pd

    df = pd.read_parquet(file_path)

    for _, row in df.iterrows():
        yield row.to_dict()


# -----------------------------
# Step 3: Sample data
# -----------------------------
def sample_arxiv_data(input_dir, output_file, n=10000):
    if os.path.exists(output_file):
        print(f"✅ Sample file already exists at {output_file}, skipping.")
        return

    print(f"🔄 Sampling {n} records from parquet files...")

    parquet_files = glob.glob(f"{input_dir}/**/*.parquet", recursive=True)

    if not parquet_files:
        raise FileNotFoundError("No parquet files found. Did download fail?")

    sampled_data = []

    for file in parquet_files:
        print(f"📄 Processing {file}")

        for item in iter_parquet_records(file):
            if len(sampled_data) >= n:
                break

            title = str(item.get("title", "")).strip()
            abstract = str(item.get("abstract", "")).strip()

            if not title or not abstract:
                continue

            paper_id = str(item.get("arxiv_id", "")).strip()

            sampled_data.append({
                "id": paper_id,
                "title": title,
                "abstract": abstract,
                "link": f"https://arxiv.org/abs/{paper_id}"
            })

        if len(sampled_data) >= n:
            break

    if not sampled_data:
        raise ValueError("No valid records found.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(sampled_data, f, ensure_ascii=False, indent=2)

    print(f"✅ Done! Saved {len(sampled_data)} records to {output_file}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    raw_dir = DEFAULT_RAW_DIR
    sample_path = DEFAULT_SAMPLE_PATH
    sample_size = DEFAULT_SAMPLE_SIZE

    try:
        download_snapshot(raw_dir)
        sample_arxiv_data(raw_dir, sample_path, n=sample_size)

    except Exception as exc:
        print(f"❌ Failed: {exc}")
        sys.exit(1)