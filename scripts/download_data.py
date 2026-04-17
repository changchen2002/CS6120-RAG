import os
import sys

from huggingface_hub import snapshot_download


DEFAULT_SNAPSHOT_REPO = os.getenv(
    "ARXIV_HF_REPO",
    "permutans/arxiv-papers-by-subject"
)
DEFAULT_RAW_DIR = os.getenv("ARXIV_RAW_DIR", "data/arxiv_raw")
DEFAULT_HF_CACHE_DIR = os.getenv("HF_HOME", "hf_cache")


# -----------------------------
# Download parquet snapshot
# -----------------------------
def download_snapshot(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print(f"🔽 Downloading dataset: {DEFAULT_SNAPSHOT_REPO}")

    snapshot_download(
        repo_id=DEFAULT_SNAPSHOT_REPO,
        repo_type="dataset",
        allow_patterns=[
            "data/cs.AI/2023/*/00000000.parquet",
            "data/cs.AI/2024/*/00000000.parquet",
            "data/cs.AI/2025/*/00000000.parquet"
        ],
        # Make re-runs incremental / restart-safe and avoid re-downloading content.
        cache_dir=DEFAULT_HF_CACHE_DIR,
        # Write real files under local_dir (not symlinks to cache) so ingest can scan `data/arxiv_raw/`.
        local_dir_use_symlinks=False,
        local_dir=output_dir,
        token=False,
    )

    print(f"✅ Download complete: {output_dir}")


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    raw_dir = DEFAULT_RAW_DIR

    try:
        download_snapshot(raw_dir)
        print("✅ Snapshot ready. You can now run ingestion from the raw directory:")
        print("   python scripts/ingest_arxiv.py data/arxiv_raw")

    except Exception as exc:
        print(f"❌ Failed: {exc}")
        sys.exit(1)