"""Download the pre-converted INT4 OpenVINO IR of Qwen2.5-1.5B-Instruct (~0.9 GB, not gated).

Source: the OpenVINO organisation on Hugging Face (Apache-2.0, like Qwen2.5 itself).

Run:  python -m planner.download_model
"""
import argparse

from huggingface_hub import snapshot_download

from planner.planner import MODEL_DIR, MODEL_ID


def main():
    parser = argparse.ArgumentParser(description=f"Download {MODEL_ID} (~0.9 GB) to {MODEL_DIR}.")
    parser.add_argument("--force", action="store_true", help="re-sync even if the folder already exists")
    args = parser.parse_args()
    if MODEL_DIR.exists() and not args.force:
        print(f"{MODEL_DIR} already exists; pass --force to re-sync")
        return
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    print(f"downloaded {MODEL_ID} to {path}")


if __name__ == "__main__":
    main()
