"""Download the pre-converted INT4 OpenVINO IR of Qwen2.5-1.5B-Instruct from the OpenVINO Hugging Face org.

Run:  .venv\\Scripts\\python.exe -m planner.download_model
"""
from huggingface_hub import snapshot_download

from planner.planner import MODEL_DIR, MODEL_ID


def main():
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    print(f"downloaded {MODEL_ID} to {path}")


if __name__ == "__main__":
    main()
