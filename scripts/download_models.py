#!/usr/bin/env python3
"""Model download utility for closed contour inference environment.

Downloads GGUF and Hugging Face weights into structured directories:
./models/gguf/<filename>.gguf
./models/hf/<model_id>/
"""

import argparse
import os
import sys

from huggingface_hub import hf_hub_download, snapshot_download


def download_models(
    models_dir: str = "./models",
    llm_repo: str = "unsloth/Qwen3.8-27B-GGUF",
    llm_file: str = "Qwen3.8-27B-UD-Q4_K_M.gguf",
    embedding_repo: str = "Qwen/Qwen3-Embedding-0.6B",
    reranker_repo: str = "Qwen/Qwen3-Reranker-0.6B",
    skip_llm: bool = False,
    skip_embedding: bool = False,
    skip_reranker: bool = False,
) -> None:
    gguf_dir = os.path.join(models_dir, "gguf")
    hf_dir = os.path.join(models_dir, "hf")

    os.makedirs(gguf_dir, exist_ok=True)
    os.makedirs(hf_dir, exist_ok=True)

    print(f"Target directories prepared: {gguf_dir}, {hf_dir}")

    if not skip_llm:
        print(f"Downloading LLM GGUF: {llm_repo} -> {llm_file}...")
        hf_hub_download(
            repo_id=llm_repo,
            filename=llm_file,
            local_dir=gguf_dir,
        )
        print("LLM GGUF downloaded.")

    if not skip_embedding:
        target_embed_dir = os.path.join(hf_dir, embedding_repo.replace("/", "_"))
        os.makedirs(target_embed_dir, exist_ok=True)
        print(f"Downloading Embedding weights: {embedding_repo} -> {target_embed_dir}...")
        snapshot_download(
            repo_id=embedding_repo,
            local_dir=target_embed_dir,
        )
        print("Embedding weights downloaded.")

    if not skip_reranker:
        target_rerank_dir = os.path.join(hf_dir, reranker_repo.replace("/", "_"))
        os.makedirs(target_rerank_dir, exist_ok=True)
        print(f"Downloading Reranker weights: {reranker_repo} -> {target_rerank_dir}...")
        snapshot_download(
            repo_id=reranker_repo,
            local_dir=target_rerank_dir,
        )
        print("Reranker weights downloaded.")

    print("All requested models downloaded successfully into offline storage.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download model weights for offline inference subsystem into ./models"
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
        help="Root models directory (default: ./models)",
    )
    parser.add_argument(
        "--llm-repo",
        default="unsloth/Qwen3.8-27B-GGUF",
        help="HuggingFace repo for LLM GGUF",
    )
    parser.add_argument(
        "--llm-file",
        default="Qwen3.8-27B-UD-Q4_K_M.gguf",
        help="GGUF filename to download",
    )
    parser.add_argument(
        "--embedding-repo",
        default="Qwen/Qwen3-Embedding-0.6B",
        help="HuggingFace repo for embedding model",
    )
    parser.add_argument(
        "--reranker-repo",
        default="Qwen/Qwen3-Reranker-0.6B",
        help="HuggingFace repo for reranker model",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip downloading LLM GGUF model",
    )
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip downloading embedding model",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip downloading reranker model",
    )

    args = parser.parse_args()
    try:
        download_models(
            models_dir=args.models_dir,
            llm_repo=args.llm_repo,
            llm_file=args.llm_file,
            embedding_repo=args.embedding_repo,
            reranker_repo=args.reranker_repo,
            skip_llm=args.skip_llm,
            skip_embedding=args.skip_embedding,
            skip_reranker=args.skip_reranker,
        )
    except Exception as exc:
        print(f"Error downloading models: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
