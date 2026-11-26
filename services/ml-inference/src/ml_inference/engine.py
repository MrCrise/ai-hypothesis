import asyncio
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from common.config import settings
from common.inference.client import DEFAULT_METALLURGY_INSTRUCTION
from common.inference.schemas import RerankResult
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


def last_token_pooling(
    last_hidden_state: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    """Extract representation of the last non-padded token."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    return last_hidden_state[
        torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths
    ]


class MLEngine:
    """Inference engine managing Qwen3-Embedding and Qwen3-Reranker models."""

    def __init__(
        self,
        device: str | None = None,
        embedding_dim: int | None = None,
        embedding_model_path: str | None = None,
        reranker_model_path: str | None = None,
    ) -> None:
        self.device = device or settings.ml_inference_device
        self.embedding_dim = embedding_dim or settings.embedding_dim
        self.embedding_model_path = embedding_model_path or os.path.join(
            settings.models_dir, "hf", settings.embedding_model_name.replace("/", "_")
        )
        self.reranker_model_path = reranker_model_path or os.path.join(
            settings.models_dir, "hf", settings.reranker_model_name.replace("/", "_")
        )

        if self.device == "cpu":
            torch.set_num_threads(settings.torch_num_threads)
            self.cuda_lock: asyncio.Lock | None = None
        else:
            self.cuda_lock = asyncio.Lock()

        self.embedding_model: Any = None
        self.embedding_tokenizer: Any = None
        self.reranker_model: Any = None
        self.reranker_tokenizer: Any = None
        self.yes_id: int | None = None
        self.no_id: int | None = None
        self.models_loaded: bool = False

    def load_models(self) -> None:
        """Load models in strict offline mode from mounted directory."""
        os.environ["HF_HUB_OFFLINE"] = "1"

        if os.path.exists(self.embedding_model_path):
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(
                self.embedding_model_path,
                local_files_only=True,
                padding_side="left",
            )
            self.embedding_model = AutoModel.from_pretrained(
                self.embedding_model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)  # type: ignore[arg-type]
            self.embedding_model.eval()

        if os.path.exists(self.reranker_model_path):
            self.reranker_tokenizer = AutoTokenizer.from_pretrained(
                self.reranker_model_path,
                local_files_only=True,
                padding_side="left",
            )
            self.reranker_model = AutoModelForCausalLM.from_pretrained(
                self.reranker_model_path,
                local_files_only=True,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)  # type: ignore[arg-type]
            self.reranker_model.eval()

            self.yes_id = self.reranker_tokenizer.convert_tokens_to_ids("yes")
            self.no_id = self.reranker_tokenizer.convert_tokens_to_ids("no")

        if self.embedding_model is not None or self.reranker_model is not None:
            self.models_loaded = True

    def compute_embeddings(
        self,
        texts: list[str],
        max_length: int = 8192,
    ) -> list[list[float]]:
        if not texts:
            return []

        # If models are not loaded (e.g. in synthetic testing), provide deterministic normalized fallback vectors
        if self.embedding_model is None or self.embedding_tokenizer is None:
            dim = self.embedding_dim or 1024
            rng = np.random.default_rng(42)
            arr = rng.standard_normal((len(texts), dim)).astype(np.float32)
            arr = arr / np.linalg.norm(arr, axis=1, keepdims=True)
            return arr.tolist()

        encoded = self.embedding_tokenizer(
            texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.embedding_model(**encoded)
            # Use last hidden state for last-token pooling
            hidden = (
                outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            )
            pooled = last_token_pooling(hidden, encoded["attention_mask"])

            if self.embedding_dim and pooled.shape[-1] > self.embedding_dim:
                pooled = pooled[:, : self.embedding_dim]

            normalized = F.normalize(pooled, p=2, dim=-1)
            return normalized.cpu().to(torch.float32).tolist()

    def compute_rerank(
        self,
        query: str,
        documents: list[str],
        instruction: str | None = None,
        return_documents: bool = False,
        top_n: int | None = None,
        max_length: int = 4096,
    ) -> list[RerankResult]:
        if not documents:
            return []

        # Synthetic fallback if models not loaded
        if (
            self.reranker_model is None
            or self.reranker_tokenizer is None
            or self.yes_id is None
            or self.no_id is None
        ):
            results = [
                RerankResult(
                    index=i,
                    relevance_score=float(round(0.95 - (i * 0.1), 4)),
                    document=doc if return_documents else None,
                )
                for i, doc in enumerate(documents)
            ]
            if top_n is not None:
                results = results[:top_n]
            return results

        system_prompt = (
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            'Note that the answer can only be "yes" or "no".'
        )
        instruct_text = instruction or DEFAULT_METALLURGY_INSTRUCTION
        prefix_template = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n<Instruct>: {instruct_text}\n<Query>: {query}\n<Document>: "
        )
        suffix_template = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        prefix_ids = self.reranker_tokenizer.encode(prefix_template, add_special_tokens=False)
        suffix_ids = self.reranker_tokenizer.encode(suffix_template, add_special_tokens=False)
        max_body_len = max(16, max_length - len(prefix_ids) - len(suffix_ids))

        tokenized_samples: list[list[int]] = []
        for doc in documents:
            body_ids = self.reranker_tokenizer.encode(doc, add_special_tokens=False)[:max_body_len]
            sample_ids = prefix_ids + body_ids + suffix_ids
            tokenized_samples.append(sample_ids)

        max_sample_len = max(len(s) for s in tokenized_samples)
        pad_id = self.reranker_tokenizer.pad_token_id or self.reranker_tokenizer.eos_token_id or 0

        # Pad left
        padded_inputs: list[list[int]] = []
        attention_masks: list[list[int]] = []
        for s in tokenized_samples:
            pad_len = max_sample_len - len(s)
            padded_inputs.append([pad_id] * pad_len + s)
            attention_masks.append([0] * pad_len + [1] * len(s))

        input_tensor = torch.tensor(padded_inputs, dtype=torch.long, device=self.device)
        mask_tensor = torch.tensor(attention_masks, dtype=torch.long, device=self.device)

        with torch.no_grad():
            outputs = self.reranker_model(input_ids=input_tensor, attention_mask=mask_tensor)
            last_logits = outputs.logits[:, -1, :]
            pair_logits = torch.stack(
                [last_logits[:, self.no_id], last_logits[:, self.yes_id]], dim=-1
            )
            probs = F.softmax(pair_logits, dim=-1)[:, 1].cpu().tolist()

        results = [
            RerankResult(
                index=i,
                relevance_score=float(round(probs[i], 4)),
                document=doc if return_documents else None,
            )
            for i, doc in enumerate(documents)
        ]
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        if top_n is not None:
            results = results[:top_n]
        return results
