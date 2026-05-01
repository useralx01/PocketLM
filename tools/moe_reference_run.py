"""Capture a Hugging Face MoE reference run for correctness debugging.

This is intentionally a one-off diagnostic tool: it loads a local model folder,
runs greedy generation, and writes the generated ids/text plus lightweight
router/output metadata for comparison slices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _safe_model_name(model_id: str) -> str:
    return model_id.replace(".", "_").replace("-", "_").replace("/", "_")


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    return {
        "shape": [int(value) for value in detached.shape],
        "mean": float(detached.mean().item()),
        "std": float(detached.std().item()) if detached.numel() > 1 else 0.0,
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
    }


def capture_reference(
    *,
    model_id: str,
    model_path: Path,
    output_dir: Path,
    prompt: str,
    max_new_tokens: int,
) -> Path:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            do_sample=False,
            temperature=None,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            output_router_logits=True,
        )
        forward = model(
            **inputs,
            use_cache=False,
            output_hidden_states=True,
            output_router_logits=True,
            return_dict=True,
        )

    prompt_len = int(inputs["input_ids"].shape[-1])
    generated_ids = output.sequences[0, prompt_len:].detach().cpu().tolist()
    decoded = tokenizer.decode(generated_ids, skip_special_tokens=False)
    full_decoded = tokenizer.decode(output.sequences[0].detach().cpu().tolist(), skip_special_tokens=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    first_token_embedding = model.get_input_embeddings()(inputs["input_ids"][:, :1]).detach().cpu().float().numpy()
    np.save(output_dir / "embedding_first_token.npy", first_token_embedding)
    if forward.hidden_states:
        np.save(output_dir / "final_hidden_before_lm_head.npy", forward.hidden_states[-1].detach().cpu().float().numpy())

    router_summary: list[dict[str, Any]] = []
    if getattr(forward, "router_logits", None):
        for layer_index, logits in enumerate(forward.router_logits):
            logits_cpu = logits.detach().float().cpu()
            top_values, top_ids = torch.topk(torch.softmax(logits_cpu[:, -1, :], dim=-1), k=min(8, logits_cpu.shape[-1]))
            router_summary.append(
                {
                    "layer": layer_index,
                    "logits": _tensor_summary(logits_cpu),
                    "top_expert_ids": top_ids[0].tolist() if top_ids.ndim == 2 else top_ids.tolist(),
                    "top_weights": top_values[0].tolist() if top_values.ndim == 2 else top_values.tolist(),
                }
            )

    payload = {
        "model_id": model_id,
        "model_path": str(model_path),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "config": {
            "model_type": getattr(config, "model_type", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_experts": getattr(config, "num_experts", getattr(config, "num_local_experts", None)),
            "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
            "norm_topk_prob": getattr(config, "norm_topk_prob", None),
        },
        "prompt_token_ids": inputs["input_ids"][0].detach().cpu().tolist(),
        "generated_token_ids": generated_ids,
        "decoded_generated_text": decoded,
        "decoded_full_text": full_decoded,
        "embedding_first_token": _tensor_summary(torch.from_numpy(first_token_embedding)),
        "final_hidden_before_lm_head": _tensor_summary(forward.hidden_states[-1]) if forward.hidden_states else None,
        "router_summary": router_summary,
    }
    reference_path = output_dir / "reference.json"
    reference_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return reference_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a HF Transformers MoE oracle run.")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-root", default="tests/fixtures")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_root) / f"{_safe_model_name(args.model_id)}_moe_reference"
    reference_path = capture_reference(
        model_id=args.model_id,
        model_path=Path(args.model_path),
        output_dir=output_dir,
        prompt=args.prompt,
        max_new_tokens=int(args.max_new_tokens),
    )
    print(reference_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
