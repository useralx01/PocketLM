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


def _load_input_ids(model_path: Path, prompt: str) -> tuple[torch.Tensor, str, Any | None]:
    input_ids_path = model_path / "input_ids.json"
    if input_ids_path.exists():
        payload = json.loads(input_ids_path.read_text(encoding="utf-8"))
        token_ids = payload["input_ids"] if isinstance(payload, dict) else payload
        return torch.tensor([list(map(int, token_ids))], dtype=torch.long), "input_ids.json", None

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    return inputs["input_ids"], "tokenizer", tokenizer


def _decode_ids(tokenizer: Any | None, token_ids: list[int]) -> str:
    if tokenizer is None:
        return " ".join(str(value) for value in token_ids)
    return tokenizer.decode(token_ids, skip_special_tokens=False)


def _select_last_router_logits(logits: torch.Tensor) -> torch.Tensor:
    logits_cpu = logits.detach().float().cpu()
    if logits_cpu.ndim == 2:
        return logits_cpu[-1:, :]
    if logits_cpu.ndim == 3:
        return logits_cpu[:, -1, :]
    return logits_cpu.reshape(-1, logits_cpu.shape[-1])[-1:, :]


def _module_for_name(root: torch.nn.Module, dotted_name: str) -> torch.nn.Module | None:
    current: Any = root
    for part in dotted_name.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current if isinstance(current, torch.nn.Module) else None


def _register_checkpoint_hooks(model: torch.nn.Module, captures: dict[str, torch.Tensor]) -> list[Any]:
    handles: list[Any] = []
    layers = _module_for_name(model, "model.layers")
    if layers is None or not hasattr(layers, "__getitem__"):
        return handles
    layer0 = layers[0]

    def capture(name: str):
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            tensor = output[0] if isinstance(output, tuple) else output
            if isinstance(tensor, torch.Tensor):
                captures[name] = tensor.detach().cpu().float()

        return hook

    for name, module_name in [
        ("layer0_attention_output", "self_attn"),
        ("layer0_moe_output", "mlp"),
        ("layer0_mixtral_moe_output", "block_sparse_moe"),
    ]:
        module = _module_for_name(layer0, module_name)
        if module is not None:
            handles.append(module.register_forward_hook(capture(name)))

    for name, module_name in [
        ("layer0_router_logits", "mlp.gate"),
        ("layer0_mixtral_router_logits", "block_sparse_moe.gate"),
    ]:
        module = _module_for_name(layer0, module_name)
        if module is not None:
            handles.append(module.register_forward_hook(capture(name)))

    return handles


def capture_reference(
    *,
    model_id: str,
    model_path: Path,
    output_dir: Path,
    prompt: str,
    max_new_tokens: int,
) -> Path:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()

    input_ids, input_source, tokenizer = _load_input_ids(model_path, prompt)
    attention_mask = torch.ones_like(input_ids)
    captured_tensors: dict[str, torch.Tensor] = {}
    hook_handles = _register_checkpoint_hooks(model, captured_tensors)
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            temperature=None,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            output_hidden_states=True,
            pad_token_id=getattr(config, "pad_token_id", None) or getattr(config, "eos_token_id", None),
        )
        forward = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
    for handle in hook_handles:
        handle.remove()

    prompt_len = int(input_ids.shape[-1])
    generated_ids = output.sequences[0, prompt_len:].detach().cpu().tolist()
    decoded = _decode_ids(tokenizer, generated_ids)
    full_decoded = _decode_ids(tokenizer, output.sequences[0].detach().cpu().tolist())

    output_dir.mkdir(parents=True, exist_ok=True)
    first_token_embedding = model.get_input_embeddings()(input_ids[:, :1]).detach().cpu().float().numpy()
    np.save(output_dir / "embedding_first_token.npy", first_token_embedding)
    if forward.hidden_states:
        np.save(output_dir / "final_hidden_before_lm_head.npy", forward.hidden_states[-1].detach().cpu().float().numpy())
        if len(forward.hidden_states) > 1:
            np.save(output_dir / "layer0_combined_hidden.npy", forward.hidden_states[1].detach().cpu().float().numpy())
    for name, tensor in captured_tensors.items():
        np.save(output_dir / f"{name}.npy", tensor.numpy())

    router_summary: list[dict[str, Any]] = []
    if getattr(forward, "router_logits", None):
        for layer_index, logits in enumerate(forward.router_logits):
            logits_cpu = logits.detach().float().cpu()
            last_logits = _select_last_router_logits(logits)
            top_values, top_ids = torch.topk(torch.softmax(last_logits, dim=-1), k=min(8, last_logits.shape[-1]))
            router_summary.append(
                {
                    "layer": layer_index,
                    "logits": _tensor_summary(logits_cpu),
                    "top_expert_ids": top_ids[0].tolist() if top_ids.ndim == 2 else top_ids.tolist(),
                    "top_weights": top_values[0].tolist() if top_values.ndim == 2 else top_values.tolist(),
                }
            )
    elif "layer0_router_logits" in captured_tensors or "layer0_mixtral_router_logits" in captured_tensors:
        logits = captured_tensors.get("layer0_router_logits", captured_tensors.get("layer0_mixtral_router_logits"))
        assert logits is not None
        last_logits = _select_last_router_logits(logits)
        top_values, top_ids = torch.topk(torch.softmax(last_logits, dim=-1), k=min(8, last_logits.shape[-1]))
        router_summary.append(
            {
                "layer": 0,
                "logits": _tensor_summary(logits),
                "top_expert_ids": top_ids[0].tolist() if top_ids.ndim == 2 else top_ids.tolist(),
                "top_weights": top_values[0].tolist() if top_values.ndim == 2 else top_values.tolist(),
            }
        )

    payload = {
        "model_id": model_id,
        "model_path": str(model_path),
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "input_source": input_source,
        "config": {
            "model_type": getattr(config, "model_type", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_experts": getattr(config, "num_experts", getattr(config, "num_local_experts", None)),
            "num_experts_per_tok": getattr(config, "num_experts_per_tok", None),
            "norm_topk_prob": getattr(config, "norm_topk_prob", None),
        },
        "prompt_token_ids": input_ids[0].detach().cpu().tolist(),
        "generated_token_ids": generated_ids,
        "decoded_generated_text": decoded,
        "decoded_full_text": full_decoded,
        "embedding_first_token": _tensor_summary(torch.from_numpy(first_token_embedding)),
        "final_hidden_before_lm_head": _tensor_summary(forward.hidden_states[-1]) if forward.hidden_states else None,
        "captured_checkpoints": sorted(captured_tensors),
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
    parser.add_argument("--output-dir")
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new-tokens", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root) / f"{_safe_model_name(args.model_id)}_moe_reference"
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
