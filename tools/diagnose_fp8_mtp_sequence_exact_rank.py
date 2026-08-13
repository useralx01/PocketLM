from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.fp8_source import (  # noqa: E402
    _fp8_lm_head_chunk_rows,
    _load_deepseek_config,
    _load_fp8_lm_head_full_cached,
    _load_regular_tensor,
    _read_tensor_rows,
    _rms_norm_any,
    apply_fp8_exact_mtp_default_env,
    find_tensor_catalog_entry,
    run_fp8_mtp_draft_step,
    run_fp8_prompt_prefill,
)
from pcketlm.core.runtime.tokenizer_runtime import decode_token_ids_to_text, prepare_prompt_text  # noqa: E402


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _parse_token_ids(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).replace(";", ",").split(",") if part.strip()]


def _target_rank(
    model_id: str,
    hidden: torch.Tensor,
    *,
    target_token_id: int,
    norm_name: str,
    head_name: str,
) -> dict:
    started = time.perf_counter()
    norm_tensor, blockers = _load_regular_tensor(model_id, norm_name)
    head_entry = find_tensor_catalog_entry(model_id, head_name)
    config = _load_deepseek_config(model_id)
    if norm_tensor is None:
        blockers.append(f"Final norm tensor {norm_name} did not materialize.")
    if head_entry is None:
        blockers.append(f"{head_name} is not present in the tensor catalog.")
    if head_entry is not None and head_entry.dtype not in {"BF16", "F16", "F32"}:
        blockers.append(f"{head_name} has unsupported dtype {head_entry.dtype}.")
    if blockers or norm_tensor is None or head_entry is None:
        return {"ready": False, "blockers": blockers, "elapsed_seconds": round(time.perf_counter() - started, 4)}

    hidden_2d = hidden.reshape(-1, hidden.shape[-1])[-1:].to(dtype=torch.bfloat16)
    norm_tensor = norm_tensor.to(device=hidden_2d.device)
    normalized_u16 = _rms_norm_any(hidden_2d, norm_tensor.float(), float(config["rms_norm_eps"]))
    normalized = normalized_u16.float()
    vocab_size = int(head_entry.shape[0])
    hidden_size = int(head_entry.shape[1])
    if int(normalized.shape[-1]) != hidden_size:
        blockers.append(f"Hidden size {normalized.shape[-1]} does not match head hidden size {hidden_size}.")
        return {"ready": False, "blockers": blockers, "elapsed_seconds": round(time.perf_counter() - started, 4)}
    if not (0 <= int(target_token_id) < vocab_size):
        blockers.append(f"Target token {target_token_id} is outside vocab size {vocab_size}.")
        return {"ready": False, "blockers": blockers, "elapsed_seconds": round(time.perf_counter() - started, 4)}

    rows_per_chunk = max(1, int(_fp8_lm_head_chunk_rows()))
    cached_head, cache_loaded_bytes = _load_fp8_lm_head_full_cached(head_entry)
    loaded_bytes = int(cache_loaded_bytes)
    target_start = int(target_token_id)
    if cached_head is not None:
        target_weight = cached_head[target_start : target_start + 1]
    else:
        target_weight = _read_tensor_rows(head_entry, target_start, target_start + 1)
        loaded_bytes += hidden_size * 2
    target_logit = float(F.linear(normalized, target_weight.to(device=normalized.device).float()).reshape(-1)[0].item())

    greater = 0
    equal = 0
    chunk_count = 0
    for start in range(0, vocab_size, rows_per_chunk):
        end = min(vocab_size, start + rows_per_chunk)
        if cached_head is not None:
            weight = cached_head[start:end]
        else:
            weight = _read_tensor_rows(head_entry, start, end)
            loaded_bytes += (end - start) * hidden_size * 2
        logits = F.linear(normalized, weight.to(device=normalized.device).float()).reshape(-1)
        greater += int((logits > target_logit).sum().item())
        equal += int((logits == target_logit).sum().item())
        chunk_count += 1

    return {
        "ready": True,
        "target_logit": target_logit,
        "target_rank": int(greater + 1),
        "equal_logit_count": int(equal),
        "chunk_rows": int(rows_per_chunk),
        "chunk_count": int(chunk_count),
        "loaded_head_bytes": int(loaded_bytes),
        "blockers": [],
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Record exact target-token ranks from DeepSeek V3's MTP head.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--sequence-token-ids", required=True)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    out_path = Path(args.out)
    env = apply_fp8_exact_mtp_default_env()
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    target_ids = _parse_token_ids(args.sequence_token_ids)
    target_text, target_text_blockers = decode_token_ids_to_text(args.model_id, target_ids)
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "prepared_prompt": prepared.prepared_prompt,
        "sequence_token_ids": list(target_ids),
        "sequence_text": target_text,
        "sequence_text_blockers": list(target_text_blockers),
        "env": env,
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "steps": [],
        "ready": False,
    }
    _write(out_path, payload)
    if not prepared.ready or not target_ids:
        return 1

    prefill_started = time.perf_counter()
    prefill = run_fp8_prompt_prefill(
        args.model_id,
        list(prepared.token_ids),
        layer_count=args.layers,
        include_tail=True,
    )
    payload.update(
        {
            "prompt_token_count": len(prepared.token_ids),
            "layer_count": int(prefill.layer_count),
            "prefill_ready": bool(prefill.ready),
            "prefill_elapsed_seconds": round(time.perf_counter() - prefill_started, 4),
            "prefill_top_token_ids": [] if prefill.tail is None else list(prefill.tail.top_token_ids),
            "prefill_blockers": list(prefill.blockers),
        }
    )
    _write(out_path, payload)
    if not prefill.ready or prefill.output_tensor is None:
        return 1

    config = _load_deepseek_config(args.model_id)
    mtp_layer = int(config["num_hidden_layers"])
    norm_name = f"model.layers.{mtp_layer}.shared_head.norm.weight"
    head_name = f"model.layers.{mtp_layer}.shared_head.head.weight"
    previous_hidden = prefill.output_tensor[:, -1:, :]
    mtp_cache = None
    input_token = int(prepared.token_ids[-1])
    position = len(prepared.token_ids) - 1
    blockers: list[str] = []
    for index, target_token_id in enumerate(target_ids):
        step_started = time.perf_counter()
        step = run_fp8_mtp_draft_step(
            args.model_id,
            input_token,
            previous_hidden,
            position=position,
            previous_kv_cache=mtp_cache,
            top_k=1,
            include_tail=False,
        )
        blockers.extend(step.blockers)
        rank_payload = (
            _target_rank(
                args.model_id,
                step.output_tensor,
                target_token_id=int(target_token_id),
                norm_name=norm_name,
                head_name=head_name,
            )
            if step.ready and step.output_tensor is not None
            else {"ready": False, "blockers": list(step.blockers)}
        )
        blockers.extend(rank_payload.get("blockers", []))
        step_payload = step.to_dict()
        step_payload.update(
            {
                "step_index": int(index + 1),
                "target_token_id": int(target_token_id),
                "target_rank": rank_payload.get("target_rank"),
                "target_logit": rank_payload.get("target_logit"),
                "rank_elapsed_seconds": rank_payload.get("elapsed_seconds"),
                "rank_ready": bool(rank_payload.get("ready")),
                "step_wall_seconds": round(time.perf_counter() - step_started, 4),
            }
        )
        payload["steps"].append(step_payload)
        payload.update(
            {
                "rank_sequence": [item.get("target_rank") for item in payload["steps"]],
                "elapsed_seconds": round(time.perf_counter() - started, 4),
                "blockers": list(blockers),
            }
        )
        _write(out_path, payload)
        if not step.ready or step.output_tensor is None or not rank_payload.get("ready"):
            break
        input_token = int(target_token_id)
        previous_hidden = step.output_tensor
        mtp_cache = step.next_kv_cache
        position += 1

    payload["ready"] = bool(not blockers and len(payload["steps"]) == len(target_ids))
    payload["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    _write(out_path, payload)
    print(
        json.dumps(
            {
                "ready": payload["ready"],
                "rank_sequence": payload.get("rank_sequence", []),
                "elapsed_seconds": payload["elapsed_seconds"],
            },
            indent=2,
        )
    )
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
