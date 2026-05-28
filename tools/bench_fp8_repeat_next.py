from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pcketlm.core.runtime.speculative import FP8CachedVerifierSession
from pcketlm.core.runtime.fp8_source import _load_deepseek_config
from pcketlm.core.runtime.tokenizer_runtime import decode_token_ids_to_text, load_generation_settings, prepare_prompt_text


def _text_quality(text: str, token_ids: list[int]) -> dict:
    non_whitespace = sum(1 for char in text if not char.isspace())
    unique_tokens = len(set(int(value) for value in token_ids))
    return {
        "non_whitespace_chars": non_whitespace,
        "unique_token_count": unique_tokens,
        "is_nonblank": non_whitespace > 0,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _visible_token_ids(token_ids: list[int], eos_token_ids: set[int]) -> tuple[list[int], bool]:
    visible: list[int] = []
    for token_id in token_ids:
        if int(token_id) in eos_token_ids:
            return visible, True
        visible.append(int(token_id))
    return visible, False


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark exact DeepSeek FP8 repeat-next chunk verification.")
    parser.add_argument("model_id")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new", type=int, default=96)
    parser.add_argument("--k", type=int, default=96)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-non-whitespace", type=int, default=8)
    parser.add_argument("--min-unique-tokens", type=int, default=3)
    args = parser.parse_args()

    out_path = Path(args.out)
    started = time.perf_counter()
    prepared = prepare_prompt_text(args.model_id, args.prompt, apply_chat_format=True)
    configured_layers = int(args.layers) if args.layers is not None else int(_load_deepseek_config(args.model_id)["num_hidden_layers"])
    eos_token_ids = {int(value) for value in load_generation_settings(args.model_id).eos_token_ids}
    payload: dict = {
        "model_id": args.model_id,
        "prompt": args.prompt,
        "max_new_tokens": int(args.max_new),
        "k": int(args.k),
        "layer_count": configured_layers,
        "ready": False,
        "prepared_ready": bool(prepared.ready),
        "prepared_blockers": list(prepared.blockers),
        "passes": [],
    }
    _write(out_path, payload)
    if not prepared.ready:
        return 1

    session = FP8CachedVerifierSession(args.model_id, list(prepared.token_ids), layer_count=configured_layers)
    payload.update(
        {
            "session_blockers": list(session.blockers),
            "prefill_elapsed_seconds": round(time.perf_counter() - started, 4),
            "session_layers_executed": int(session.layers_executed),
            "session_expected_layers_executed": int(session.expected_layers_executed),
        }
    )
    _write(out_path, payload)
    if session.blockers:
        return 1

    generated: list[int] = []
    accepted = 0
    corrected = 0
    pending_correction: int | None = None
    layers_executed = int(session.layers_executed)
    expected_layers = int(session.expected_layers_executed)
    while len(generated) < int(args.max_new):
        remaining = int(args.max_new) - len(generated)
        batch_k = min(int(args.k), remaining)
        candidates = [int(session.next_token_id)] * batch_k
        pass_started = time.perf_counter()
        verification = session.verify(candidates)
        pass_wall = round(time.perf_counter() - pass_started, 4)
        layers_executed += int(verification.layers_executed)
        expected_layers += int(verification.expected_layers_executed)
        accepted_this_pass = 0
        for index, candidate in enumerate(candidates):
            if int(candidate) != int(verification.verifier_token_ids[index]):
                break
            if pending_correction is not None and accepted_this_pass == 0 and int(candidate) == int(pending_correction):
                pending_correction = None
            else:
                generated.append(int(candidate))
            accepted += 1
            accepted_this_pass += 1
            if int(candidate) in eos_token_ids:
                break
            if len(generated) >= int(args.max_new):
                break
        session.accept_prefix(accepted_this_pass)
        corrected_token = None
        stopped_by_eos = bool(generated and int(generated[-1]) in eos_token_ids)
        if len(generated) < int(args.max_new) and not stopped_by_eos:
            corrected_token = int(verification.verifier_token_ids[accepted_this_pass])
            generated.append(corrected_token)
            pending_correction = corrected_token
            corrected += 1
            stopped_by_eos = corrected_token in eos_token_ids
        payload["passes"].append(
            {
                "pass_index": len(payload["passes"]) + 1,
                "batch_k": batch_k,
                "next_token_id": int(candidates[0]),
                "accepted_this_pass": accepted_this_pass,
                "corrected_token_id": corrected_token,
                "verifier_head": [int(value) for value in verification.verifier_token_ids[:16]],
                "verify_wall_seconds": pass_wall,
                "verify_elapsed_seconds": float(verification.elapsed_seconds),
                "layers_executed": int(verification.layers_executed),
                "expected_layers_executed": int(verification.expected_layers_executed),
                "ready": bool(verification.ready),
                "blockers": list(verification.blockers),
            }
        )
        elapsed = time.perf_counter() - started
        visible_ids, stopped_by_eos = _visible_token_ids(generated, eos_token_ids)
        generated_text, text_blockers = decode_token_ids_to_text(args.model_id, visible_ids)
        raw_generated_text, raw_text_blockers = decode_token_ids_to_text(args.model_id, generated)
        quality = _text_quality(generated_text, visible_ids)
        quality["min_non_whitespace_chars"] = int(args.min_non_whitespace)
        quality["min_unique_token_count"] = int(args.min_unique_tokens)
        quality["is_complete_short_answer"] = bool(
            stopped_by_eos and int(quality["non_whitespace_chars"]) >= 3 and bool(visible_ids)
        )
        quality["is_useful_text"] = bool(
            int(quality["non_whitespace_chars"]) >= int(args.min_non_whitespace)
            and int(quality["unique_token_count"]) >= int(args.min_unique_tokens)
        ) or bool(quality["is_complete_short_answer"])
        payload.update(
            {
                "ready": bool(
                    not session.blockers
                    and not text_blockers
                    and (len(generated) >= int(args.max_new) or stopped_by_eos)
                    and quality["is_useful_text"]
                ),
                "generated_token_ids": list(generated),
                "visible_token_ids": list(visible_ids),
                "generated_text": generated_text,
                "raw_generated_text": raw_generated_text,
                "stopped_by_eos": bool(stopped_by_eos),
                "text_quality": quality,
                "generated_tokens": len(generated),
                "accepted_token_count": accepted,
                "corrected_token_count": corrected,
                "elapsed_seconds": round(elapsed, 4),
                "seconds_per_visible_token": round(elapsed / max(1, len(visible_ids)), 4),
                "verify_seconds_per_accepted_token": None if accepted <= 0 else round(sum(p["verify_elapsed_seconds"] for p in payload["passes"]) / accepted, 4),
                "layers_executed": layers_executed,
                "expected_layers_executed": expected_layers,
                "anti_cheat_passed": layers_executed == expected_layers,
                "blockers": list(session.blockers)
                + list(text_blockers)
                + list(raw_text_blockers)
                + ([] if quality["is_nonblank"] else ["Generated text is blank/whitespace only."])
                + ([] if quality["is_useful_text"] else ["Generated text did not meet useful-text gate."]),
            }
        )
        _write(out_path, payload)
        if not verification.ready:
            return 1
        if stopped_by_eos:
            break
    return 0 if payload["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
