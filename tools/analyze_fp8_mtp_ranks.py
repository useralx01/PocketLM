from __future__ import annotations

import argparse
import json
from pathlib import Path


def _rank(token_id: int, top_ids: list[int]) -> int | None:
    for index, value in enumerate(top_ids):
        if int(value) == int(token_id):
            return int(index + 1)
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze verifier-token ranks inside recorded DeepSeek MTP top-k steps.")
    parser.add_argument("proof_json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    proof = json.loads(Path(args.proof_json).read_text(encoding="utf-8"))
    rows = []
    rank_counts: dict[str, int] = {}
    missing = 0
    total = 0
    for pass_payload in proof.get("passes") or []:
        verifier_ids = [int(value) for value in pass_payload.get("verifier_token_ids") or []]
        steps = list(pass_payload.get("mtp_steps") or [])
        for step_index, step in enumerate(steps):
            if step_index >= len(verifier_ids):
                break
            token_id = int(verifier_ids[step_index])
            top_ids = [int(value) for value in step.get("top_token_ids") or []]
            rank = _rank(token_id, top_ids)
            total += 1
            if rank is None:
                missing += 1
                rank_counts["missing"] = rank_counts.get("missing", 0) + 1
            else:
                rank_counts[str(rank)] = rank_counts.get(str(rank), 0) + 1
            rows.append(
                {
                    "pass_index": int(pass_payload.get("pass_index") or 0),
                    "step_index": int(step_index),
                    "verifier_token_id": int(token_id),
                    "mtp_top_token_ids": top_ids,
                    "rank": rank,
                    "accepted_in_pass": int(pass_payload.get("accepted") or 0),
                }
            )

    payload = {
        "source_proof": str(Path(args.proof_json)),
        "model_id": proof.get("model_id"),
        "strategy": proof.get("strategy"),
        "total_compared_steps": total,
        "missing_from_mtp_topk": missing,
        "rank_counts": rank_counts,
        "rows": rows,
        "tree_useful": bool(total and missing < total),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ["total_compared_steps", "missing_from_mtp_topk", "rank_counts", "tree_useful"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
