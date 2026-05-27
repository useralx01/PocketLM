from __future__ import annotations

import argparse
import json
from pathlib import Path


def _quality(text: str, token_ids: list[int], *, min_non_whitespace: int, min_unique_tokens: int) -> dict:
    non_whitespace = sum(1 for char in text if not char.isspace())
    unique_tokens = len(set(int(value) for value in token_ids))
    return {
        "non_whitespace_chars": non_whitespace,
        "unique_token_count": unique_tokens,
        "min_non_whitespace_chars": int(min_non_whitespace),
        "min_unique_token_count": int(min_unique_tokens),
        "is_useful_text": non_whitespace >= int(min_non_whitespace) and unique_tokens >= int(min_unique_tokens),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze whether an FP8 generation proof produced useful visible text.")
    parser.add_argument("json_path")
    parser.add_argument("--min-non-whitespace", type=int, default=8)
    parser.add_argument("--min-unique-tokens", type=int, default=3)
    args = parser.parse_args()

    payload = json.loads(Path(args.json_path).read_text(encoding="utf-8"))
    token_ids = [int(value) for value in payload.get("generated_token_ids", [])]
    text = str(payload.get("generated_text") or "")
    quality = _quality(
        text,
        token_ids,
        min_non_whitespace=int(args.min_non_whitespace),
        min_unique_tokens=int(args.min_unique_tokens),
    )
    result = {
        "source": str(args.json_path),
        "ready": bool(payload.get("ready")),
        "generated_tokens": int(payload.get("generated_tokens") or len(token_ids)),
        "seconds_per_visible_token": payload.get("seconds_per_visible_token"),
        "accepted_token_count": payload.get("accepted_token_count"),
        "corrected_token_count": payload.get("corrected_token_count"),
        "text_preview": text[:120],
        "quality": quality,
        "target_10s_useful_passed": bool(
            payload.get("seconds_per_visible_token") is not None
            and float(payload["seconds_per_visible_token"]) <= 10.0
            and quality["is_useful_text"]
        ),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["target_10s_useful_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
