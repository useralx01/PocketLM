"""Source model inspection helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ConfigSummary:
    """Subset of model config values useful to pcketlm."""

    architecture: str | None = None
    model_type: str | None = None
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    intermediate_size: int | None = None
    moe_intermediate_size: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    max_position_embeddings: int | None = None
    vocab_size: int | None = None
    torch_dtype: str | None = None

    def to_dict(self) -> dict:
        """Serialize the config summary."""
        return {
            "architecture": self.architecture,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "intermediate_size": self.intermediate_size,
            "moe_intermediate_size": self.moe_intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
            "torch_dtype": self.torch_dtype,
        }


@dataclass(slots=True)
class SourceInspection:
    """Inspected metadata for a source model folder."""

    config: ConfigSummary
    format_name: str
    index_present: bool
    expected_shards: int
    present_shards: int
    total_size_bytes: int | None = None


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def inspect_qwen_source(model_dir: Path) -> SourceInspection:
    """Inspect a Qwen-family source directory."""
    config_payload = _read_json(model_dir / "config.json") or {}
    architectures = config_payload.get("architectures") or []
    config = ConfigSummary(
        architecture=architectures[0] if architectures else None,
        model_type=config_payload.get("model_type"),
        hidden_size=config_payload.get("hidden_size"),
        num_hidden_layers=config_payload.get("num_hidden_layers"),
        num_attention_heads=config_payload.get("num_attention_heads"),
        num_key_value_heads=config_payload.get("num_key_value_heads"),
        intermediate_size=config_payload.get("intermediate_size"),
        moe_intermediate_size=config_payload.get("moe_intermediate_size"),
        num_experts=config_payload.get("num_experts"),
        num_experts_per_tok=config_payload.get("num_experts_per_tok"),
        max_position_embeddings=config_payload.get("max_position_embeddings"),
        vocab_size=config_payload.get("vocab_size"),
        torch_dtype=config_payload.get("torch_dtype"),
    )

    index_payload = _read_json(model_dir / "model.safetensors.index.json")
    if not index_payload:
        return SourceInspection(
            config=config,
            format_name="unknown",
            index_present=False,
            expected_shards=0,
            present_shards=0,
            total_size_bytes=None,
        )

    weight_map = index_payload.get("weight_map") or {}
    shard_names = sorted(set(weight_map.values()))
    present_shards = sum(1 for shard in shard_names if (model_dir / shard).exists())
    metadata = index_payload.get("metadata") or {}
    return SourceInspection(
        config=config,
        format_name="safetensors-sharded",
        index_present=True,
        expected_shards=len(shard_names),
        present_shards=present_shards,
        total_size_bytes=metadata.get("total_size"),
    )
