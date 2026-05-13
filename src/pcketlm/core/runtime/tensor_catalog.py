"""Tensor-aware cataloging for sharded safetensors model sources."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pcketlm.core.runtime.source import describe_runtime_source
from pcketlm.core.storage.paths import streaming_model_root

FP8_DTYPES = {"F8_E4M3", "F8_E4M3FN", "F8_E5M2"}
SCALE_SUFFIXES = (".scale_inv", "_scale_inv", ".scale", "_scale")
DTYPE_ALIASES = {
    "BFLOAT16": "BF16",
    "FLOAT16": "F16",
    "FP16": "F16",
    "FLOAT32": "F32",
    "FP32": "F32",
    "FP8_E4M3": "F8_E4M3",
    "FP8_E4M3FN": "F8_E4M3FN",
    "FLOAT8_E4M3": "F8_E4M3",
    "FLOAT8_E4M3FN": "F8_E4M3FN",
    "FP8_E5M2": "F8_E5M2",
    "FLOAT8_E5M2": "F8_E5M2",
}


@dataclass(slots=True)
class TensorCatalogEntry:
    """One tensor entry discovered from a safetensors shard header."""

    tensor_name: str
    shard_name: str
    shard_path: Path
    dtype: str
    shape: list[int]
    data_offset_start: int
    data_offset_end: int
    data_nbytes: int
    layer_index: int | None
    component_group: str
    expert_index: int | None = None
    tensor_role: str = "weight"
    scale_tensor_name: str | None = None
    weight_tensor_name: str | None = None
    physical_format: str = "safetensors"

    def to_dict(self) -> dict:
        return {
            "tensor_name": self.tensor_name,
            "shard_name": self.shard_name,
            "shard_path": str(self.shard_path),
            "dtype": self.dtype,
            "shape": list(self.shape),
            "data_offset_start": self.data_offset_start,
            "data_offset_end": self.data_offset_end,
            "data_nbytes": self.data_nbytes,
            "layer_index": self.layer_index,
            "expert_index": self.expert_index,
            "component_group": self.component_group,
            "tensor_role": self.tensor_role,
            "scale_tensor_name": self.scale_tensor_name,
            "weight_tensor_name": self.weight_tensor_name,
            "physical_format": self.physical_format,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorCatalogEntry":
        return cls(
            tensor_name=str(payload.get("tensor_name", "")),
            shard_name=str(payload.get("shard_name", "")),
            shard_path=Path(payload.get("shard_path", "")),
            dtype=str(payload.get("dtype", "unknown")),
            shape=[int(value) for value in payload.get("shape", [])],
            data_offset_start=int(payload.get("data_offset_start", 0)),
            data_offset_end=int(payload.get("data_offset_end", 0)),
            data_nbytes=int(payload.get("data_nbytes", 0)),
            layer_index=payload.get("layer_index"),
            expert_index=payload.get("expert_index"),
            component_group=str(payload.get("component_group", "other")),
            tensor_role=str(payload.get("tensor_role", "weight")),
            scale_tensor_name=payload.get("scale_tensor_name"),
            weight_tensor_name=payload.get("weight_tensor_name"),
            physical_format=str(payload.get("physical_format", "safetensors")),
        )


@dataclass(slots=True)
class TensorCatalog:
    """Persisted tensor-aware view of a sharded safetensors source."""

    model_id: str
    model_dir: Path
    catalog_path: Path
    tensor_count: int
    shard_count: int
    layer_count: int
    hidden_size: int | None = None
    num_hidden_layers: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None
    vocab_size: int | None = None
    intermediate_size: int | None = None
    num_experts: int | None = None
    num_experts_per_tok: int | None = None
    moe_intermediate_size: int | None = None
    model_type: str | None = None
    dtype_counts: dict[str, int] = field(default_factory=dict)
    component_group_counts: dict[str, int] = field(default_factory=dict)
    tensor_role_counts: dict[str, int] = field(default_factory=dict)
    fp8_weight_count: int = 0
    fp8_scale_count: int = 0
    fp8_pair_count: int = 0
    fp8_weight_bytes: int = 0
    fp8_scale_bytes: int = 0
    tensors: list[TensorCatalogEntry] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "model_dir": str(self.model_dir),
            "catalog_path": str(self.catalog_path),
            "tensor_count": self.tensor_count,
            "shard_count": self.shard_count,
            "layer_count": self.layer_count,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "intermediate_size": self.intermediate_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "moe_intermediate_size": self.moe_intermediate_size,
            "model_type": self.model_type,
            "dtype_counts": dict(self.dtype_counts),
            "component_group_counts": dict(self.component_group_counts),
            "tensor_role_counts": dict(self.tensor_role_counts),
            "fp8_weight_count": self.fp8_weight_count,
            "fp8_scale_count": self.fp8_scale_count,
            "fp8_pair_count": self.fp8_pair_count,
            "fp8_weight_bytes": self.fp8_weight_bytes,
            "fp8_scale_bytes": self.fp8_scale_bytes,
            "tensors": [entry.to_dict() for entry in self.tensors],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorCatalog":
        return cls(
            model_id=str(payload.get("model_id", "")),
            model_dir=Path(payload.get("model_dir", "")),
            catalog_path=Path(payload.get("catalog_path", "")),
            tensor_count=int(payload.get("tensor_count", 0)),
            shard_count=int(payload.get("shard_count", 0)),
            layer_count=int(payload.get("layer_count", 0)),
            hidden_size=_optional_int(payload.get("hidden_size")),
            num_hidden_layers=_optional_int(payload.get("num_hidden_layers")),
            num_attention_heads=_optional_int(payload.get("num_attention_heads")),
            num_key_value_heads=_optional_int(payload.get("num_key_value_heads")),
            vocab_size=_optional_int(payload.get("vocab_size")),
            intermediate_size=_optional_int(payload.get("intermediate_size")),
            num_experts=_optional_int(payload.get("num_experts")),
            num_experts_per_tok=_optional_int(payload.get("num_experts_per_tok")),
            moe_intermediate_size=_optional_int(payload.get("moe_intermediate_size")),
            model_type=payload.get("model_type"),
            dtype_counts={str(key): int(value) for key, value in (payload.get("dtype_counts") or {}).items()},
            component_group_counts={
                str(key): int(value) for key, value in (payload.get("component_group_counts") or {}).items()
            },
            tensor_role_counts={
                str(key): int(value) for key, value in (payload.get("tensor_role_counts") or {}).items()
            },
            fp8_weight_count=int(payload.get("fp8_weight_count", 0)),
            fp8_scale_count=int(payload.get("fp8_scale_count", 0)),
            fp8_pair_count=int(payload.get("fp8_pair_count", 0)),
            fp8_weight_bytes=int(payload.get("fp8_weight_bytes", 0)),
            fp8_scale_bytes=int(payload.get("fp8_scale_bytes", 0)),
            tensors=[TensorCatalogEntry.from_dict(item) for item in payload.get("tensors", [])],
            blockers=list(payload.get("blockers", [])),
            ready=bool(payload.get("ready", False)),
        )


def tensor_catalog_path(model_id: str) -> Path:
    """Return the persisted tensor catalog path for one model."""
    return streaming_model_root(model_id) / "tensor-catalog.json"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def canonical_dtype(dtype: str) -> str:
    """Normalize safetensors dtype spelling for runtime planning."""
    normalized = str(dtype).strip()
    upper = normalized.upper()
    return DTYPE_ALIASES.get(normalized, DTYPE_ALIASES.get(upper, upper))


def is_fp8_dtype(dtype: str) -> bool:
    """Return whether a catalog dtype is an FP8 physical weight dtype."""
    return canonical_dtype(dtype) in FP8_DTYPES


def scale_partner_name(tensor_name: str, all_tensor_names: set[str]) -> str | None:
    """Return the weight tensor paired with a scale tensor, if present."""
    for suffix in SCALE_SUFFIXES:
        if tensor_name.endswith(suffix):
            partner = tensor_name[: -len(suffix)]
            if partner in all_tensor_names:
                return partner
    return None


def _model_config_values(model_dir: Path) -> dict[str, int | None]:
    payload = _read_json(model_dir / "config.json")
    num_experts = _optional_int(
        payload.get("num_experts", payload.get("num_local_experts", payload.get("n_routed_experts")))
    )
    moe_intermediate_size = _optional_int(payload.get("moe_intermediate_size"))
    if moe_intermediate_size is None and num_experts:
        moe_intermediate_size = _optional_int(payload.get("intermediate_size"))
    return {
        "hidden_size": _optional_int(payload.get("hidden_size")),
        "num_hidden_layers": _optional_int(payload.get("num_hidden_layers")),
        "num_attention_heads": _optional_int(payload.get("num_attention_heads")),
        "num_key_value_heads": _optional_int(payload.get("num_key_value_heads")),
        "vocab_size": _optional_int(payload.get("vocab_size")),
        "intermediate_size": _optional_int(payload.get("intermediate_size")),
        "num_experts": num_experts,
        "num_experts_per_tok": _optional_int(payload.get("num_experts_per_tok")),
        "moe_intermediate_size": moe_intermediate_size,
        "model_type": payload.get("model_type"),
    }


def _read_safetensors_header(shard_path: Path) -> dict:
    with shard_path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header_payload = handle.read(header_length).decode("utf-8")
    return json.loads(header_payload)


def _layer_index_for_tensor(tensor_name: str) -> int | None:
    prefix = "model.layers."
    if not tensor_name.startswith(prefix):
        return None
    remainder = tensor_name[len(prefix) :]
    layer_part = remainder.split(".", 1)[0]
    return int(layer_part) if layer_part.isdigit() else None


def _expert_index_for_tensor(tensor_name: str) -> int | None:
    for marker in (".mlp.experts.", ".block_sparse_moe.experts."):
        if marker not in tensor_name:
            continue
        remainder = tensor_name.split(marker, 1)[1]
        expert_part = remainder.split(".", 1)[0]
        return int(expert_part) if expert_part.isdigit() else None
    return None


def _component_group_for_tensor(tensor_name: str) -> str:
    if tensor_name.startswith("model.embed_tokens."):
        return "embeddings"
    if tensor_name.startswith("lm_head."):
        return "lm_head"
    if tensor_name.startswith("model.norm."):
        return "final_norm"
    if ".mlp.experts." in tensor_name or ".block_sparse_moe.experts." in tensor_name:
        return "expert_mlp"
    if ".mlp.gate." in tensor_name or ".block_sparse_moe.gate." in tensor_name:
        return "router"
    if ".self_attn." in tensor_name:
        return "attention"
    if ".mlp." in tensor_name:
        return "mlp"
    if ".input_layernorm." in tensor_name or ".post_attention_layernorm." in tensor_name:
        return "layer_norm"
    if "rotary_emb" in tensor_name:
        return "rotary"
    return "other"


def _empty_catalog(model_id: str, model_dir: Path, blockers: list[str]) -> TensorCatalog:
    return TensorCatalog(
        model_id=model_id,
        model_dir=model_dir,
        catalog_path=tensor_catalog_path(model_id),
        tensor_count=0,
        shard_count=0,
        layer_count=0,
        blockers=blockers,
        ready=False,
    )


def build_tensor_catalog(model_id: str, model_dir: Path) -> TensorCatalog:
    """Build and persist a tensor-aware catalog from a sharded safetensors source."""
    catalog_path = tensor_catalog_path(model_id)
    source = describe_runtime_source(model_dir)
    blockers = list(source.missing_runtime_files)

    if not source.ready or not source.index_path:
        return _empty_catalog(
            model_id,
            model_dir,
            blockers or ["Runtime source is not ready enough to build a tensor catalog."],
        )

    index_payload = _read_json(source.index_path)
    config_values = _model_config_values(model_dir)
    weight_map = index_payload.get("weight_map") or {}
    if not weight_map:
        return _empty_catalog(model_id, model_dir, ["Safetensors index does not contain a weight map."])

    tensors: list[TensorCatalogEntry] = []
    dtype_counts: dict[str, int] = {}
    component_group_counts: dict[str, int] = {}
    tensor_role_counts: dict[str, int] = {}
    scale_partners: dict[str, str] = {}
    fp8_weight_count = 0
    fp8_scale_count = 0
    fp8_weight_bytes = 0
    fp8_scale_bytes = 0
    seen_layers: set[int] = set()
    all_tensor_names = set(weight_map)
    for tensor_name in all_tensor_names:
        partner = scale_partner_name(tensor_name, all_tensor_names)
        if partner is not None:
            scale_partners[tensor_name] = partner

    for shard_path in source.shard_paths:
        header = _read_safetensors_header(shard_path)
        shard_name = shard_path.name
        for tensor_name, tensor_meta in header.items():
            if tensor_name == "__metadata__":
                continue
            mapped_shard_name = weight_map.get(tensor_name)
            if mapped_shard_name != shard_name:
                blockers.append(
                    f"Tensor {tensor_name} is mapped to {mapped_shard_name or 'unknown'} in the index but found in {shard_name}."
                )
                continue

            offsets = tensor_meta.get("data_offsets") or [0, 0]
            layer_index = _layer_index_for_tensor(tensor_name)
            expert_index = _expert_index_for_tensor(tensor_name)
            if layer_index is not None:
                seen_layers.add(layer_index)
            component_group = _component_group_for_tensor(tensor_name)
            dtype = canonical_dtype(str(tensor_meta.get("dtype", "unknown")))
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
            component_group_counts[component_group] = component_group_counts.get(component_group, 0) + 1
            scale_for_weight = None
            tensor_role = "weight"
            weight_partner = None
            if tensor_name in scale_partners:
                tensor_role = "scale_companion"
                weight_partner = scale_partners[tensor_name]
            else:
                for scale_name, partner_name in scale_partners.items():
                    if partner_name == tensor_name:
                        scale_for_weight = scale_name
                        break
            tensor_role_counts[tensor_role] = tensor_role_counts.get(tensor_role, 0) + 1
            data_nbytes = int(offsets[1]) - int(offsets[0])
            if is_fp8_dtype(dtype) and tensor_role == "weight":
                fp8_weight_count += 1
                fp8_weight_bytes += data_nbytes
            if tensor_role == "scale_companion":
                fp8_scale_count += 1
                fp8_scale_bytes += data_nbytes
            tensors.append(
                TensorCatalogEntry(
                    tensor_name=tensor_name,
                    shard_name=shard_name,
                    shard_path=shard_path,
                    dtype=dtype,
                    shape=[int(value) for value in tensor_meta.get("shape", [])],
                    data_offset_start=int(offsets[0]),
                    data_offset_end=int(offsets[1]),
                    data_nbytes=data_nbytes,
                    layer_index=layer_index,
                    expert_index=expert_index,
                    component_group=component_group,
                    tensor_role=tensor_role,
                    scale_tensor_name=scale_for_weight,
                    weight_tensor_name=weight_partner,
                    physical_format="fp8_block_scaled" if is_fp8_dtype(dtype) and scale_for_weight else "safetensors",
                )
            )

    blockers = list(dict.fromkeys(blockers))
    catalog = TensorCatalog(
        model_id=model_id,
        model_dir=model_dir,
        catalog_path=catalog_path,
        tensor_count=len(tensors),
        shard_count=len({entry.shard_name for entry in tensors}),
        layer_count=len(seen_layers),
        **config_values,
        dtype_counts=dtype_counts,
        component_group_counts=component_group_counts,
        tensor_role_counts=tensor_role_counts,
        fp8_weight_count=fp8_weight_count,
        fp8_scale_count=fp8_scale_count,
        fp8_pair_count=len(scale_partners),
        fp8_weight_bytes=fp8_weight_bytes,
        fp8_scale_bytes=fp8_scale_bytes,
        tensors=tensors,
        blockers=blockers,
        ready=bool(tensors) and not blockers,
    )
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.write_text(json.dumps(catalog.to_dict(), indent=2), encoding="utf-8")
    return catalog


def load_tensor_catalog(model_id: str) -> TensorCatalog:
    """Load a persisted tensor-aware catalog for one model."""
    catalog_path = tensor_catalog_path(model_id)
    if not catalog_path.exists():
        return TensorCatalog(
            model_id=model_id,
            model_dir=Path(),
            catalog_path=catalog_path,
            tensor_count=0,
            shard_count=0,
            layer_count=0,
            blockers=["Tensor catalog does not exist yet."],
            ready=False,
        )

    mtime_ns = catalog_path.stat().st_mtime_ns
    payload = _load_tensor_catalog_payload(str(catalog_path), mtime_ns)
    payload["catalog_path"] = str(catalog_path)
    return TensorCatalog.from_dict(payload)


def load_tensor_entry_index(model_id: str) -> dict[str, TensorCatalogEntry]:
    """Load a name-indexed tensor catalog with mtime-based invalidation."""
    catalog_path = tensor_catalog_path(model_id)
    if not catalog_path.exists():
        return {}
    mtime_ns = catalog_path.stat().st_mtime_ns
    return _load_tensor_entry_index(str(catalog_path), mtime_ns)


def find_tensor_catalog_entry(model_id: str, tensor_name: str) -> TensorCatalogEntry | None:
    """Find one tensor entry without repeatedly scanning the catalog list."""
    return load_tensor_entry_index(model_id).get(tensor_name)


@lru_cache(maxsize=16)
def _load_tensor_catalog_payload(catalog_path: str, mtime_ns: int) -> dict:
    """Load persisted catalog JSON with automatic invalidation when the file changes."""
    del mtime_ns
    return _read_json(Path(catalog_path))


@lru_cache(maxsize=16)
def _load_tensor_entry_index(catalog_path: str, mtime_ns: int) -> dict[str, TensorCatalogEntry]:
    """Load persisted catalog entries as a dict with automatic invalidation."""
    del mtime_ns
    payload = _read_json(Path(catalog_path))
    if not payload.get("ready", False):
        return {}
    return {
        str(item.get("tensor_name", "")): TensorCatalogEntry.from_dict(item)
        for item in payload.get("tensors", [])
        if item.get("tensor_name")
    }
