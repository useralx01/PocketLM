"""Planning-only optimized artifact manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog
from pcketlm.core.storage.paths import artifacts_root


DEFAULT_SMALL_PACK_MAX_BYTES = 1 * 1024 * 1024
DEFAULT_FRONT_ATTENTION_PACK_LAYERS = 2
DEFAULT_FRONT_KV_PACK_LAYERS = 13
DEFAULT_PROJECTION_PACK_MAX_BYTES = 64 * 1024 * 1024
DEFAULT_RUNTIME_PACK_BUDGET_BYTES = 512 * 1024 * 1024
STANDARD_RUNTIME_PACK_BUDGET_BYTES = 192 * 1024 * 1024
BOOSTED_RUNTIME_PACK_BUDGET_BYTES = 384 * 1024 * 1024


@dataclass(slots=True)
class OptimizedArtifactManifest:
    """A reversible plan for a future derived runtime artifact."""

    model_id: str
    artifact_id: str
    strategy: str
    profile_id: str | None
    created_at: str
    artifact_path: Path
    source_tensor_count: int
    source_shard_count: int
    source_layer_count: int
    tensor_pack_path: Path | None = None
    packed_tensor_count: int = 0
    packed_nbytes: int = 0
    packed_tensor_names: list[str] = field(default_factory=list)
    planned_outputs: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "artifact_id": self.artifact_id,
            "strategy": self.strategy,
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "artifact_path": str(self.artifact_path),
            "source_tensor_count": self.source_tensor_count,
            "source_shard_count": self.source_shard_count,
            "source_layer_count": self.source_layer_count,
            "tensor_pack_path": None if self.tensor_pack_path is None else str(self.tensor_pack_path),
            "packed_tensor_count": self.packed_tensor_count,
            "packed_nbytes": self.packed_nbytes,
            "packed_mb": round(self.packed_nbytes / (1024**2), 2),
            "packed_tensor_names": list(self.packed_tensor_names),
            "planned_outputs": list(self.planned_outputs),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "OptimizedArtifactManifest":
        return cls(
            model_id=str(payload.get("model_id", "")),
            artifact_id=str(payload.get("artifact_id", "")),
            strategy=str(payload.get("strategy", "")),
            profile_id=None if payload.get("profile_id") is None else str(payload.get("profile_id")),
            created_at=str(payload.get("created_at", "")),
            artifact_path=Path(payload.get("artifact_path", "")),
            source_tensor_count=int(payload.get("source_tensor_count", 0)),
            source_shard_count=int(payload.get("source_shard_count", 0)),
            source_layer_count=int(payload.get("source_layer_count", 0)),
            tensor_pack_path=None if payload.get("tensor_pack_path") is None else Path(payload.get("tensor_pack_path")),
            packed_tensor_count=int(payload.get("packed_tensor_count", 0)),
            packed_nbytes=int(payload.get("packed_nbytes", 0)),
            packed_tensor_names=list(payload.get("packed_tensor_names", [])),
            planned_outputs=list(payload.get("planned_outputs", [])),
            blockers=list(payload.get("blockers", [])),
            ready=bool(payload.get("ready", False)),
        )


def optimized_artifact_manifest_path(
    model_id: str,
    *,
    profile_id: str | None = None,
    strategy: str = "runtime-pack-plan",
) -> Path:
    """Return the manifest path for one planned optimized artifact."""
    suffix = f".{profile_id}" if profile_id else ""
    return artifacts_root(model_id) / f"{strategy}{suffix}.artifact.json"


def optimized_tensor_pack_path(
    model_id: str,
    *,
    profile_id: str | None = None,
    strategy: str = "runtime-pack-plan",
) -> Path:
    """Return the safetensors path for the small-tensor runtime pack."""
    suffix = f".{profile_id}" if profile_id else ""
    return artifacts_root(model_id) / f"{strategy}{suffix}.small.safetensors"


def _small_pack_entries(catalog, max_tensor_bytes: int) -> list:
    return [
        entry
        for entry in catalog.tensors
        if 0 < int(entry.data_nbytes) <= max_tensor_bytes
        and entry.component_group in {"layer_norm", "final_norm", "rotary", "other"}
    ]


def _front_attention_pack_entries(
    catalog,
    layer_count: int,
    max_tensor_bytes: int,
    *,
    include_output_projection: bool = False,
) -> list:
    if layer_count <= 0:
        return []
    markers = [".q_proj.", ".k_proj.", ".v_proj."]
    if include_output_projection:
        markers.append(".o_proj.")
    return [
        entry
        for entry in catalog.tensors
        if entry.layer_index is not None
        and entry.layer_index < layer_count
        and 0 < int(entry.data_nbytes) <= max_tensor_bytes
        and ".self_attn." in entry.tensor_name
        and any(marker in entry.tensor_name for marker in markers)
    ]


def _front_key_value_pack_entries(
    catalog,
    layer_count: int,
    max_tensor_bytes: int,
) -> list:
    if layer_count <= 0:
        return []
    markers = [".k_proj.", ".v_proj."]
    return [
        entry
        for entry in catalog.tensors
        if entry.layer_index is not None
        and entry.layer_index < layer_count
        and 0 < int(entry.data_nbytes) <= max_tensor_bytes
        and ".self_attn." in entry.tensor_name
        and any(marker in entry.tensor_name for marker in markers)
    ]


def _dedupe_entries(entries: list) -> list:
    by_name = {entry.tensor_name: entry for entry in entries}
    return [by_name[name] for name in sorted(by_name)]


def _limit_entries_to_budget(entries: list, max_pack_bytes: int | None) -> list:
    if max_pack_bytes is None or max_pack_bytes <= 0:
        return entries
    selected = []
    total_nbytes = 0
    for entry in entries:
        entry_nbytes = int(entry.data_nbytes)
        if selected and total_nbytes + entry_nbytes > max_pack_bytes:
            continue
        if not selected and entry_nbytes > max_pack_bytes:
            continue
        selected.append(entry)
        total_nbytes += entry_nbytes
    return selected


def _materialize_tensor_pack(catalog, pack_path: Path, entries: list) -> tuple[list[str], int, list[str]]:
    """Write a safetensors runtime pack from selected original source tensors."""
    blockers: list[str] = []
    tensors: dict[str, torch.Tensor] = {}
    entries_by_shard: dict[Path, list] = {}
    for entry in entries:
        entries_by_shard.setdefault(entry.shard_path, []).append(entry)

    for shard_path, shard_entries in entries_by_shard.items():
        try:
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for entry in shard_entries:
                    tensors[entry.tensor_name] = handle.get_tensor(entry.tensor_name).detach().cpu().clone()
        except Exception as exc:  # pragma: no cover - defensive IO path
            blockers.append(f"Failed to pack tensors from {shard_path.name}: {exc}")

    if blockers:
        return [], 0, blockers
    if not tensors:
        blockers.append("No small tensors matched the runtime-pack criteria.")
        return [], 0, blockers

    pack_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(pack_path))
    packed_nbytes = sum(tensor.element_size() * tensor.nelement() for tensor in tensors.values())
    return sorted(tensors), packed_nbytes, []


def build_optimized_artifact_manifest(
    model_id: str,
    *,
    profile_id: str | None = None,
    strategy: str = "runtime-pack-plan",
    materialize_small_pack: bool = False,
    materialize_front_attention_pack: bool = False,
    materialize_front_kv_pack: bool = False,
    small_pack_max_bytes: int = DEFAULT_SMALL_PACK_MAX_BYTES,
    front_attention_layers: int = DEFAULT_FRONT_ATTENTION_PACK_LAYERS,
    front_kv_layers: int = DEFAULT_FRONT_KV_PACK_LAYERS,
    projection_pack_max_bytes: int = DEFAULT_PROJECTION_PACK_MAX_BYTES,
    include_attention_output_projection: bool = False,
    max_pack_bytes: int | None = None,
) -> OptimizedArtifactManifest:
    """Build and persist an optimized artifact manifest, optionally with a small tensor pack."""
    catalog = load_tensor_catalog(model_id)
    artifact_path = optimized_artifact_manifest_path(model_id, profile_id=profile_id, strategy=strategy)
    tensor_pack_path = optimized_tensor_pack_path(model_id, profile_id=profile_id, strategy=strategy)
    blockers = list(catalog.blockers)
    if not catalog.ready:
        blockers.append("Tensor catalog must be ready before planning optimized artifacts.")
    packed_tensor_names: list[str] = []
    packed_nbytes = 0
    materialize_pack = materialize_small_pack or materialize_front_attention_pack or materialize_front_kv_pack
    if catalog.ready and materialize_pack:
        selected_entries = []
        if materialize_small_pack:
            selected_entries.extend(_small_pack_entries(catalog, small_pack_max_bytes))
        if materialize_front_attention_pack:
            selected_entries.extend(
                _front_attention_pack_entries(
                    catalog,
                    front_attention_layers,
                    projection_pack_max_bytes,
                    include_output_projection=include_attention_output_projection,
                )
            )
        if materialize_front_kv_pack:
            selected_entries.extend(
                _front_key_value_pack_entries(
                    catalog,
                    front_kv_layers,
                    projection_pack_max_bytes,
                )
            )
        selected_entries = _limit_entries_to_budget(_dedupe_entries(selected_entries), max_pack_bytes)
        packed_tensor_names, packed_nbytes, pack_blockers = _materialize_tensor_pack(
            catalog,
            tensor_pack_path,
            selected_entries,
        )
        blockers.extend(pack_blockers)
    planned_outputs = [
        "runtime tensor layout manifest",
        "profile-linked artifact metadata",
    ]
    if materialize_small_pack:
        planned_outputs.append("small repeated tensor safetensors pack")
    if materialize_front_attention_pack:
        projection_label = "Q/K/V/O" if include_attention_output_projection else "Q/K/V"
        planned_outputs.append(f"front {front_attention_layers} layer {projection_label} projection tensor pack")
    if materialize_front_kv_pack:
        planned_outputs.append(f"front {front_kv_layers} layer K/V projection tensor pack")
    if max_pack_bytes is not None and max_pack_bytes > 0:
        planned_outputs.append(f"runtime pack budget capped at {round(max_pack_bytes / (1024**2), 2)} MB")
    if not materialize_pack:
        planned_outputs.append("future derived weight pack placeholder")
    manifest = OptimizedArtifactManifest(
        model_id=model_id,
        artifact_id=artifact_path.stem,
        strategy=strategy,
        profile_id=profile_id,
        created_at=datetime.now(UTC).isoformat(),
        artifact_path=artifact_path,
        source_tensor_count=catalog.tensor_count,
        source_shard_count=catalog.shard_count,
        source_layer_count=catalog.layer_count,
        tensor_pack_path=tensor_pack_path if packed_tensor_names else None,
        packed_tensor_count=len(packed_tensor_names),
        packed_nbytes=packed_nbytes,
        packed_tensor_names=packed_tensor_names,
        planned_outputs=planned_outputs,
        blockers=list(dict.fromkeys(blockers)),
        ready=catalog.ready and not blockers and (not materialize_pack or bool(packed_tensor_names)),
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return manifest


def latest_optimized_artifact_manifest(model_id: str) -> OptimizedArtifactManifest | None:
    """Return the newest optimized artifact manifest for one model, if any."""
    manifests = list_optimized_artifact_manifests(model_id)
    return manifests[0] if manifests else None


def list_optimized_artifact_manifests(model_id: str) -> list[OptimizedArtifactManifest]:
    """Return optimized artifact manifests newest-first."""
    root = artifacts_root(model_id)
    if not root.exists():
        return []
    loaded: list[OptimizedArtifactManifest] = []
    manifests = sorted(root.glob("*.artifact.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in manifests:
        try:
            loaded.append(OptimizedArtifactManifest.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return loaded


def select_runtime_artifact_manifest(
    model_id: str,
    *,
    free_memory_bytes: int | None = None,
    runtime_preset: str = "standard",
) -> OptimizedArtifactManifest | None:
    """Select the largest ready runtime pack that fits the current safety budget."""
    ready_manifests = [
        manifest
        for manifest in list_optimized_artifact_manifests(model_id)
        if manifest.ready
        and manifest.tensor_pack_path is not None
        and manifest.tensor_pack_path.exists()
        and manifest.packed_tensor_names
    ]
    if not ready_manifests:
        return None
    normalized_preset = str(runtime_preset or "standard").strip().lower()
    if normalized_preset in {"boost", "boosted", "high-ram", "high_ram"}:
        budget_bytes = BOOSTED_RUNTIME_PACK_BUDGET_BYTES
    else:
        budget_bytes = STANDARD_RUNTIME_PACK_BUDGET_BYTES
    fitting = [manifest for manifest in ready_manifests if manifest.packed_nbytes <= budget_bytes]
    candidates = fitting or ready_manifests
    return max(candidates, key=lambda manifest: (manifest.packed_nbytes, manifest.created_at))
