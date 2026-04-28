"""Tensor-aware execution planning built from the persisted tensor catalog."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from pcketlm.core.runtime.tensor_catalog import TensorCatalog, TensorCatalogEntry, build_tensor_catalog, load_tensor_catalog
from pcketlm.core.storage.paths import streaming_model_root


@dataclass(slots=True)
class TensorExecutionUnit:
    """One grouped execution unit made from tensor catalog entries."""

    unit_id: str
    label: str
    phase: str
    layer_index: int | None
    component_group: str
    tensor_count: int
    total_nbytes: int
    shard_names: list[str] = field(default_factory=list)
    tensor_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "unit_id": self.unit_id,
            "label": self.label,
            "phase": self.phase,
            "layer_index": self.layer_index,
            "component_group": self.component_group,
            "tensor_count": self.tensor_count,
            "total_nbytes": self.total_nbytes,
            "shard_names": list(self.shard_names),
            "tensor_names": list(self.tensor_names),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorExecutionUnit":
        return cls(
            unit_id=str(payload.get("unit_id", "")),
            label=str(payload.get("label", "")),
            phase=str(payload.get("phase", "")),
            layer_index=payload.get("layer_index"),
            component_group=str(payload.get("component_group", "other")),
            tensor_count=int(payload.get("tensor_count", 0)),
            total_nbytes=int(payload.get("total_nbytes", 0)),
            shard_names=[str(value) for value in payload.get("shard_names", [])],
            tensor_names=[str(value) for value in payload.get("tensor_names", [])],
        )


@dataclass(slots=True)
class TensorExecutionPlan:
    """Execution-oriented grouping of tensors for the first runtime bridge."""

    model_id: str
    plan_path: Path
    catalog_path: Path
    unit_count: int
    phases: list[str] = field(default_factory=list)
    units: list[TensorExecutionUnit] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "plan_path": str(self.plan_path),
            "catalog_path": str(self.catalog_path),
            "unit_count": self.unit_count,
            "phases": list(self.phases),
            "units": [unit.to_dict() for unit in self.units],
            "blockers": list(self.blockers),
            "ready": self.ready,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "TensorExecutionPlan":
        return cls(
            model_id=str(payload.get("model_id", "")),
            plan_path=Path(payload.get("plan_path", "")),
            catalog_path=Path(payload.get("catalog_path", "")),
            unit_count=int(payload.get("unit_count", 0)),
            phases=[str(value) for value in payload.get("phases", [])],
            units=[TensorExecutionUnit.from_dict(item) for item in payload.get("units", [])],
            blockers=list(payload.get("blockers", [])),
            ready=bool(payload.get("ready", False)),
        )


def tensor_execution_plan_path(model_id: str) -> Path:
    """Return the persisted tensor execution plan path for one model."""
    return streaming_model_root(model_id) / "tensor-execution-plan.json"


def _phase_for_group(layer_index: int | None, component_group: str) -> str:
    if layer_index is None and component_group == "embeddings":
        return "prefill"
    if layer_index is None and component_group in {"final_norm", "lm_head"}:
        return "decode-head"
    if component_group == "layer_norm":
        return "layer-entry"
    if component_group == "attention":
        return "layer-attention"
    if component_group == "mlp":
        return "layer-mlp"
    return "misc"


def _unit_label(layer_index: int | None, component_group: str) -> str:
    if layer_index is None:
        return component_group.replace("_", " ").title()
    return f"Layer {layer_index} {component_group.replace('_', ' ').title()}"


def _group_key(entry: TensorCatalogEntry) -> tuple[int, int, str]:
    if entry.layer_index is None and entry.component_group == "embeddings":
        layer_sort = -1
    elif entry.layer_index is None:
        layer_sort = 10_000
    else:
        layer_sort = entry.layer_index
    phase_order = {
        "prefill": 0,
        "layer-entry": 1,
        "layer-attention": 2,
        "layer-mlp": 3,
        "decode-head": 4,
        "misc": 5,
    }
    phase = _phase_for_group(entry.layer_index, entry.component_group)
    return (layer_sort, phase_order.get(phase, 99), entry.component_group)


def _build_units(catalog: TensorCatalog) -> list[TensorExecutionUnit]:
    grouped: dict[tuple[int | None, str], list[TensorCatalogEntry]] = {}
    for entry in sorted(catalog.tensors, key=_group_key):
        key = (entry.layer_index, entry.component_group)
        grouped.setdefault(key, []).append(entry)

    units: list[TensorExecutionUnit] = []
    def grouped_key(item: tuple[int | None, str]) -> tuple[int, int, str]:
        layer_index, component_group = item
        phase = _phase_for_group(layer_index, component_group)
        phase_order = {
            "prefill": 0,
            "layer-entry": 1,
            "layer-attention": 2,
            "layer-mlp": 3,
            "decode-head": 4,
            "misc": 5,
        }
        if layer_index is None and component_group == "embeddings":
            layer_sort = -1
        elif layer_index is None:
            layer_sort = 10_000
        else:
            layer_sort = layer_index
        return (layer_sort, phase_order.get(phase, 99), component_group)

    for layer_index, component_group in sorted(grouped.keys(), key=grouped_key):
        entries = grouped[(layer_index, component_group)]
        phase = _phase_for_group(layer_index, component_group)
        unit_id = (
            f"{component_group}"
            if layer_index is None
            else f"layer-{layer_index:02d}-{component_group}"
        )
        units.append(
            TensorExecutionUnit(
                unit_id=unit_id,
                label=_unit_label(layer_index, component_group),
                phase=phase,
                layer_index=layer_index,
                component_group=component_group,
                tensor_count=len(entries),
                total_nbytes=sum(entry.data_nbytes for entry in entries),
                shard_names=sorted({entry.shard_name for entry in entries}),
                tensor_names=[entry.tensor_name for entry in entries],
            )
        )
    return units


def build_tensor_execution_plan(model_id: str, model_dir: Path) -> TensorExecutionPlan:
    """Build and persist the first tensor-aware execution plan for one model."""
    plan_path = tensor_execution_plan_path(model_id)
    catalog = load_tensor_catalog(model_id)
    if not catalog.ready:
        catalog = build_tensor_catalog(model_id, model_dir)

    if not catalog.ready:
        return TensorExecutionPlan(
            model_id=model_id,
            plan_path=plan_path,
            catalog_path=catalog.catalog_path,
            unit_count=0,
            blockers=list(catalog.blockers) or ["Tensor catalog is not ready yet."],
            ready=False,
        )

    units = _build_units(catalog)
    phases: list[str] = []
    for unit in units:
        if unit.phase not in phases:
            phases.append(unit.phase)

    plan = TensorExecutionPlan(
        model_id=model_id,
        plan_path=plan_path,
        catalog_path=catalog.catalog_path,
        unit_count=len(units),
        phases=phases,
        units=units,
        blockers=[],
        ready=bool(units),
    )
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(plan.to_dict(), indent=2), encoding="utf-8")
    return plan


def load_tensor_execution_plan(model_id: str) -> TensorExecutionPlan:
    """Load a persisted tensor execution plan for one model."""
    plan_path = tensor_execution_plan_path(model_id)
    if not plan_path.exists():
        return TensorExecutionPlan(
            model_id=model_id,
            plan_path=plan_path,
            catalog_path=Path(),
            unit_count=0,
            blockers=["Tensor execution plan does not exist yet."],
            ready=False,
        )

    mtime_ns = plan_path.stat().st_mtime_ns
    payload = _load_tensor_execution_plan_payload(str(plan_path), mtime_ns)
    payload["plan_path"] = str(plan_path)
    return TensorExecutionPlan.from_dict(payload)


@lru_cache(maxsize=16)
def _load_tensor_execution_plan_payload(plan_path: str, mtime_ns: int) -> dict:
    """Load persisted execution-plan JSON with automatic invalidation when the file changes."""
    del mtime_ns
    return json.loads(Path(plan_path).read_text(encoding="utf-8"))
