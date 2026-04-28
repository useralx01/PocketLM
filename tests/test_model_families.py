from pcketlm.core.model_families import (
    family_label,
    family_priority_summary,
    family_runtime_status,
    normalize_family_key,
)


def test_normalize_family_key_uses_known_aliases() -> None:
    assert normalize_family_key("qwen2") == "qwen"
    assert normalize_family_key("Kronk") == "kronos"
    assert normalize_family_key("gemma3") == "gemma"


def test_family_labels_and_runtime_status_are_product_facing() -> None:
    assert family_label("qwen2.5") == "Qwen"
    assert family_runtime_status("qwen") == "active"
    assert family_runtime_status("kimi") == "planned"
    assert family_label("new-family") == "New Family"
    assert family_runtime_status("new-family") == "unverified"


def test_family_priority_summary_keeps_owner_priority_order() -> None:
    summary = family_priority_summary()

    assert [item["key"] for item in summary[:4]] == ["qwen", "kimi", "kronos", "gemma"]
    assert summary[0]["runtime_status"] == "active"
