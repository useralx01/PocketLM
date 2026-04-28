from pathlib import Path

from pcketlm.core.runtime.bootstrap import (
    RuntimeComponentCheck,
    RuntimeDependencyStatus,
    build_runtime_bootstrap,
    create_runtime_session,
)


def test_build_runtime_bootstrap_blocked_by_source_and_dependencies(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime

    monkeypatch.setattr(
        runtime.bootstrap,
        "probe_runtime_dependencies",
        lambda: RuntimeDependencyStatus(
            transformers=False,
            tokenizers=False,
            safetensors=False,
            torch=False,
        ),
    )
    monkeypatch.setattr(
        runtime.bootstrap,
        "_load_runtime_components",
        lambda model_dir, dependencies: RuntimeComponentCheck(),
    )

    result = build_runtime_bootstrap("qwen-test", tmp_path)

    assert result.can_attempt_load is False
    assert any("torch" in blocker for blocker in result.blockers)
    assert any("not runtime-ready yet" in blocker for blocker in result.blockers)


def test_create_runtime_session_ready(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import runtime

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (tmp_path / "vocab.json").write_text("{}", encoding="utf-8")
    (tmp_path / "merges.txt").write_text("", encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(
        '{"metadata":{"total_size":1},"weight_map":{"a":"model-00001-of-00001.safetensors"}}',
        encoding="utf-8",
    )
    (tmp_path / "model-00001-of-00001.safetensors").write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        runtime.bootstrap,
        "probe_runtime_dependencies",
        lambda: RuntimeDependencyStatus(
            transformers=True,
            tokenizers=True,
            safetensors=True,
            torch=True,
        ),
    )
    monkeypatch.setattr(
        runtime.bootstrap,
        "_load_runtime_components",
        lambda model_dir, dependencies: RuntimeComponentCheck(
            config_loadable=True,
            tokenizer_loadable=True,
        ),
    )

    session = create_runtime_session("qwen-test", tmp_path)

    assert session.status == "ready"
    assert session.can_attempt_load is True
    assert session.source_ready is True
