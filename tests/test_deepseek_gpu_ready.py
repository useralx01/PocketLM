import importlib.util
from pathlib import Path

from pcketlm.core.runtime.tensor_catalog import build_tensor_catalog
from tools.deepseek_gpu_ready import check_deepseek_gpu_ready, main
from tools.pack_fp8 import pack_model_dir_to_fp8

_RUNTIME_TEST_PATH = Path(__file__).with_name("test_runtime_fp8_source.py")
_RUNTIME_SPEC = importlib.util.spec_from_file_location("_runtime_fp8_source_helpers", _RUNTIME_TEST_PATH)
assert _RUNTIME_SPEC is not None and _RUNTIME_SPEC.loader is not None
_RUNTIME_HELPERS = importlib.util.module_from_spec(_RUNTIME_SPEC)
_RUNTIME_SPEC.loader.exec_module(_RUNTIME_HELPERS)
_write_fp8_runtime_fixture = _RUNTIME_HELPERS._write_fp8_runtime_fixture


def test_deepseek_gpu_ready_reports_cuda_blocker_with_ready_pack(tmp_path, monkeypatch) -> None:
    import tools.pack_fp8 as pack_tool

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    pack_model_dir_to_fp8(
        model_dir,
        model_dir / "artifacts" / "fp8_pack",
        model_id=model_id,
        pack_bytes=1024,
    )

    payload = check_deepseek_gpu_ready(
        model_id=model_id,
        start_layer=0,
        layer_count=1,
        resident_budget_bytes=1024 * 1024,
    )

    assert payload["catalog"]["ready"] is True
    assert payload["pack"]["manifest_exists"] is True
    assert payload["pack"]["pack_file_count"] >= 1
    if not payload["cuda"]["available"]:
        assert payload["ready"] is False
        assert "CUDA is not available on this machine." in payload["blockers"]


def test_deepseek_gpu_ready_can_run_cpu_diagnostic_when_allowed(tmp_path, monkeypatch) -> None:
    import tools.pack_fp8 as pack_tool

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    pack_model_dir_to_fp8(
        model_dir,
        model_dir / "artifacts" / "fp8_pack",
        model_id=model_id,
        pack_bytes=1024,
    )

    payload = check_deepseek_gpu_ready(
        model_id=model_id,
        start_layer=0,
        layer_count=1,
        resident_budget_bytes=1024 * 1024,
        run=True,
        allow_cpu_run=True,
    )

    assert payload["run"]["passed"] is True
    assert payload["run"]["start_layer"] == 0
    assert payload["run"]["layer_count"] == 1
    assert payload["run"]["step_summaries"]
    assert payload["run"]["final_top_token_ids"]


def test_deepseek_gpu_ready_cli_prints_json(tmp_path, monkeypatch, capsys) -> None:
    import tools.pack_fp8 as pack_tool

    monkeypatch.setattr(pack_tool, "state_root", lambda: tmp_path / "state")
    model_id, model_dir = _write_fp8_runtime_fixture(tmp_path, monkeypatch)
    build_tensor_catalog(model_id, model_dir)
    pack_model_dir_to_fp8(
        model_dir,
        model_dir / "artifacts" / "fp8_pack",
        model_id=model_id,
        pack_bytes=1024,
    )

    exit_code = main(
        [
            "--model-id",
            model_id,
            "--layer",
            "0",
            "--layers",
            "1",
            "--budget-gb",
            "1",
            "--json",
        ]
    )
    captured = capsys.readouterr().out

    assert exit_code in {0, 2}
    assert '"catalog"' in captured
    assert '"pack"' in captured
    assert '"next_action"' in captured
