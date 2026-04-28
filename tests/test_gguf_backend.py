import json
from pathlib import Path

from pcketlm.core.runtime.gguf_backend import (
    _clean_llama_cli_output,
    build_gguf_backend_status,
    build_gguf_server_status,
    find_gguf_model_files,
    run_gguf_prompt,
    start_gguf_server,
    stop_gguf_server,
)


def test_find_gguf_model_files_finds_artifact_files(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "qwen-test"
    gguf_path = tmp_path / "models" / model_id / "artifacts" / "queen-q4.gguf"
    gguf_path.parent.mkdir(parents=True)
    gguf_path.write_bytes(b"gguf")

    files = find_gguf_model_files(model_id)

    assert len(files) == 1
    assert files[0].path == gguf_path
    assert files[0].source == "artifact"


def test_build_gguf_backend_status_reports_missing_package_and_model(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime import gguf_backend

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    monkeypatch.setattr(gguf_backend, "_llama_cpp_available", lambda: False)
    monkeypatch.setattr(gguf_backend, "_sidecar_llama_cpp_available", lambda *_args: False)
    monkeypatch.setattr(gguf_backend, "llama_cli_path", lambda: tmp_path / "missing" / "llama-cli.exe")
    monkeypatch.setattr(gguf_backend, "llama_server_path", lambda: tmp_path / "missing" / "llama-server.exe")
    monkeypatch.setattr(gguf_backend, "_llama_server_is_ready", lambda: False)

    status = build_gguf_backend_status("qwen-test")

    assert status.ready is False
    assert status.package_available is False
    assert status.main_package_available is False
    assert status.sidecar_package_available is False
    assert status.llama_cli_available is False
    assert len(status.blockers) == 2


def test_build_gguf_server_status_reports_running_process(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core.runtime import gguf_backend

    monkeypatch.setattr(gguf_backend, "_llama_server_pid", lambda: 1234)
    monkeypatch.setattr(gguf_backend, "_llama_server_is_ready", lambda: True)
    monkeypatch.setattr(gguf_backend, "_process_working_set_bytes", lambda pid: 8 * 1024**3)
    monkeypatch.setattr(gguf_backend, "_read_llama_server_state", lambda: {"model_id": "qwen-test", "model_path": str(tmp_path / "queen.gguf")})

    status = build_gguf_server_status("qwen-test")

    assert status.running is True
    assert status.ready is True
    assert status.pid == 1234
    assert status.working_set_bytes == 8 * 1024**3
    assert status.model_id == "qwen-test"


def test_start_gguf_server_reports_missing_artifact(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)

    status = start_gguf_server("qwen-test")

    assert status.ready is False
    assert "No GGUF model file" in status.blockers[0]


def test_stop_gguf_server_clears_state_when_not_running(monkeypatch) -> None:
    from pcketlm.core.runtime import gguf_backend

    called = {"cleared": False}

    monkeypatch.setattr(gguf_backend, "_llama_server_pid", lambda: None)
    monkeypatch.setattr(gguf_backend, "_clear_llama_server_state", lambda: called.update(cleared=True))
    monkeypatch.setattr(gguf_backend, "build_gguf_server_status", lambda model_id: gguf_backend.GGUFServerStatus(running=False, ready=False, model_id=model_id))

    status = stop_gguf_server("qwen-test")

    assert called["cleared"] is True
    assert status.running is False


def test_run_gguf_prompt_uses_injected_llama_factory(tmp_path: Path) -> None:
    gguf_path = tmp_path / "queen.gguf"
    gguf_path.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def __call__(self, prompt: str, **kwargs):
            assert prompt == "hello"
            assert kwargs["max_tokens"] == 3
            assert kwargs["stop"] == ["<|im_end|>"]
            return {"choices": [{"text": " OK"}]}

    result = run_gguf_prompt(
        "qwen-test",
        "hello",
        model_path=gguf_path,
        max_tokens=3,
        stop_strings=["<|im_end|>"],
        llama_factory=FakeLlama,
        prefer_server=False,
    )

    assert result.ready is True
    assert result.generated_text == "OK"
    assert result.model_path == gguf_path


def test_run_gguf_prompt_strips_stop_strings_from_python_backend(tmp_path: Path) -> None:
    gguf_path = tmp_path / "queen.gguf"
    gguf_path.write_bytes(b"gguf")

    class FakeLlama:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, prompt: str, **kwargs):
            assert kwargs["stop"] == ["<|im_end|>"]
            return {"choices": [{"text": " OK<|im_end|>"}]}

    result = run_gguf_prompt(
        "qwen-test",
        "hello",
        model_path=gguf_path,
        stop_strings=["<|im_end|>"],
        llama_factory=FakeLlama,
        prefer_server=False,
    )

    assert result.ready is True
    assert result.generated_text == "OK"


def test_run_gguf_prompt_uses_sidecar_when_main_package_is_missing(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core.runtime import gguf_backend

    gguf_path = tmp_path / "queen.gguf"
    gguf_path.write_bytes(b"gguf")
    sidecar_python = tmp_path / "python.exe"
    sidecar_python.write_bytes(b"exe")

    class FakeCompleted:
        returncode = 0
        stdout = '{"ready": true, "generated_text": " OK", "elapsed_seconds": 1.25}'
        stderr = ""

    monkeypatch.setattr(gguf_backend, "_llama_cpp_available", lambda: False)
    monkeypatch.setattr(gguf_backend, "_sidecar_llama_cpp_available", lambda *_args: True)
    monkeypatch.setattr(gguf_backend, "gguf_sidecar_python_path", lambda: sidecar_python)
    monkeypatch.setattr(gguf_backend, "llama_cli_path", lambda: tmp_path / "missing" / "llama-cli.exe")
    monkeypatch.setattr(gguf_backend.subprocess, "run", lambda *_args, **_kwargs: FakeCompleted())

    result = run_gguf_prompt("qwen-test", "hello", model_path=gguf_path, prefer_server=False)

    assert result.ready is True
    assert result.generated_text == "OK"
    assert result.elapsed_seconds == 1.25


def test_run_gguf_prompt_uses_standalone_cli_when_python_package_is_missing(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core.runtime import gguf_backend

    gguf_path = tmp_path / "queen.gguf"
    gguf_path.write_bytes(b"gguf")
    exe_path = tmp_path / "llama-cli.exe"
    exe_path.write_bytes(b"exe")

    class FakeCompleted:
        returncode = 0
        stdout = " OK\n"
        stderr = ""

    monkeypatch.setattr(gguf_backend, "_llama_cpp_available", lambda: False)
    monkeypatch.setattr(gguf_backend, "_sidecar_llama_cpp_available", lambda *_args: False)
    monkeypatch.setattr(gguf_backend, "llama_cli_path", lambda: exe_path)
    monkeypatch.setattr(gguf_backend.subprocess, "run", lambda *_args, **_kwargs: FakeCompleted())

    result = run_gguf_prompt("qwen-test", "hello", model_path=gguf_path, prefer_server=False)

    assert result.ready is True
    assert result.generated_text == "OK"


def test_run_gguf_prompt_prefers_merged_file_over_split_files(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core import storage
    from pcketlm.core.runtime import gguf_backend

    monkeypatch.setattr(storage.paths, "project_root", lambda: tmp_path)
    model_id = "qwen-test"
    artifact_dir = tmp_path / "models" / model_id / "artifacts"
    split_path = artifact_dir / "queen-q4-00001-of-00002.gguf"
    merged_path = artifact_dir / "queen-q4.gguf"
    split_path.parent.mkdir(parents=True)
    split_path.write_bytes(b"split")
    merged_path.write_bytes(b"merged-file")
    exe_path = tmp_path / "llama-cli.exe"
    exe_path.write_bytes(b"exe")
    captured = {}

    class FakeCompleted:
        returncode = 0
        stdout = " OK\n"
        stderr = ""

    def fake_run(args, **_kwargs):
        captured["args"] = args
        return FakeCompleted()

    monkeypatch.setattr(gguf_backend, "_llama_cpp_available", lambda: False)
    monkeypatch.setattr(gguf_backend, "_sidecar_llama_cpp_available", lambda *_args: False)
    monkeypatch.setattr(gguf_backend, "llama_cli_path", lambda: exe_path)
    monkeypatch.setattr(gguf_backend.subprocess, "run", fake_run)

    result = run_gguf_prompt(model_id, "hello", prefer_server=False)

    assert result.ready is True
    assert str(merged_path) in captured["args"]


def test_clean_llama_cli_output_removes_banner_prompt_and_timings() -> None:
    output = """
Loading model...
build      : b8963

> Reply with OK only.

OK

[ Prompt: 28.6 t/s | Generation: 1000000.0 t/s ]

Exiting...
"""

    assert _clean_llama_cli_output(output, "Reply with OK only.") == "OK"


def test_run_gguf_prompt_uses_server_when_available(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core.runtime import gguf_backend

    gguf_path = tmp_path / "queen.gguf"
    gguf_path.write_bytes(b"gguf")
    server_path = tmp_path / "llama-server.exe"
    server_path.write_bytes(b"exe")

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"content":" OK","timings":{"predicted_n":1}}'

    monkeypatch.setattr(gguf_backend, "llama_server_path", lambda: server_path)
    monkeypatch.setattr(gguf_backend, "_llama_server_is_ready", lambda: True)
    monkeypatch.setattr(gguf_backend.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())

    result = run_gguf_prompt("qwen-test", "hello", model_path=gguf_path)

    assert result.ready is True
    assert result.backend == "llama-cpp-gguf-server"
    assert result.generated_text == "OK"


def test_run_gguf_prompt_sends_server_stop_strings(tmp_path: Path, monkeypatch) -> None:
    from pcketlm.core.runtime import gguf_backend

    gguf_path = tmp_path / "queen.gguf"
    gguf_path.write_bytes(b"gguf")
    server_path = tmp_path / "llama-server.exe"
    server_path.write_bytes(b"exe")
    captured = {}

    class FakeResponse:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return b'{"content":" OK<|im_end|>","timings":{"predicted_n":1}}'

    def fake_urlopen(request, **_kwargs):
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr(gguf_backend, "llama_server_path", lambda: server_path)
    monkeypatch.setattr(gguf_backend, "_llama_server_is_ready", lambda: True)
    monkeypatch.setattr(gguf_backend.urllib.request, "urlopen", fake_urlopen)

    result = run_gguf_prompt("qwen-test", "hello", model_path=gguf_path, stop_strings=["<|im_end|>", "<|im_start|>"])

    assert result.ready is True
    assert captured["payload"]["stop"] == ["<|im_end|>", "<|im_start|>"]
    assert result.generated_text == "OK"
