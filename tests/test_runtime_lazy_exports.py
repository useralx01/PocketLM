"""GGUF control must not initialize unrelated tensor engines."""
import json
import subprocess
import sys


def test_gguf_import_does_not_load_torch():
    result=subprocess.run([sys.executable,'-c',
        'import json,sys; from pcketlm.core.runtime import gguf_backend; '
        'print(json.dumps({"torch": "torch" in sys.modules, "callable": callable(gguf_backend.run_gguf_prompt)}))'],
        capture_output=True,text=True,timeout=15,check=True)
    assert json.loads(result.stdout)=={'torch':False,'callable':True}


def test_public_exports_preserved_and_cached():
    import pcketlm.core.runtime as runtime
    from pcketlm.core.runtime.gguf_backend import GGUFPromptResult
    assert set(runtime.__all__).issubset(runtime._EXPORTS)
    assert runtime.GGUFPromptResult is GGUFPromptResult
    assert runtime.__dict__['GGUFPromptResult'] is GGUFPromptResult
    assert 'build_tensor_catalog' in dir(runtime)
    for name in runtime._EXPORTS:
        assert getattr(runtime,name) is not None
