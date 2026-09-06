from types import SimpleNamespace
from pcketlm.app.web import main as web

def test_gguf_chat_does_not_inherit_four_token_native_smoke_budget(monkeypatch):
    monkeypatch.setattr(web,'get_saved_profile',lambda *a:SimpleNamespace(settings={'default_max_new_tokens':4},runtime_mode='GGUF'))
    defaults=web._chat_request_runtime_defaults({'model_id':'qwen-test','prompt':'work','mode':'GGUF'})
    assert defaults[4]==4096

def test_explicit_gguf_output_budget_is_not_silently_clamped_to_64(monkeypatch):
    monkeypatch.setattr(web,'get_saved_profile',lambda *a:None)
    defaults=web._chat_request_runtime_defaults({'model_id':'qwen-test','prompt':'work','mode':'GGUF','max_new_tokens':8192})
    assert defaults[4]==8192
    assert web._chat_request_runtime_defaults({'model_id':'qwen-test','prompt':'work','mode':'GGUF','max_new_tokens':12})[4]==12

def test_native_smoke_bound_is_unchanged(monkeypatch):
    monkeypatch.setattr(web,'get_saved_profile',lambda *a:None)
    assert web._chat_request_runtime_defaults({'model_id':'fixture','prompt':'work','mode':'Direct Quality','max_new_tokens':8192})[4]==16
