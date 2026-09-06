import json
from pathlib import Path
import pytest
from pcketlm.core.runtime import gguf_backend as g

@pytest.mark.parametrize('payload,ready', [
    ({'choices':[{'message':{'content':'answer'},'finish_reason':'stop'}]},True),
    ({'choices':[{'message':{'content':'partial'},'finish_reason':'length'}]},False),
    ({'choices':[{'message':{'reasoning_content':'still thinking'},'finish_reason':'stop'}]},False),
    ({'error':{'message':'failure'}},False),
    ({'choices':[]},False),
])
def test_server_does_not_report_empty_or_truncated_success(monkeypatch,payload,ready):
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(payload).encode()
    monkeypatch.setattr(g,'_start_llama_server',lambda *a,**k:True)
    monkeypatch.setattr(g.urllib.request,'urlopen',lambda *a,**k:Response())
    result=g._run_gguf_prompt_server('test','',g.GGUFModelFile(Path('test.gguf'),'original',1),max_tokens=10,n_ctx=2048,n_threads=2,stop_strings=None,chat_messages=[{'role':'user','content':'test'}])
    assert result.ready is ready
    assert bool(result.blockers) is not ready

def test_launcher_lock_serializes_threads_and_releases_after_exception(tmp_path,monkeypatch):
    import threading,time
    monkeypatch.setattr(g,'state_root',lambda:tmp_path)
    active=0;maximum=0;errors=[]
    def worker():
        nonlocal active,maximum
        try:
            with g._server_start_lock():
                active+=1;maximum=max(maximum,active);time.sleep(.03);active-=1
        except Exception as exc:errors.append(exc)
    threads=[threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:thread.start()
    for thread in threads:thread.join(3)
    assert not errors and maximum==1
    assert not any(thread.is_alive() for thread in threads)
    with pytest.raises(ValueError):
        with g._server_start_lock():raise ValueError('fixture')
    with g._server_start_lock():pass

def test_loading_existing_child_never_starts_second_model(monkeypatch):
    monkeypatch.setattr(g,'_llama_server_is_ready',lambda:False)
    monkeypatch.setattr(g,'_llama_server_pid',lambda:1234)
    monkeypatch.setattr(g.subprocess,'Popen',lambda *a,**k:pytest.fail('duplicate launch'))
    assert g._start_llama_server_unlocked(g.GGUFModelFile(Path('test.gguf'),'original',1),n_ctx=4096,n_threads=6) is False

def test_qwen_coding_sampling_and_memory_profile(monkeypatch):
    assert g._gguf_sampling_settings('qwen3.6-35b-a3b')['temperature']==.6
    assert g._gguf_sampling_settings('qwen3.6-35b-a3b')['min_p']==0
    assert g._gguf_sampling_settings('different-model')=={'temperature':0}
    monkeypatch.delenv('POCKETLM_GGUF_REPACK',raising=False)
    assert g._gguf_memory_flags('qwen3.6-35b-a3b')==['--no-repack']
    assert g._gguf_memory_flags('different-model')==[]
    monkeypatch.setenv('POCKETLM_GGUF_REPACK','1')
    assert g._gguf_memory_flags('qwen3.6-35b-a3b')==[]

def test_qwen_chat_request_preserves_thinking_and_uses_profile(monkeypatch):
    captured={}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return b'{"choices":[{"message":{"content":"35"},"finish_reason":"stop"}]}'
    def request(req,**kwargs):captured.update(json.loads(req.data));return Response()
    monkeypatch.setattr(g,'_start_llama_server',lambda *a,**k:True)
    monkeypatch.setattr(g.urllib.request,'urlopen',request)
    result=g._run_gguf_prompt_server('qwen3.6-35b-a3b','',g.GGUFModelFile(Path('test.gguf'),'original',1),max_tokens=4096,n_ctx=65536,n_threads=6,stop_strings=None,chat_messages=[{'role':'user','content':'invoice'}])
    assert result.ready
    assert captured['max_tokens']==4096
    assert captured['temperature']==.6 and captured['min_p']==0
    assert captured['chat_template_kwargs']=={'enable_thinking':True}
    assert captured['reasoning_budget_tokens']==512
    assert result.timings['reasoning_budget_tokens']==512


def test_reasoning_budget_reserves_final_tokens_and_never_disables_thinking(monkeypatch):
    monkeypatch.delenv('POCKETLM_THINKING_BUDGET',raising=False)
    assert g._gguf_reasoning_settings('qwen3.6-35b-a3b',512)['reasoning_budget_tokens']==128
    assert g._gguf_reasoning_settings('qwen3.6-35b-a3b',4096)['reasoning_budget_tokens']==512
    assert 'final answer' in g._gguf_reasoning_settings('qwen3.6-35b-a3b',512)['reasoning_budget_message']
    assert g._gguf_reasoning_settings('other',512)=={}
    assert g._gguf_reasoning_settings('qwen3.6-35b-a3b',4)=={}
    for invalid in ['0','-2','invalid']:
        monkeypatch.setenv('POCKETLM_THINKING_BUDGET',invalid)
        assert g._gguf_reasoning_settings('qwen3.6-35b-a3b',512)['reasoning_budget_tokens']==128
    monkeypatch.setenv('POCKETLM_THINKING_BUDGET','99999')
    assert g._gguf_reasoning_settings('qwen3.6-35b-a3b',512)['reasoning_budget_tokens']==256
    monkeypatch.setenv('POCKETLM_THINKING_BUDGET','-1')
    assert g._gguf_reasoning_settings('qwen3.6-35b-a3b',512)=={}
