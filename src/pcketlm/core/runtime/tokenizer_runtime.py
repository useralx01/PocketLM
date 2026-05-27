"""Local tokenizer helpers for prompt-entry runtime work."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
import json
from pathlib import Path

from tokenizers import Tokenizer

from pcketlm.core.storage.paths import original_model_root


@dataclass(slots=True)
class TokenizerLoadResult:
    """Status of loading a local tokenizer from the original model files."""

    model_id: str
    tokenizer_path: Path
    blockers: list[str] = field(default_factory=list)
    ready: bool = False
    tokenizer: Tokenizer | None = None

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "tokenizer_path": str(self.tokenizer_path),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@dataclass(slots=True)
class GenerationSettings:
    """Small local generation defaults loaded from generation_config.json."""

    model_id: str
    do_sample: bool = False
    top_k: int = 5
    top_p: float = 1.0
    temperature: float = 1.0
    repetition_penalty: float = 1.0
    eos_token_ids: list[int] = field(default_factory=list)
    ready: bool = False
    blockers: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "do_sample": self.do_sample,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "eos_token_ids": list(self.eos_token_ids),
            "ready": self.ready,
            "blockers": list(self.blockers),
        }


@dataclass(slots=True)
class PreparedPromptResult:
    """Prepared prompt text plus token ids for the first prompt-session path."""

    model_id: str
    raw_prompt: str
    prepared_prompt: str
    token_ids: list[int] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    ready: bool = False

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id,
            "raw_prompt": self.raw_prompt,
            "prepared_prompt": self.prepared_prompt,
            "token_ids": list(self.token_ids),
            "blockers": list(self.blockers),
            "ready": self.ready,
        }


@lru_cache(maxsize=8)
def _tokenizer_from_path(tokenizer_path: str) -> Tokenizer:
    return Tokenizer.from_file(tokenizer_path)


def load_local_tokenizer(model_id: str) -> TokenizerLoadResult:
    """Load the local tokenizer.json for the given model id."""
    model_root = _model_metadata_root(model_id)
    tokenizer_path = model_root / "tokenizer.json"
    blockers: list[str] = []
    if not tokenizer_path.exists():
        blockers.append(f"Missing tokenizer file at {tokenizer_path}.")
        return TokenizerLoadResult(
            model_id=model_id,
            tokenizer_path=tokenizer_path,
            blockers=blockers,
            ready=False,
        )

    try:
        tokenizer = _tokenizer_from_path(str(tokenizer_path))
    except Exception as exc:  # pragma: no cover - defensive path
        blockers.append(f"Tokenizer load failed: {exc}.")
        return TokenizerLoadResult(
            model_id=model_id,
            tokenizer_path=tokenizer_path,
            blockers=blockers,
            ready=False,
        )

    return TokenizerLoadResult(
        model_id=model_id,
        tokenizer_path=tokenizer_path,
        blockers=[],
        ready=True,
        tokenizer=tokenizer,
    )


@lru_cache(maxsize=64)
def _load_json_if_present_cached(path_string: str, modified_ns: int) -> dict:
    return json.loads(Path(path_string).read_text(encoding="utf-8"))


def _load_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}
    return _load_json_if_present_cached(str(path), path.stat().st_mtime_ns)


def load_generation_settings(model_id: str) -> GenerationSettings:
    """Load basic generation defaults from the local generation config."""
    model_root = _model_metadata_root(model_id)
    generation_path = model_root / "generation_config.json"
    config_path = model_root / "config.json"
    payload = _load_json_if_present(generation_path)
    if not payload:
        payload = _load_json_if_present(config_path)

    raw_eos = payload.get("eos_token_id")
    if raw_eos is None:
        eos_token_ids: list[int] = []
    elif isinstance(raw_eos, list):
        eos_token_ids = [int(value) for value in raw_eos]
    else:
        eos_token_ids = [int(raw_eos)]

    return GenerationSettings(
        model_id=model_id,
        do_sample=bool(payload.get("do_sample", False)),
        top_k=int(payload.get("top_k", 5)),
        top_p=float(payload.get("top_p", 1.0)),
        temperature=float(payload.get("temperature", 1.0)),
        repetition_penalty=float(payload.get("repetition_penalty", 1.0)),
        eos_token_ids=eos_token_ids,
        ready=bool(payload),
        blockers=[],
    )


def prepare_prompt_text(
    model_id: str,
    prompt: str,
    system_prompt: str | None = None,
    apply_chat_format: bool = True,
) -> PreparedPromptResult:
    """Prepare a prompt in a simple instruct/chat style when metadata supports it."""
    if not prompt.strip():
        return PreparedPromptResult(
            model_id=model_id,
            raw_prompt=prompt,
            prepared_prompt=prompt,
            blockers=["Prompt text must not be empty."],
            ready=False,
        )

    tokenizer_config_path = _model_metadata_root(model_id) / "tokenizer_config.json"
    tokenizer_config = _load_json_if_present(tokenizer_config_path)
    prepared_prompt = prompt

    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder", {})
    special_contents = {
        entry.get("content")
        for entry in added_tokens_decoder.values()
        if isinstance(entry, dict) and "content" in entry
    }
    if apply_chat_format and "<|im_start|>" in special_contents and "<|im_end|>" in special_contents:
        system_text = system_prompt or "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        prepared_prompt = (
            f"<|im_start|>system\n{system_text}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

    token_ids, blockers = encode_prompt_text(model_id, prepared_prompt)
    return PreparedPromptResult(
        model_id=model_id,
        raw_prompt=prompt,
        prepared_prompt=prepared_prompt,
        token_ids=token_ids,
        blockers=blockers,
        ready=not blockers,
    )


def encode_prompt_text(model_id: str, prompt: str) -> tuple[list[int], list[str]]:
    """Encode one prompt string into token ids with the local tokenizer."""
    load_result = load_local_tokenizer(model_id)
    if not load_result.ready or load_result.tokenizer is None:
        return [], list(load_result.blockers)

    if not prompt.strip():
        return [], ["Prompt text must not be empty."]

    encoding = load_result.tokenizer.encode(prompt)
    token_ids = [int(token_id) for token_id in encoding.ids]
    if not token_ids:
        return [], ["Prompt text encoded to zero tokens."]
    return token_ids, []


def decode_token_ids_to_text(model_id: str, token_ids: list[int]) -> tuple[str, list[str]]:
    """Decode token ids back into text with the local tokenizer."""
    load_result = load_local_tokenizer(model_id)
    if not load_result.ready or load_result.tokenizer is None:
        return "", list(load_result.blockers)
    if not token_ids:
        return "", []

    try:
        text = load_result.tokenizer.decode([int(token_id) for token_id in token_ids], skip_special_tokens=False)
    except Exception as exc:  # pragma: no cover - defensive path
        return "", [f"Tokenizer decode failed: {exc}."]
    return text, []


def _model_metadata_root(model_id: str) -> Path:
    root = original_model_root(model_id)
    if (root / "tokenizer.json").exists() or (root / "config.json").exists():
        return root
    try:
        from pcketlm.core.runtime.tensor_catalog import load_tensor_catalog

        catalog_root = load_tensor_catalog(model_id).model_dir
        if (catalog_root / "tokenizer.json").exists() or (catalog_root / "config.json").exists():
            return catalog_root
    except Exception:
        pass
    return root
