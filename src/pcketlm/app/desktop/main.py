"""Tkinter test UI for the pcketlm desktop status screen."""

from __future__ import annotations

import json
import os
import threading
import tkinter as tk
from pathlib import Path
from types import SimpleNamespace
from tkinter import messagebox, ttk

from pcketlm.app.desktop.status_screen import (
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_LABEL,
    StatusScreenModel,
    build_status_screen_model,
    list_status_screen_options,
)
from pcketlm.core.benchmark import run_measured_benchmark
from pcketlm.core.profiles import profile_templates_summary
from pcketlm.core.runtime import advance_streaming_runtime, advance_streaming_runtime_safely
from pcketlm.core.runtime import run_fp8_decode_loop, run_prompt_decode_loop
from pcketlm.app.chat_shell.runtime_fp8_cli import (
    _decode_with_catalog_tokenizer,
    _encode_with_catalog_tokenizer,
    _prepare_chat_with_catalog_tokenizer,
)


PALETTE = {
    "bg": "#f4efe6",
    "panel": "#fffaf1",
    "header": "#12343b",
    "text": "#1f2933",
    "muted": "#66788a",
    "line": "#d8c9ae",
    "ready": "#2f855a",
    "blocked": "#b8322f",
    "downloading": "#b7791f",
    "neutral": "#315b7c",
}


def _pill_color(value: str) -> str:
    lowered = value.lower()
    if "ready" in lowered:
        return PALETTE["ready"]
    if "download" in lowered:
        return PALETTE["downloading"]
    if "block" in lowered or "incomplete" in lowered or "missing" in lowered:
        return PALETTE["blocked"]
    return PALETTE["neutral"]


def _parse_stop_token_ids(raw_value: str) -> tuple[list[int], list[str]]:
    values = [chunk.strip() for chunk in raw_value.split(",") if chunk.strip()]
    parsed: list[int] = []
    blockers: list[str] = []
    for value in values:
        try:
            parsed.append(int(value))
        except ValueError:
            blockers.append(f"Stop token id '{value}' is not a valid integer.")
    return parsed, blockers


def _parse_stop_strings(raw_value: str) -> list[str]:
    return [chunk.strip() for chunk in raw_value.split("|") if chunk.strip()]


def _summarize_prompt_result(result) -> str:
    stop_reason = result.stop_reason or "none"
    stop_tokens = ", ".join(str(value) for value in result.stop_token_ids) if result.stop_token_ids else "model defaults"
    stop_strings = " | ".join(result.stop_strings) if getattr(result, "stop_strings", None) else "none"
    blockers = "\n".join(f"- {blocker}" for blocker in result.blockers) if result.blockers else "- none"
    return (
        f"Strategy: {result.strategy}\n"
        f"Ready: {'yes' if result.ready else 'no'}\n"
        f"Min new tokens: {result.min_new_tokens}\n"
        f"Steps: {result.steps_completed} / {result.max_new_tokens}\n"
        f"Stop reason: {stop_reason}\n"
        f"Stop tokens: {stop_tokens}\n"
        f"Stop strings: {stop_strings}\n"
        f"Prompt token count: {len(result.prompt_token_ids)}\n"
        f"Generated token count: {len(result.generated_token_ids)}\n"
        f"Cache lengths: {json.dumps(result.cache_sequence_lengths, indent=2) if result.cache_sequence_lengths else '{}'}\n"
        f"Blockers:\n{blockers}"
    )


def _planned_action_message(action: str) -> str:
    """Return the plain-English planned-state message for an unfinished model-home action."""
    messages = {
        "Personalize": (
            "Personalization profiles are planned next. They will let this model target memory, speed, "
            "quality, and agent-style capability tradeoffs while keeping the original model untouched."
        ),
        "Compare": (
            "Comparison is planned next. It will show default-vs-profile behavior and performance so "
            "changes are proven instead of guessed."
        ),
        "Benchmark": (
            "Benchmarking is planned next. It will measure quality, speed, memory, and stability with "
            "repeatable checks."
        ),
    }
    return messages.get(action, f"{action} is planned but not wired yet.")


def _chat_runtime_hint(max_new_tokens: int = 2, mode_label: str = "Quality (full stack)") -> str:
    """Return a plain-English expectation for the current slow local runtime."""
    layer_budget = _chat_layer_budget(mode_label)
    if max_new_tokens > 4:
        estimate = "several minutes"
    elif layer_budget == 8:
        estimate = "under 30 seconds"
    elif layer_budget == 32:
        estimate = "about 40 seconds"
    elif max_new_tokens <= 2:
        estimate = "about 1 minute"
    else:
        estimate = "about 2 minutes"
    return f"Alpha runtime: keep max new tokens low for now. {max_new_tokens} token(s) can take {estimate}."


def _chat_layer_budget(mode_label: str) -> int | None:
    """Map the desktop speed/quality mode to a runtime layer budget."""
    if mode_label.startswith("Fast"):
        return 8
    if mode_label.startswith("Balanced"):
        return 32
    return None


def _chat_mode_hint(mode_label: str) -> str:
    """Return the plain-English tradeoff for one chat mode."""
    layer_budget = _chat_layer_budget(mode_label)
    if layer_budget == 8:
        return "Fast mode is only for quick smoke tests. It is much faster, but output quality can be rough."
    if layer_budget == 32:
        return "Balanced mode is a speed preview. It is faster, but it can drift or answer oddly."
    return "Quality mode uses the full current stack and gives the best current output."


def _is_deepseek_fp8_model(model_id: str) -> bool:
    normalized = str(model_id or "").lower().replace("_", "-")
    return "deepseek" in normalized and "v3" in normalized


def _run_desktop_chat_generation(
    model_id: str,
    *,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    top_p: float,
    layer_count: int | None,
    repetition_penalty: float,
    system_prompt: str | None,
    apply_chat_format: bool,
    stop_token_ids: list[int] | None,
    stop_strings: list[str] | None,
):
    if not _is_deepseek_fp8_model(model_id):
        return run_prompt_decode_loop(
            model_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            top_p=top_p,
            layer_count=layer_count,
            repetition_penalty=repetition_penalty,
            system_prompt=system_prompt,
            apply_chat_format=apply_chat_format,
            stop_token_ids=stop_token_ids or None,
            stop_strings=stop_strings or None,
        )

    if apply_chat_format:
        prepared_prompt, token_ids, blockers = _prepare_chat_with_catalog_tokenizer(
            model_id,
            prompt,
            system_prompt=system_prompt or "",
        )
    else:
        prepared_prompt = prompt
        token_ids, blockers = _encode_with_catalog_tokenizer(model_id, prompt)
    result = None
    if not blockers:
        result = run_fp8_decode_loop(
            model_id,
            token_ids[-16:],
            layer_count=layer_count,
            max_new_tokens=max_new_tokens,
        )
        blockers = list(result.blockers)
    generated_ids = [] if result is None else list(result.generated_token_ids)
    generated_text, decode_blockers = _decode_with_catalog_tokenizer(model_id, generated_ids)
    blockers.extend(decode_blockers)
    ready = bool(result is not None and result.ready and not blockers)
    return SimpleNamespace(
        ready=ready,
        strategy="deepseek-fp8-pack",
        min_new_tokens=min_new_tokens,
        steps_completed=0 if result is None else int(result.positions_completed),
        max_new_tokens=max_new_tokens,
        stop_reason="deepseek-fp8-complete" if ready else "deepseek-fp8-blocked",
        stop_token_ids=stop_token_ids or [],
        stop_strings=stop_strings or [],
        prompt_token_ids=token_ids,
        generated_token_ids=generated_ids,
        cache_sequence_lengths={}
        if result is None
        else {str(key): int(value[0].shape[1]) for key, value in result.next_kv_caches.items()},
        blockers=blockers,
        generated_text=generated_text,
        full_text=f"{prepared_prompt}\n{generated_text}",
    )


class PcketLmStatusApp:
    """Simple desktop test UI for the pcketlm status screen."""

    def __init__(self, root: tk.Tk, model_id: str = DEFAULT_MODEL_ID, model_label: str = DEFAULT_MODEL_LABEL) -> None:
        self.root = root
        self.model_id = model_id
        self.model_label = model_label
        self.current_model: StatusScreenModel | None = None
        self.last_prompt_result = None
        self.model_options = []
        self.model_var = tk.StringVar()
        self.system_prompt_var = tk.StringVar(value="Be helpful and brief.")
        self.raw_prompt_var = tk.BooleanVar(value=False)
        self.max_new_tokens_var = tk.StringVar(value="2")
        self.min_new_tokens_var = tk.StringVar(value="1")
        self.top_p_var = tk.StringVar(value="1.0")
        self.repetition_penalty_var = tk.StringVar(value="1.1")
        self.chat_mode_var = tk.StringVar(value="Quality (full stack)")
        self.stop_token_ids_var = tk.StringVar()
        self.stop_strings_var = tk.StringVar()
        self.chat_history: list[tuple[str, str]] = []
        self.prompt_run_active = False
        self._content_window_id: int | None = None
        self._build_window()
        self.refresh()

    def _build_window(self) -> None:
        self.root.title("pcketlm")
        self.root.configure(bg=PALETTE["bg"])
        self.root.geometry("980x760")
        self.root.minsize(900, 700)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Pcket.Horizontal.TProgressbar", troughcolor="#eadfcd", background=PALETTE["neutral"])

        shell = tk.Frame(self.root, bg=PALETTE["bg"])
        shell.pack(fill="both", expand=True, padx=18, pady=18)

        header = tk.Frame(shell, bg=PALETTE["header"], padx=20, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="pcketlm", font=("Segoe UI Semibold", 24), fg="#fff7ea", bg=PALETTE["header"]).pack(anchor="w")
        tk.Label(
            header,
            text="Model Status",
            font=("Segoe UI", 12),
            fg="#d8efe6",
            bg=PALETTE["header"],
        ).pack(anchor="w", pady=(4, 0))

        selector_row = tk.Frame(shell, bg=PALETTE["bg"])
        selector_row.pack(fill="x", pady=(14, 0))
        tk.Label(selector_row, text="Model", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["bg"]).pack(side="left")
        self.model_picker = ttk.Combobox(selector_row, textvariable=self.model_var, state="readonly", width=70)
        self.model_picker.pack(side="left", padx=(12, 0))
        self.model_picker.bind("<<ComboboxSelected>>", self.on_model_selected)
        self._button(selector_row, "Reload Models", self.reload_models).pack(side="left", padx=(10, 0))

        content_shell = tk.Frame(shell, bg=PALETTE["bg"])
        content_shell.pack(fill="both", expand=True, pady=(16, 0))

        self.canvas = tk.Canvas(
            content_shell,
            bg=PALETTE["bg"],
            highlightthickness=0,
            bd=0,
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(content_shell, orient="vertical", command=self.canvas.yview)
        scrollbar.pack(side="right", fill="y")
        self.canvas.configure(yscrollcommand=scrollbar.set)

        self.body = tk.Frame(self.canvas, bg=PALETTE["bg"])
        self._content_window_id = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.body.bind("<Configure>", self._sync_scroll_region)
        self.canvas.bind("<Configure>", self._resize_canvas_content)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        self.identity_card = self._card(self.body)
        self.identity_card.pack(fill="x")
        self.identity_title = tk.Label(self.identity_card, text="", font=("Segoe UI Semibold", 18), fg=PALETTE["text"], bg=PALETTE["panel"])
        self.identity_title.pack(anchor="w")
        self.identity_meta = tk.Label(self.identity_card, text="", font=("Segoe UI", 11), fg=PALETTE["muted"], bg=PALETTE["panel"], justify="left")
        self.identity_meta.pack(anchor="w", pady=(8, 0))

        self.home_card = self._card(self.body)
        self.home_card.pack(fill="x", pady=(14, 0))
        self._section_title(self.home_card, "Model Home")
        home_actions = tk.Frame(self.home_card, bg=PALETTE["panel"])
        home_actions.pack(fill="x", pady=(12, 0))
        self._button(home_actions, "Chat", self.open_chat_home).pack(side="left")
        self._button(home_actions, "Personalize", self.show_profile_templates).pack(side="left", padx=(10, 0))
        self._button(home_actions, "Compare", lambda: self.show_planned_action("Compare")).pack(side="left", padx=(10, 0))
        self._button(home_actions, "Inspect", self.show_details).pack(side="left", padx=(10, 0))
        self._button(home_actions, "Benchmark", self.show_benchmark_readiness).pack(side="left", padx=(10, 0))
        self.model_actions_label = self._body_label(self.home_card)

        self.acq_card = self._card(self.body)
        self.acq_card.pack(fill="x", pady=(14, 0))
        self.acq_status = self._section_title(self.acq_card, "Acquisition")
        self.acq_progress = ttk.Progressbar(self.acq_card, style="Pcket.Horizontal.TProgressbar", maximum=100)
        self.acq_progress.pack(fill="x", pady=(12, 6))
        self.acq_progress_label = self._value_label(self.acq_card)
        self.acq_download = self._value_label(self.acq_card)
        self.acq_shards = self._value_label(self.acq_card)
        self.acq_summary = self._body_label(self.acq_card)

        self.runtime_card = self._card(self.body)
        self.runtime_card.pack(fill="x", pady=(14, 0))
        self.runtime_status_label = self._section_title(self.runtime_card, "Runtime Readiness")

        grid = tk.Frame(self.runtime_card, bg=PALETTE["panel"])
        grid.pack(fill="x", pady=(12, 0))
        self.runtime_rows: dict[str, tk.Label] = {}
        for row, label in enumerate(("Status", "Config", "Tokenizer", "Weights", "Runtime libs", "Load attempt")):
            tk.Label(grid, text=label, font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).grid(row=row, column=0, sticky="w", padx=(0, 12), pady=3)
            value = tk.Label(grid, text="", font=("Segoe UI", 10), fg=PALETTE["text"], bg=PALETTE["panel"])
            value.grid(row=row, column=1, sticky="w", pady=3)
            self.runtime_rows[label] = value
        self.runtime_summary = self._body_label(self.runtime_card)

        self.streaming_card = self._card(self.body)
        self.streaming_card.pack(fill="x", pady=(14, 0))
        self.streaming_status_label = self._section_title(self.streaming_card, "Streaming Control")
        self.streaming_action = self._value_label(self.streaming_card)
        self.streaming_rotation = self._value_label(self.streaming_card)
        self.streaming_cache = self._value_label(self.streaming_card)
        self.streaming_windows = self._body_label(self.streaming_card)
        self.streaming_summary = self._body_label(self.streaming_card)

        self.blockers_card = self._card(self.body)
        self.blockers_card.pack(fill="x", pady=(14, 0))
        self._section_title(self.blockers_card, "Current Blockers")
        self.blockers_text = tk.Text(
            self.blockers_card,
            height=5,
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
            relief="flat",
            wrap="word",
            font=("Segoe UI", 10),
        )
        self.blockers_text.pack(fill="x", pady=(12, 0))
        self.blockers_text.configure(state="disabled")

        self.next_card = self._card(self.body)
        self.next_card.pack(fill="x", pady=(14, 0))
        self._section_title(self.next_card, "Recommended Action")
        self.next_step_label = self._body_label(self.next_card)

        self.prompt_card = self._card(self.body)
        self.prompt_card.pack(fill="x", pady=(14, 0))
        self._section_title(self.prompt_card, "Chat")

        prompt_controls = tk.Frame(self.prompt_card, bg=PALETTE["panel"])
        prompt_controls.pack(fill="x", pady=(12, 0))
        prompt_controls.grid_columnconfigure(1, weight=1)

        tk.Label(prompt_controls, text="Prompt", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).grid(row=0, column=0, sticky="nw", padx=(0, 12), pady=(0, 8))
        self.prompt_text = tk.Text(prompt_controls, height=4, wrap="word", font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.prompt_text.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        self.prompt_text.insert("1.0", "hello world")

        tk.Label(prompt_controls, text="System prompt", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).grid(row=1, column=0, sticky="w", padx=(0, 12), pady=4)
        self.system_prompt_entry = tk.Entry(prompt_controls, textvariable=self.system_prompt_var, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.system_prompt_entry.grid(row=1, column=1, sticky="ew", pady=4)

        options_row = tk.Frame(prompt_controls, bg=PALETTE["panel"])
        options_row.grid(row=2, column=1, sticky="ew", pady=(6, 0))
        tk.Checkbutton(
            options_row,
            text="Raw prompt",
            variable=self.raw_prompt_var,
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
            activebackground=PALETTE["panel"],
            activeforeground=PALETTE["text"],
            selectcolor=PALETTE["panel"],
            font=("Segoe UI", 10),
        ).pack(side="left")

        token_row = tk.Frame(self.prompt_card, bg=PALETTE["panel"])
        token_row.pack(fill="x", pady=(12, 0))

        tk.Label(token_row, text="Max new tokens", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.max_new_tokens_entry = tk.Entry(token_row, textvariable=self.max_new_tokens_var, width=8, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.max_new_tokens_entry.pack(side="left", padx=(10, 18))

        tk.Label(token_row, text="Min new tokens", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.min_new_tokens_entry = tk.Entry(token_row, textvariable=self.min_new_tokens_var, width=8, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.min_new_tokens_entry.pack(side="left", padx=(10, 18))

        tk.Label(token_row, text="Top-p", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.top_p_entry = tk.Entry(token_row, textvariable=self.top_p_var, width=8, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.top_p_entry.pack(side="left", padx=(10, 18))

        tk.Label(token_row, text="Repetition penalty", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.repetition_penalty_entry = tk.Entry(token_row, textvariable=self.repetition_penalty_var, width=8, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.repetition_penalty_entry.pack(side="left", padx=(10, 18))

        tk.Label(token_row, text="Stop token ids", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.stop_token_ids_entry = tk.Entry(token_row, textvariable=self.stop_token_ids_var, width=20, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.stop_token_ids_entry.pack(side="left", padx=(10, 0))

        mode_row = tk.Frame(self.prompt_card, bg=PALETTE["panel"])
        mode_row.pack(fill="x", pady=(12, 0))
        tk.Label(mode_row, text="Runtime mode", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.chat_mode_picker = ttk.Combobox(
            mode_row,
            textvariable=self.chat_mode_var,
            state="readonly",
            width=28,
            values=("Quality (full stack)", "Balanced (32 layers)", "Fast (8 layers)"),
        )
        self.chat_mode_picker.pack(side="left", padx=(10, 0))
        self.chat_mode_picker.bind("<<ComboboxSelected>>", self.on_chat_mode_selected)

        stop_row = tk.Frame(self.prompt_card, bg=PALETTE["panel"])
        stop_row.pack(fill="x", pady=(12, 0))
        tk.Label(stop_row, text="Stop strings (use | between values)", font=("Segoe UI Semibold", 10), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(side="left")
        self.stop_strings_entry = tk.Entry(stop_row, textvariable=self.stop_strings_var, font=("Segoe UI", 10), bg="#fffdf8", fg=PALETTE["text"], relief="solid", borderwidth=1)
        self.stop_strings_entry.pack(side="left", fill="x", expand=True, padx=(10, 0))

        prompt_actions = tk.Frame(self.prompt_card, bg=PALETTE["panel"])
        prompt_actions.pack(fill="x", pady=(12, 0))
        self.send_button = self._button(prompt_actions, "Send", self.run_prompt_test)
        self.send_button.pack(side="left")
        self._button(prompt_actions, "Clear Chat", self.clear_chat_history).pack(side="left", padx=(10, 0))

        self.prompt_summary = self._body_label(self.prompt_card)
        self.prompt_summary.configure(
            text=f"{_chat_runtime_hint(mode_label=self.chat_mode_var.get())} {_chat_mode_hint(self.chat_mode_var.get())}"
        )

        tk.Label(self.prompt_card, text="Conversation", font=("Segoe UI Semibold", 11), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(anchor="w", pady=(12, 0))
        self.chat_history_text = tk.Text(
            self.prompt_card,
            height=10,
            bg="#fffdf8",
            fg=PALETTE["text"],
            relief="solid",
            borderwidth=1,
            wrap="word",
            font=("Segoe UI", 10),
        )
        self.chat_history_text.pack(fill="x", pady=(8, 0))
        self.chat_history_text.configure(state="disabled")

        tk.Label(self.prompt_card, text="Latest response", font=("Segoe UI Semibold", 11), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(anchor="w", pady=(12, 0))
        self.prompt_output_text = tk.Text(
            self.prompt_card,
            height=5,
            bg="#fffdf8",
            fg=PALETTE["text"],
            relief="solid",
            borderwidth=1,
            wrap="word",
            font=("Segoe UI", 10),
        )
        self.prompt_output_text.pack(fill="x", pady=(8, 0))
        self.prompt_output_text.configure(state="disabled")

        tk.Label(self.prompt_card, text="Session details", font=("Segoe UI Semibold", 11), fg=PALETTE["muted"], bg=PALETTE["panel"]).pack(anchor="w", pady=(12, 0))
        self.prompt_details_text = tk.Text(
            self.prompt_card,
            height=9,
            bg=PALETTE["panel"],
            fg=PALETTE["text"],
            relief="flat",
            wrap="word",
            font=("Consolas", 10),
        )
        self.prompt_details_text.pack(fill="x", pady=(8, 0))
        self.prompt_details_text.configure(state="disabled")

        actions = tk.Frame(shell, bg=PALETTE["bg"])
        actions.pack(fill="x", pady=(14, 0))
        self._button(actions, "Refresh Status", self.refresh).pack(side="left")
        self._button(actions, "Open Model Folder", self.open_model_folder).pack(side="left", padx=(10, 0))
        self._button(actions, "Retry Preflight", self.refresh).pack(side="left", padx=(10, 0))
        self._button(actions, "Advance Stream", self.advance_stream).pack(side="left", padx=(10, 0))
        self._button(actions, "Safe Advance", self.safe_advance_stream).pack(side="left", padx=(10, 0))
        self._button(actions, "View Details", self.show_details).pack(side="left", padx=(10, 0))

    def _sync_scroll_region(self, event=None) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _resize_canvas_content(self, event) -> None:
        if self._content_window_id is not None:
            self.canvas.itemconfigure(self._content_window_id, width=event.width)

    def _on_mousewheel(self, event) -> None:
        if self.root.focus_displayof() is None:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _card(self, parent: tk.Misc) -> tk.Frame:
        return tk.Frame(parent, bg=PALETTE["panel"], highlightbackground=PALETTE["line"], highlightthickness=1, padx=18, pady=16)

    def _section_title(self, parent: tk.Misc, text: str) -> tk.Label:
        return tk.Label(parent, text=text, font=("Segoe UI Semibold", 15), fg=PALETTE["text"], bg=PALETTE["panel"])

    def _value_label(self, parent: tk.Misc) -> tk.Label:
        label = tk.Label(parent, text="", font=("Consolas", 11), fg=PALETTE["text"], bg=PALETTE["panel"], anchor="w")
        label.pack(fill="x", pady=1)
        return label

    def _body_label(self, parent: tk.Misc) -> tk.Label:
        label = tk.Label(parent, text="", font=("Segoe UI", 11), fg=PALETTE["text"], bg=PALETTE["panel"], justify="left", wraplength=860, anchor="w")
        label.pack(fill="x", pady=(10, 0))
        return label

    def _button(self, parent: tk.Misc, text: str, command) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            font=("Segoe UI Semibold", 10),
            bg="#f0e5d0",
            fg=PALETTE["text"],
            activebackground="#e5d5ba",
            activeforeground=PALETTE["text"],
            relief="flat",
            padx=14,
            pady=8,
            cursor="hand2",
        )

    def refresh(self) -> None:
        try:
            self.reload_models(quiet=True)
            model = build_status_screen_model(model_id=self.model_id, model_label=self.model_label)
        except Exception as exc:  # pragma: no cover - UI guard
            messagebox.showerror("pcketlm", f"Failed to refresh status:\n\n{exc}")
            return
        self.current_model = model
        self._render(model)

    def _render(self, model: StatusScreenModel) -> None:
        self.identity_title.configure(text=model.model_label)
        self.identity_meta.configure(
            text=(
                f"Family: {model.family_label}    Support: {model.family_runtime_status}    Format: {model.format_label}\n"
                f"Source: {model.source_label}\n"
                f"Folder: {model.model_dir}"
            )
        )
        self.model_actions_label.configure(text="\n".join(f"- {action}" for action in model.model_actions))

        self.acq_status.configure(text=f"Acquisition  |  {model.acquisition_status}", fg=_pill_color(model.acquisition_status))
        self.acq_progress["value"] = model.acquisition_progress_pct
        self.acq_progress_label.configure(text=f"Progress: {model.acquisition_progress_text}")
        self.acq_download.configure(text=f"Downloaded: {model.acquisition_download_text}")
        self.acq_shards.configure(text=f"Shards: {model.acquisition_shards_text}")
        self.acq_summary.configure(text=model.acquisition_summary)

        self.runtime_status_label.configure(text=f"Runtime Readiness  |  {model.runtime_status}", fg=_pill_color(model.runtime_status))
        self.runtime_rows["Status"].configure(text=model.runtime_status, fg=_pill_color(model.runtime_status))
        self.runtime_rows["Config"].configure(text=model.config_status, fg=_pill_color(model.config_status))
        self.runtime_rows["Tokenizer"].configure(text=model.tokenizer_status, fg=_pill_color(model.tokenizer_status))
        self.runtime_rows["Weights"].configure(text=model.weights_status, fg=_pill_color(model.weights_status))
        self.runtime_rows["Runtime libs"].configure(text=model.runtime_libs_status, fg=_pill_color(model.runtime_libs_status))
        self.runtime_rows["Load attempt"].configure(text=model.load_status, fg=_pill_color(model.load_status))
        self.runtime_summary.configure(text=model.runtime_summary)

        self.streaming_status_label.configure(
            text=f"Streaming Control  |  {model.streaming_status}",
            fg=_pill_color(model.streaming_status),
        )
        self.streaming_action.configure(text=f"Next action: {model.streaming_action_label}", fg=_pill_color(model.streaming_status))
        self.streaming_rotation.configure(text=model.streaming_rotation_text)
        self.streaming_cache.configure(text=model.streaming_cache_text)
        self.streaming_windows.configure(text=model.streaming_window_text)
        self.streaming_summary.configure(text=model.streaming_summary)

        self.blockers_text.configure(state="normal")
        self.blockers_text.delete("1.0", "end")
        if model.blocker_category or model.blocker_severity:
            self.blockers_text.insert(
                "end",
                f"Category: {model.blocker_category or 'unknown'}\nSeverity: {model.blocker_severity or 'unknown'}\n\n",
            )
        for blocker in model.blockers:
            self.blockers_text.insert("end", f"- {blocker}\n")
        if model.warnings:
            self.blockers_text.insert("end", "\nWarnings:\n")
            for warning in model.warnings:
                self.blockers_text.insert("end", f"- {warning}\n")
        self.blockers_text.configure(state="disabled")

        self.next_step_label.configure(text=model.next_step)

    def reload_models(self, quiet: bool = False) -> None:
        self.model_options = list_status_screen_options()
        labels = [option.selection_label for option in self.model_options]
        self.model_picker["values"] = labels
        current_index = 0
        for index, option in enumerate(self.model_options):
            if option.model_id == self.model_id:
                current_index = index
                break
        if self.model_options:
            selected = self.model_options[current_index]
            self.model_id = selected.model_id
            self.model_label = selected.model_label
            self.model_var.set(selected.selection_label)
        else:
            self.model_var.set("No models detected")
            if not quiet:
                messagebox.showinfo("pcketlm", "No registered or detected local models were found yet.")

    def on_model_selected(self, event=None) -> None:
        selection = self.model_var.get()
        for option in self.model_options:
            if option.selection_label == selection:
                self.model_id = option.model_id
                self.model_label = option.model_label
                self.refresh()
                break

    def on_chat_mode_selected(self, event=None) -> None:
        self.prompt_summary.configure(
            text=f"{_chat_runtime_hint(mode_label=self.chat_mode_var.get())} {_chat_mode_hint(self.chat_mode_var.get())}"
        )

    def open_model_folder(self) -> None:
        if not self.current_model:
            return
        os.startfile(self.current_model.model_dir)

    def advance_stream(self) -> None:
        if not self.current_model:
            return
        try:
            result = advance_streaming_runtime(self.current_model.model_id, self.current_model.model_dir)
        except Exception as exc:  # pragma: no cover - UI guard
            messagebox.showerror("pcketlm", f"Failed to advance stream:\n\n{exc}")
            return

        if result.executed:
            messagebox.showinfo("pcketlm", f"{result.action_label} completed.")
        else:
            joined = "\n".join(result.blockers) if result.blockers else result.summary
            messagebox.showinfo("pcketlm", f"{result.action_label}\n\n{joined}")
        self.refresh()

    def safe_advance_stream(self) -> None:
        if not self.current_model:
            return
        try:
            result = advance_streaming_runtime_safely(self.current_model.model_id, self.current_model.model_dir)
        except Exception as exc:  # pragma: no cover - UI guard
            messagebox.showerror("pcketlm", f"Failed to safe-advance stream:\n\n{exc}")
            return

        if result.executed:
            messagebox.showinfo("pcketlm", f"{result.action_label} completed.")
        else:
            joined = "\n".join(result.blockers) if result.blockers else result.summary
            messagebox.showinfo("pcketlm", f"{result.action_label}\n\n{joined}")
        self.refresh()

    def show_details(self) -> None:
        if not self.current_model:
            return
        top = tk.Toplevel(self.root)
        top.title("pcketlm details")
        top.geometry("820x620")
        top.configure(bg=PALETTE["panel"])
        text = tk.Text(top, bg=PALETTE["panel"], fg=PALETTE["text"], font=("Consolas", 10), wrap="none")
        text.pack(fill="both", expand=True, padx=12, pady=12)
        details = dict(self.current_model.details)
        if self.last_prompt_result is not None:
            details["last_prompt_result"] = self.last_prompt_result.to_dict()
        text.insert("1.0", json.dumps(details, indent=2))
        text.configure(state="disabled")

    def open_chat_home(self) -> None:
        self.canvas.update_idletasks()
        y_position = max(self.prompt_card.winfo_y() - 12, 0)
        bbox = self.canvas.bbox("all")
        total_height = bbox[3] if bbox else self.body.winfo_height()
        if total_height > 0:
            self.canvas.yview_moveto(min(y_position / total_height, 1.0))
        self.prompt_text.focus_set()

    def show_planned_action(self, action: str) -> None:
        messagebox.showinfo(f"pcketlm {action}", _planned_action_message(action))

    def show_profile_templates(self) -> None:
        messagebox.showinfo(
            "pcketlm Personalize",
            (
                "First safe profile targets are ready as templates. Artifact generation is not enabled yet.\n\n"
                f"{profile_templates_summary()}"
            ),
        )

    def show_benchmark_readiness(self) -> None:
        if not self.current_model:
            return
        model_id = self.current_model.model_id
        model_dir = self.current_model.model_dir
        self.prompt_summary.configure(
            text="Benchmark is running locally across Fast, Balanced, and Quality. This can take a few minutes."
        )

        def worker() -> None:
            try:
                benchmark = run_measured_benchmark(model_id, model_dir)
            except Exception as exc:  # pragma: no cover - UI guard
                self.root.after(0, lambda exc=exc: messagebox.showerror("pcketlm Benchmark", f"Benchmark failed:\n\n{exc}"))
                return
            self.root.after(0, lambda benchmark=benchmark: self._finish_measured_benchmark(benchmark))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_measured_benchmark(self, benchmark) -> None:
        case_lines = "\n".join(
            (
                f"- {case.label}: {case.elapsed_seconds}s, "
                f"{'ready' if case.ready else 'blocked'}, "
                f"output: {case.generated_text or '(empty)'}"
            )
            for case in benchmark.cases
        )
        blockers = "\n".join(f"- {blocker}" for blocker in benchmark.blockers) if benchmark.blockers else "- none"
        warnings = "\n".join(f"- {warning}" for warning in benchmark.warnings) if benchmark.warnings else "- none"
        settings = benchmark.runtime_settings
        settings_lines = "\n".join(
            [
                f"- math dtype: {settings.get('math_dtype', 'unknown')}",
                f"- lm_head chunk rows: {settings.get('lm_head_chunk_rows', 'unknown')}",
            ]
        )
        self.prompt_summary.configure(text=f"Benchmark saved: {benchmark.benchmark_path}")
        messagebox.showinfo(
            "pcketlm Benchmark",
            (
                f"Status: {benchmark.status}\n\n"
                f"{benchmark.summary}\n\n"
                f"Modes:\n{case_lines or '- none'}\n\n"
                f"Runtime settings:\n{settings_lines}\n\n"
                f"Warnings:\n{warnings}\n\n"
                f"Blockers:\n{blockers}\n\n"
                f"Saved: {benchmark.benchmark_path}"
            ),
        )

    def clear_chat_history(self) -> None:
        self.chat_history.clear()
        self.chat_history_text.configure(state="normal")
        self.chat_history_text.delete("1.0", "end")
        self.chat_history_text.configure(state="disabled")

    def _set_prompt_busy(self, busy: bool) -> None:
        self.prompt_run_active = busy
        self.send_button.configure(
            text="Working..." if busy else "Send",
            state="disabled" if busy else "normal",
        )
        if busy:
            try:
                max_new_tokens = int(self.max_new_tokens_var.get().strip())
            except ValueError:
                max_new_tokens = 2
            self.prompt_summary.configure(
                text=f"Generating locally. {_chat_runtime_hint(max_new_tokens, self.chat_mode_var.get())}"
            )

    def run_prompt_test(self) -> None:
        if not self.current_model:
            return
        if self.prompt_run_active:
            return

        prompt = self.prompt_text.get("1.0", "end").strip()
        if not prompt:
            messagebox.showerror("pcketlm", "Prompt cannot be empty.")
            return

        try:
            max_new_tokens = int(self.max_new_tokens_var.get().strip())
        except ValueError:
            messagebox.showerror("pcketlm", "Max new tokens must be a valid integer.")
            return
        if max_new_tokens <= 0:
            messagebox.showerror("pcketlm", "Max new tokens must be greater than 0.")
            return
        try:
            min_new_tokens = int(self.min_new_tokens_var.get().strip())
        except ValueError:
            messagebox.showerror("pcketlm", "Min new tokens must be a valid integer.")
            return
        if min_new_tokens <= 0:
            messagebox.showerror("pcketlm", "Min new tokens must be greater than 0.")
            return

        try:
            repetition_penalty = float(self.repetition_penalty_var.get().strip())
        except ValueError:
            messagebox.showerror("pcketlm", "Repetition penalty must be a valid number.")
            return
        if repetition_penalty <= 0:
            messagebox.showerror("pcketlm", "Repetition penalty must be greater than 0.")
            return
        try:
            top_p = float(self.top_p_var.get().strip())
        except ValueError:
            messagebox.showerror("pcketlm", "Top-p must be a valid number.")
            return
        if top_p <= 0 or top_p > 1:
            messagebox.showerror("pcketlm", "Top-p must be greater than 0 and no more than 1.")
            return

        stop_token_ids, parse_blockers = _parse_stop_token_ids(self.stop_token_ids_var.get())
        if parse_blockers:
            messagebox.showerror("pcketlm", "\n".join(parse_blockers))
            return
        stop_strings = _parse_stop_strings(self.stop_strings_var.get())

        system_prompt = self.system_prompt_var.get().strip() or None
        model_id = self.current_model.model_id
        apply_chat_format = not self.raw_prompt_var.get()
        layer_budget = _chat_layer_budget(self.chat_mode_var.get())
        self._set_prompt_busy(True)

        def worker() -> None:
            try:
                result = _run_desktop_chat_generation(
                    model_id,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    top_p=top_p,
                    layer_count=layer_budget,
                    repetition_penalty=repetition_penalty,
                    system_prompt=system_prompt,
                    apply_chat_format=apply_chat_format,
                    stop_token_ids=stop_token_ids or None,
                    stop_strings=stop_strings or None,
                )
            except Exception as exc:  # pragma: no cover - UI guard
                self.root.after(0, lambda: self._finish_prompt_error(exc))
                return
            self.root.after(0, lambda: self._finish_prompt_result(prompt, result))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_prompt_error(self, exc: Exception) -> None:
        self._set_prompt_busy(False)
        messagebox.showerror("pcketlm", f"Chat failed:\n\n{exc}")

    def _finish_prompt_result(self, prompt: str, result) -> None:
        self._set_prompt_busy(False)
        self.last_prompt_result = result
        self.chat_history.append(("User", prompt))
        self.chat_history.append(("Assistant", result.generated_text if result.generated_text else "(no generated text)"))
        self.prompt_summary.configure(
            text=(
                f"Status: {'Ready' if result.ready else 'Blocked'}  |  "
                f"Strategy: {result.strategy}  |  "
                f"Stop: {result.stop_reason or 'none'}"
            )
        )

        generated_output = result.generated_text if result.generated_text else "(no generated text)"
        full_text = result.full_text if result.full_text else prompt
        output_body = (
            f"Generated text:\n{generated_output}\n\n"
            f"Full text:\n{full_text}"
        )
        chat_body = "\n\n".join(f"{speaker}:\n{text}" for speaker, text in self.chat_history)
        self.chat_history_text.configure(state="normal")
        self.chat_history_text.delete("1.0", "end")
        self.chat_history_text.insert("1.0", chat_body)
        self.chat_history_text.configure(state="disabled")

        self.prompt_output_text.configure(state="normal")
        self.prompt_output_text.delete("1.0", "end")
        self.prompt_output_text.insert("1.0", output_body)
        self.prompt_output_text.configure(state="disabled")

        self.prompt_details_text.configure(state="normal")
        self.prompt_details_text.delete("1.0", "end")
        self.prompt_details_text.insert("1.0", _summarize_prompt_result(result))
        self.prompt_details_text.configure(state="disabled")


def run() -> None:
    root = tk.Tk()
    PcketLmStatusApp(root)
    root.mainloop()


if __name__ == "__main__":
    run()
