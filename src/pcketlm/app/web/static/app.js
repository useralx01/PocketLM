const state = {
  activeScreen: "chat",
  mode: "GGUF",
  status: null,
  messages: [],
  activeJobId: null,
  activeStartedAt: null,
  activeTimer: null,
  cancelRequested: false,
  lastRequest: null,
  lastRuntimeDetails: null,
  activeProfileId: "",
  statusRefreshTimer: null,
  sessionId: globalThis.crypto?.randomUUID ? globalThis.crypto.randomUUID() : `${Date.now()}-${Math.random()}`,
};

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => Array.from(document.querySelectorAll(selector));

function escapeText(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

function statusClass(value) {
  const normalized = String(value || "").toLowerCase();
  if (normalized === "proven" || normalized === "ready" || normalized === "active") return "status-proven";
  if (normalized === "experimental") return "status-experimental";
  if (normalized === "unsupported" || normalized === "blocked") return "status-unsupported";
  return "status-missing";
}

function runLaunchSequence() {
  const overlay = $("#launch-sequence");
  const status = $("#launch-status");
  if (!overlay || !status) return;
  const steps = ["Starting local control", "Checking runtimes", "Reading model evidence", "Ready"];
  steps.forEach((step, index) => window.setTimeout(() => { status.textContent = step; }, index * 260));
  window.setTimeout(() => overlay.classList.add("complete"), 1150);
  window.setTimeout(() => overlay.remove(), 1600);
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function setScreen(screen) {
  state.activeScreen = screen;
  $$(".nav-item").forEach((item) => item.classList.toggle("active", item.dataset.screen === screen));
  $$(".screen").forEach((item) => item.classList.toggle("active", item.id === `screen-${screen}`));
}

function setMode(mode) {
  const previousMode = state.mode;
  state.mode = mode;
  $$("#mode-picker button").forEach((button) => button.classList.toggle("selected", button.dataset.mode === mode));
  const hints = {
    GGUF: "GGUF uses the installed model's native llama.cpp chat template.",
    "Direct Quality": "Direct Quality uses Pocket's dense runtime and is currently slow.",
    "Direct Quick": "Direct Quick uses the dense runtime for one-token checks.",
    "Direct Agent": "Direct Agent caps dense-runtime replies to two tokens for short local work.",
    "Direct Balanced": "Direct Balanced is a dense-runtime speed preview. It can drift or answer oddly.",
    "Direct Fast": "Direct Fast is only for direct-runtime smoke tests. Output quality can be rough.",
  };
  const tokenInput = $("#max-new-tokens");
  if (tokenInput) {
    tokenInput.max = mode === "GGUF" ? 32768 : 16;
    if (mode === "GGUF" && previousMode !== "GGUF") tokenInput.value = 4096;
    if (mode === "Direct Quick") tokenInput.value = 1;
    if (mode === "Direct Agent") tokenInput.value = Math.min(Number(tokenInput.value || 2), 2);
    if (Number(tokenInput.value || 4) > Number(tokenInput.max)) tokenInput.value = tokenInput.max;
  }
  $("#chat-subtitle").textContent = hints[mode] || hints.GGUF;
}

function renderStatus(payload) {
  state.status = payload;
  const model = payload.active_model;
  $("#sidebar-model").textContent = model.label || model.model_id;
  $("#sidebar-runtime").textContent = `${model.effective_runtime_status || model.runtime_status} - ${model.streaming_status}`;
  $("#model-pill").textContent = model.label || model.model_id;
  renderSpeedTarget(payload.qwen14b_speed_target || {});
  $("#settings-runtime").textContent = `${model.effective_runtime_status || model.runtime_status}. ${model.effective_summary || model.summary || ""}`;
  $("#settings-storage").textContent = payload.project_root;
  renderRuntimeGrid("#settings-runtime-grid", {
    elapsed_seconds: null,
    timings: {},
    runtime_settings: payload.runtime_settings || {},
    warm_runner: payload.warm_runner || {},
  });
  renderBackendReport(payload.backend_report);
  renderLoadRuntime(payload);
  renderDownloadMeters(payload.downloads);
  renderGgufServer(payload);
  renderProfileSelect(payload.profiles || []);
  renderRuntimePresetControls(payload);

  $("#model-list").innerHTML = (payload.models || []).map((item) => {
    const stateLabel = item.runnable ? item.implementation_status : item.source_status === "missing" ? "Missing" : item.implementation_status;
    return `
    <div class="item">
      <div class="item-title"><span>${escapeText(item.label)}</span><span class="status-chip ${statusClass(stateLabel)}">${escapeText(stateLabel)}</span></div>
      <p>${escapeText(item.family)} / ${escapeText(item.capability)} / ${escapeText(item.backend)}</p>
      <div class="mini-metrics"><span>${escapeText(item.source_status)}</span><span>${escapeText(item.source_origin)}</span><span>${escapeText(item.format_name)}</span></div>
    </div>
  `; }).join("") || `<p class="muted">No registered models found.</p>`;

  $("#profile-list").innerHTML = (payload.profiles || []).map((item) => `
    <div class="item">
      <div class="item-title"><span>${escapeText(item.label)}</span><span class="pill">${escapeText(item.runtime_mode || "Quality")}</span></div>
      <p>${escapeText(item.summary)}</p>
      <div class="mini-metrics">
        <span>${escapeText(item.artifact_ready ? "artifact ready" : "profile record")}</span>
        <span>${escapeText((item.capability_priorities || []).slice(0, 2).join(", ") || "general")}</span>
      </div>
    </div>
  `).join("");
  renderCompare(payload.profile_compare);

  renderBenchmark(payload.benchmark, payload.benchmark_history);
  renderBackendComparison(payload.backend_comparison);
}

function renderSupervisor(report) {
  const chip = $("#supervisor-status");
  const summary = $("#supervisor-summary");
  const grid = $("#supervisor-grid");
  if (!chip || !summary || !grid) return;
  const reportStatus = report.status || "unknown";
  const system = report.system || {};
  const gpu = report.gpu || {};
  const modelSummary = report.summary || {};
  chip.textContent = reportStatus;
  chip.className = `status-chip ${statusClass(reportStatus)}`;
  summary.textContent = reportStatus === "ready"
    ? "This desktop has a runnable local model. The report stays on this computer."
    : reportStatus === "partial"
      ? "PocketLM works, but this desktop still needs a model before generation can be tested."
      : "This installation has a blocking compatibility issue.";
  const metrics = [
    ["Install", String(report.installation_id || "unknown").slice(0, 8)],
    ["System", `${system.os || "Unknown"} ${system.os_release || ""}`.trim()],
    ["CPU threads", system.logical_cpu_count ?? "n/a"],
    ["Memory", system.memory_total_gb == null ? "n/a" : `${system.memory_total_gb} GB`],
    ["GPU", gpu.available ? `${gpu.device_count} CUDA` : "CPU only"],
    ["Models", modelSummary.runnable_model_count ?? 0],
  ];
  grid.innerHTML = metrics.map(([label, value]) => `
    <div class="runtime-metric"><span>${escapeText(label)}</span><strong>${escapeText(value)}</strong></div>
  `).join("");
}

async function runSupervisorCheck() {
  const button = $("#supervisor-check");
  const summary = $("#supervisor-summary");
  if (button) button.disabled = true;
  if (summary) summary.textContent = "Checking hardware, runtimes, and local models...";
  try {
    renderSupervisor(await api("/api/supervisor/check", { method: "POST", body: "{}" }));
  } catch (error) {
    if (summary) summary.textContent = `Check failed: ${error.message}`;
  } finally {
    if (button) button.disabled = false;
  }
}

function renderSpeedTarget(target) {
  const note = $("#speed-target-note");
  const loadButton = $("#chat-load-gguf");
  if (!note) return;
  if (!target.applies) {
    note.textContent = target.summary || "";
    if (loadButton) loadButton.hidden = true;
    return;
  }
  const server = target.gguf_server_ready ? "loaded" : target.gguf_server_running ? "loading" : "not loaded";
  const direct = target.slow_direct_seconds_per_token
    ? `Direct is about ${formatSeconds(target.slow_direct_seconds_per_token)}/token.`
    : "Direct is slow.";
  const load = target.estimated_cold_load_seconds
    ? `Cold load about ${formatSeconds(target.estimated_cold_load_seconds)}.`
    : "";
  note.textContent = `Speed target 2-4s/token. GGUF server ${server}. ${direct} ${load}`.trim();
  if (loadButton) {
    loadButton.hidden = target.gguf_server_ready;
    loadButton.disabled = target.gguf_server_running;
    loadButton.textContent = target.gguf_server_running ? "Loading fast model..." : "Load fast model";
  }
}

function renderProfileSelect(profiles) {
  const select = $("#profile-select");
  if (!select) return;
  const current = state.activeProfileId;
  select.innerHTML = `<option value="">Default</option>` + profiles.map((profile) => `
    <option value="${escapeText(profile.profile_id)}">${escapeText(profile.label)}</option>
  `).join("");
  select.value = profiles.some((profile) => profile.profile_id === current) ? current : "";
}

function renderRuntimePresetControls(payload) {
  const select = $("#tensor-cache-preset");
  const policy = payload.runtime_settings?.tensor_residency_policy || {};
  if (select) select.value = policy.tensor_cache_preset || "standard";
  const status = $("#tensor-cache-preset-status");
  if (status) {
    status.textContent = policy.memory_guard_active
      ? "Low RAM guard active"
      : policy.adaptive_boost_active
        ? "Boosted cache active"
        : "Standard cache active";
  }
  const warmSelect = $("#agent-warm-runner");
  const warmMode = payload.runtime_settings?.agent_warm_runner
    || payload.runtime_settings?.saved_runtime_settings?.agent_warm_runner
    || "off";
  if (warmSelect) warmSelect.value = warmMode;
  const warmStatus = $("#agent-warm-runner-status");
  if (warmStatus) {
    warmStatus.textContent = warmMode === "off"
      ? "Agent warm runner off"
      : `Agent warm runner ${warmMode}`;
  }
}

function renderCompare(compare) {
  if (!compare) {
    $("#compare-default").innerHTML = `<p class="muted">No profile comparison loaded.</p>`;
    $("#compare-list").innerHTML = "";
    return;
  }
  $("#compare-default").innerHTML = `
    <div class="item">
      <div class="item-title"><span>Default</span><span class="pill">${escapeText(compare.default_runtime_mode || "Quality")}</span></div>
      <p>${compare.default_elapsed_seconds ? `Latest Quality benchmark: ${formatSeconds(compare.default_elapsed_seconds)}.` : "Run a benchmark to attach current timing to this baseline."}</p>
    </div>
  `;
  $("#compare-list").innerHTML = (compare.profiles || []).map((profile) => `
    <div class="item">
      <div class="item-title"><span>${escapeText(profile.label)}</span><span class="pill">${escapeText(profile.runtime_mode)}</span></div>
      <p>${escapeText(profile.difference_from_default)}<br>${escapeText(profile.summary)}</p>
      <div class="mini-metrics">
        <span>${formatSeconds(profile.reference_elapsed_seconds)}</span>
        <span>${escapeText(profile.reference_output || "benchmark needed")}</span>
      </div>
    </div>
  `).join("") || `<p class="muted">No saved profiles found.</p>`;
}

function renderLoadRuntime(payload) {
  const model = payload.active_model || {};
  const direct = payload.direct_runtime || {};
  const engine = payload.engine_decision || {};
  const backendReport = payload.backend_report || {};
  const recommendedBackend = engine.recommended_backend_id || backendReport.recommended_backend_id || "direct-cpu";
  const guardrails = payload.model_guardrails || payload.speed_status?.model_guardrails || {};
  $("#load-runtime-card").innerHTML = `
    <div class="item">
      <div class="item-title"><span>${escapeText(model.label || model.model_id)}</span><span class="pill">${escapeText(model.effective_runtime_status || model.runtime_status || "Unknown")}</span></div>
      <p>${escapeText(direct.summary || model.effective_summary || model.summary || "")}</p>
    </div>
    <div class="item compact-item">
      <div class="item-title"><span>Recommended backend</span><span class="pill">${escapeText(recommendedBackend)}</span></div>
      <p>${escapeText(backendReport.recommended_summary || engine.summary || "Direct CPU remains available as the dense fallback.")}</p>
      <div class="mini-metrics">
        <span>${escapeText(`active ${engine.selected_engine || "direct-cpu"}`)}</span>
        <span>${escapeText(engine.selected_backend || "torch-cpu")}</span>
      </div>
    </div>
    <div class="item compact-item">
      <div class="item-title"><span>Direct runtime guard</span><span class="pill">${escapeText(guardrails.status || "standard")}</span></div>
      <p>${escapeText(guardrails.summary || "Standard direct-runtime safety checks are active.")}</p>
      <div class="mini-metrics">
        <span>${escapeText(guardrails.free_ram_mb ? `${guardrails.free_ram_mb} MB free` : "RAM n/a")}</span>
        <span>${escapeText(guardrails.proven_max_new_tokens ? `${guardrails.proven_max_new_tokens} tokens proven` : "standard model")}</span>
        <span>${escapeText(guardrails.scoped_safetensor_handle_cache?.default_enabled === false ? "safe handles" : "auto handles")}</span>
      </div>
    </div>
    <div class="item">
      <div class="item-title"><span>Model folder</span><span class="pill">Local</span></div>
      <p>${escapeText(model.model_dir || "No folder detected")}</p>
    </div>
  `;
  $("#support-plan").innerHTML = (payload.compatibility_matrix || []).map((entry) => `
    <div class="item compact-item">
      <div class="item-title"><span>${escapeText(entry.label)}</span><span class="status-chip ${statusClass(entry.status)}">${escapeText(entry.status)}</span></div>
      <p>${escapeText(entry.capability)} / ${escapeText(entry.backend)} / chat ${escapeText(entry.chat_status)}</p>
      <p>${escapeText(entry.evidence)}</p>
      <div class="mini-metrics">
        <span>${escapeText(entry.local_ready ? "local path ready" : "weights absent")}</span>
        <span>${escapeText(entry.implementation_status)}</span>
        <span>${escapeText(entry.default_path ? "default when installed" : "specialized path")}</span>
      </div>
      ${(entry.blockers || []).length ? `<p class="blocker-text">${escapeText(entry.blockers.join(" "))}</p>` : ""}
    </div>
  `).join("");
}

function renderDownloadMeters(downloads) {
  const element = $("#download-meter-card");
  if (!element) return;
  const records = downloads?.records || [];
  if (!records.length) {
    element.innerHTML = `<p class="muted">No active model downloads.</p>`;
    return;
  }
  element.innerHTML = records.map((item) => {
    const pct = Number(item.progress_pct ?? 0);
    const boundedPct = Number.isFinite(pct) ? Math.max(0, Math.min(100, pct)) : 0;
    const label = item.model_id || "model";
    const status = item.status || "unknown";
    const onDisk = item.bytes_on_disk_gb ?? bytesToGiB(item.bytes_on_disk);
    const expected = item.expected_bytes_gb ?? bytesToGiB(item.expected_bytes);
    const fileCount = `${item.present_expected_file_count ?? 0}/${item.expected_file_count ?? "?"} files`;
    const updated = item.updated_at ? `updated ${formatClockTime(item.updated_at)}` : "waiting";
    return `
      <div class="item">
        <div class="item-title"><span>${escapeText(label)}</span><span class="pill">${escapeText(status)}</span></div>
        <div class="progress-track" aria-label="${escapeText(label)} download progress">
          <div class="progress-fill" style="width: ${boundedPct.toFixed(2)}%"></div>
        </div>
        <div class="mini-metrics">
          <span>${boundedPct.toFixed(2)}%</span>
          <span>${escapeText(onDisk)} / ${escapeText(expected)}</span>
          <span>${escapeText(fileCount)}</span>
          <span>${escapeText(updated)}</span>
        </div>
        ${item.error ? `<p class="muted">Error: ${escapeText(item.error)}</p>` : ""}
      </div>
    `;
  }).join("");
}

function renderGgufServer(payload) {
  const element = $("#gguf-server-card");
  if (!element) return;
  const model = payload.active_model || {};
  const backend = payload.gguf_backend || {};
  const server = backend.llama_server || {};
  const files = backend.model_files || [];
  const artifactSummary = backend.artifact_summary || {};
  const estimate = backend.load_estimate || {};
  const modelFile = files.find((file) => file.path === estimate.model_path)
    || files.find((file) => !(file.path || "").includes("-of-"))
    || files[0];
  const stateLabel = server.ready ? "Ready" : server.running ? "Loading" : "Unloaded";
  const ram = server.working_set_bytes ? formatBytes(server.working_set_bytes) : "0 MB";
  const expectedRam = estimate.expected_ram_mb ? `${estimate.expected_ram_mb} MB expected` : "RAM n/a";
  const coldLoad = estimate.estimated_cold_load_seconds ? formatSeconds(estimate.estimated_cold_load_seconds) : "n/a";
  const toggleAction = server.running ? "stop" : "start";
  const toggleDisabled = (!server.running && !modelFile) || (server.ready && toggleAction === "start");
  const artifactList = files.length
    ? files.map((file) => `
      <div class="mini-metrics artifact-row">
        <span>${escapeText(file.name || "GGUF file")}</span>
        <span>${escapeText(file.kind || "file")}</span>
        <span>${escapeText(file.size_gb ? `${file.size_gb} GB` : formatBytes(file.size_bytes))}</span>
        <span>${escapeText(file.location || file.source || "local")}</span>
      </div>
      <p class="muted">${escapeText(file.path || "")}</p>
    `).join("")
    : `<p class="muted">No GGUF artifact found for ${escapeText(model.model_id || "this model")}.</p>`;
  element.innerHTML = `
    <div class="item">
      <div class="item-title"><span>llama.cpp</span><span class="pill">${escapeText(stateLabel)}</span></div>
      <p>${escapeText(server.summary || backend.summary || "GGUF backend status unavailable.")}</p>
      <div class="mini-metrics">
        <span>${escapeText(server.pid ? `PID ${server.pid}` : "not running")}</span>
        <span>${escapeText(ram)}</span>
        <span>${escapeText(backend.llama_cli_available ? "runtime ready" : "runtime missing")}</span>
      </div>
      <div class="button-row">
        <button class="secondary compact" id="gguf-toggle-button" data-action="${toggleAction}" ${toggleDisabled ? "disabled" : ""}>${server.running ? "Unload" : "Load"}</button>
      </div>
    </div>
    <div class="item compact-item">
      <div class="item-title"><span>Selected GGUF</span><span class="pill">${escapeText(estimate.state || (modelFile ? "ready" : "missing"))}</span></div>
      <p>${escapeText(modelFile?.path || `No GGUF artifact found for ${model.model_id || "this model"}.`)}</p>
      <div class="mini-metrics">
        <span>${escapeText(modelFile?.name || "missing")}</span>
        <span>${escapeText(modelFile?.size_gb ? `${modelFile.size_gb} GB file` : "size n/a")}</span>
        <span>${escapeText(expectedRam)}</span>
        <span>${escapeText(`cold load ${coldLoad}`)}</span>
      </div>
      ${estimate.blockers?.length ? `<p class="muted">${escapeText(estimate.blockers.join(" "))}</p>` : ""}
    </div>
    <div class="item compact-item">
      <div class="item-title"><span>GGUF files</span><span class="pill">${files.length}</span></div>
      <p>${escapeText(artifactSummary.summary || "Disk usage unavailable.")}</p>
      <div class="mini-metrics">
        <span>${escapeText(`${artifactSummary.complete_file_count ?? 0} complete`)}</span>
        <span>${escapeText(`${artifactSummary.split_shard_count ?? 0} split`)}</span>
        <span>${escapeText(`${artifactSummary.total_size_gb ?? 0} GB total`)}</span>
      </div>
      ${artifactList}
    </div>
  `;
  $("#gguf-toggle-button")?.addEventListener("click", (event) => controlGgufServer(event.currentTarget.dataset.action || "status"));
}

function formatSeconds(value) {
  if (value === null || value === undefined || value === "") return "n/a";
  const number = Number(value);
  if (!Number.isFinite(number)) return String(value);
  return `${number.toFixed(number >= 10 ? 1 : 2)}s`;
}

function formatBytes(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "0 MB";
  return `${(number / (1024 * 1024)).toFixed(1)} MB`;
}

function bytesToGiB(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "0 GB";
  return `${(number / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function formatClockTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function runtimeRows(details) {
  const timings = details?.timings || {};
  const speed = details?.generation_speed || {};
  const settings = details?.runtime_settings || {};
  const warmRunner = details?.warm_runner || {};
  const warmMemory = warmRunner.memory || {};
  const cache = settings.tensor_residency || {};
  const policy = settings.tensor_residency_policy || {};
  return [
    ["Total", formatSeconds(details?.elapsed_seconds ?? timings.total)],
    ["Token speed", speed.generation_seconds_per_token ? `${speed.generation_seconds_per_token}s/token` : "n/a"],
    ["Speed target", speed.target_met ? "met" : speed.generation_seconds_per_token ? "missed" : "n/a"],
    ["Prefill stack", formatSeconds(timings.prefill_stack)],
    ["Continuation stack", formatSeconds(timings.continuation_stack)],
    ["Decode tail", formatSeconds((Number(timings.prefill_decode_tail || 0) + Number(timings.continuation_decode_tail || 0)) || timings.prefill_decode_tail)],
    ["Torch threads", settings.torch_threads ?? "n/a"],
    ["Math dtype", settings.math_dtype || "n/a"],
    ["LM head chunk", settings.lm_head_chunk_rows ?? "n/a"],
    ["Cache hits", cache.hits ?? "n/a"],
    ["Cache misses", cache.misses ?? "n/a"],
    ["Resident cache", formatBytes(cache.resident_bytes)],
    ["Cache preset", policy.tensor_cache_preset || "standard"],
    ["Cache policy", policy.memory_guard_active ? "low RAM" : `${policy.front_layer_count ?? "n/a"} front`],
    ["Cache cap", formatBytes(policy.max_resident_bytes)],
    ["Agent runner", warmRunner.state || "stopped"],
    ["Agent requests", warmRunner.request_count ?? 0],
    ["Agent last run", formatSeconds(warmRunner.last_latency_seconds)],
    ["Agent prefix", warmRunner.prefix_reuse_available ? `${warmRunner.reusable_token_count ?? 0} tokens` : "not ready"],
    ["Agent memory", warmMemory.process_working_set_mb ? `${warmMemory.process_working_set_mb} MB` : "n/a"],
  ];
}

function renderRuntimeGrid(selector, details) {
  const element = $(selector);
  if (!element) return;
  element.innerHTML = runtimeRows(details).map(([label, value]) => `
    <div class="runtime-metric">
      <span>${escapeText(label)}</span>
      <strong>${escapeText(value)}</strong>
    </div>
  `).join("");
}

function renderBackendReport(report) {
  const element = $("#backend-report");
  if (!element) return;
  if (!report || !Array.isArray(report.candidates)) {
    element.innerHTML = `<p class="muted">Backend report unavailable.</p>`;
    return;
  }
  element.innerHTML = `
    <div class="item">
      <div class="item-title"><span>Recommended</span><span class="pill">${escapeText(report.recommended_backend_id || "direct-cpu")}</span></div>
      <p>${escapeText(report.recommended_summary || "")}</p>
    </div>
    ${report.candidates.map((candidate) => `
      <div class="item compact-item">
        <div class="item-title">
          <span>${escapeText(candidate.label)}</span>
          <span class="pill">${escapeText(candidate.status)}</span>
        </div>
        <p>${escapeText(candidate.summary)}</p>
      </div>
    `).join("")}
  `;
}

function renderRuntimePanel(details) {
  state.lastRuntimeDetails = details;
  const panel = $("#runtime-panel");
  if (!panel) return;
  panel.hidden = !details;
  if (details) renderRuntimeGrid("#runtime-grid", details);
}

function renderBenchmark(run, history) {
  if (!run || !run.cases) {
    $("#benchmark-status").textContent = "No measured benchmark yet. Run one to compare modes.";
    $("#benchmark-table").innerHTML = "";
    $("#benchmark-history").innerHTML = "";
    return;
  }
  $("#benchmark-status").textContent = run.summary || run.status || "Measured benchmark loaded.";
  const runtimeSettings = run.runtime_settings || {};
  $("#benchmark-table").innerHTML = run.cases.map((item) => `
    <div class="benchmark-card">
      <div class="benchmark-head">
        <strong>${escapeText(item.label)}</strong>
        <span class="time">${formatSeconds(item.elapsed_seconds)}</span>
      </div>
      <p>${escapeText(item.generated_text || "(empty)")}</p>
      <div class="mini-metrics">
        <span>${escapeText(item.backend || "direct-cpu")}</span>
        <span>${escapeText(item.prompt_kind || "short")}</span>
        <span>${escapeText(item.backend?.includes("gguf") ? `${item.max_new_tokens || "?"} tokens` : `${item.layer_count ?? "full"} layers`)}</span>
        <span>${escapeText(item.ready ? "ready" : "blocked")}</span>
        <span>${escapeText(item.tensor_residency?.hits ?? 0)} hits</span>
        <span>${formatBytes(item.tensor_residency?.resident_bytes)}</span>
        ${item.timing_summary?.stack_seconds ? `<span>${formatSeconds(item.timing_summary.stack_seconds)} stack</span>` : ""}
        ${item.timing_summary?.tensor_load_seconds ? `<span>${formatSeconds(item.timing_summary.tensor_load_seconds)} loads</span>` : ""}
        ${item.timing_summary?.bottleneck ? `<span>${escapeText(item.timing_summary.bottleneck)}</span>` : ""}
      </div>
    </div>
  `).join("");
  if (runtimeSettings.math_dtype || runtimeSettings.torch_threads || runtimeSettings.lm_head_chunk_rows) {
    $("#benchmark-table").insertAdjacentHTML("beforeend", `
      <div class="benchmark-card settings-card">
        <div class="benchmark-head"><strong>Runtime Settings</strong><span class="pill">Latest</span></div>
        <div class="mini-metrics">
          <span>${escapeText(runtimeSettings.math_dtype || "dtype n/a")}</span>
          <span>${escapeText(runtimeSettings.torch_threads ?? "threads n/a")} threads</span>
          <span>${escapeText(runtimeSettings.lm_head_chunk_rows ?? "chunk n/a")} chunk</span>
        </div>
      </div>
    `);
  }
  renderBenchmarkHistory(history);
}

function renderBackendComparison(comparison) {
  const element = $("#backend-comparison-table");
  if (!element) return;
  if (!comparison || !Array.isArray(comparison.rows)) {
    element.innerHTML = `<p class="muted">Run a benchmark to compare backends.</p>`;
    return;
  }
  const tags = comparison.tags || [];
  const rowsById = Object.fromEntries((comparison.rows || []).map((row) => [row.backend_id, row]));
  const orderedTags = ["fastest", "best quality", "lowest RAM", "recommended"];
  const tagCards = orderedTags.map((tag) => {
    const tagInfo = tags.find((item) => item.tag === tag);
    const row = tagInfo ? rowsById[tagInfo.backend_id] : null;
    if (!row) {
      return `
        <div class="benchmark-card">
          <div class="benchmark-head"><strong>${escapeText(tag)}</strong><span class="pill">needed</span></div>
          <p>Run both GGUF and full benchmarks to fill this row.</p>
        </div>
      `;
    }
    return `
      <div class="benchmark-card">
        <div class="benchmark-head"><strong>${escapeText(tag)}</strong><span class="pill">${escapeText(row.label)}</span></div>
        <p>${escapeText(tagInfo.reason || row.summary || "")}</p>
        <div class="mini-metrics">
          <span>${escapeText(row.ready ? "measured" : row.status || "needed")}</span>
          <span>${escapeText(row.seconds_per_token ? `${row.seconds_per_token}s/token` : "speed n/a")}</span>
          <span>${escapeText(row.elapsed_seconds ? formatSeconds(row.elapsed_seconds) : "time n/a")}</span>
          <span>${escapeText(row.memory_mb ? `${row.memory_mb} MB ${row.memory_kind || "memory"}` : "RAM n/a")}</span>
        </div>
      </div>
    `;
  }).join("");
  const missingRows = (comparison.rows || [])
    .filter((row) => !row.ready)
    .map((row) => `
      <div class="benchmark-card">
        <div class="benchmark-head"><strong>${escapeText(row.label)}</strong><span class="pill">${escapeText(row.status || "needed")}</span></div>
        <p>${escapeText(row.summary || "Benchmark needed.")}</p>
      </div>
    `).join("");
  element.innerHTML = tagCards + missingRows;
}

function renderBenchmarkHistory(history) {
  const element = $("#benchmark-history");
  if (!element) return;
  if (!history || !history.run_count) {
    element.innerHTML = `<p class="muted">Run measured benchmarks to build history.</p>`;
    return;
  }
  element.innerHTML = (history.labels || []).map((item) => `
    <div class="benchmark-card">
      <div class="benchmark-head"><strong>${escapeText(item.label)}</strong><span class="time">${escapeText(item.count)} runs</span></div>
      <div class="mini-metrics">
        <span>best ${formatSeconds(item.best_seconds)}</span>
        <span>avg ${formatSeconds(item.average_seconds)}</span>
        <span>worst ${formatSeconds(item.worst_seconds)}</span>
        ${item.average_stack_seconds ? `<span>stack avg ${formatSeconds(item.average_stack_seconds)}</span>` : ""}
        ${item.average_tensor_load_seconds ? `<span>loads avg ${formatSeconds(item.average_tensor_load_seconds)}</span>` : ""}
        ${item.average_decode_tail_seconds ? `<span>tail avg ${formatSeconds(item.average_decode_tail_seconds)}</span>` : ""}
      </div>
    </div>
  `).join("") || `<p class="muted">No complete benchmark history yet.</p>`;
}

function renderMessages() {
  const history = $("#chat-history");
  if (!state.messages.length) {
    history.innerHTML = `
      <div class="empty-state">
        <div class="logo large">P</div>
        <h2>What can I help with?</h2>
        <p>Running locally through Pocket LLM. Quality is default; Quick uses the full model for shorter replies.</p>
        <div class="prompt-grid">
          <button data-prompt="Say hello in one short sentence.">Quick hello</button>
          <button data-prompt="Explain what Pocket LLM can do in plain English.">Explain Pocket LLM</button>
          <button data-prompt="Write a tiny Python function and explain it.">Code check</button>
          <button data-prompt="Give me a short project plan for local AI agents.">Plan an agent</button>
        </div>
      </div>`;
    bindPromptButtons();
    return;
  }
  history.innerHTML = state.messages.map((message) => `
    <div class="message ${message.role}">
      <div class="bubble">${escapeText(message.text)}</div>
    </div>
  `).join("");
  history.scrollTop = history.scrollHeight;
}

function bindPromptButtons() {
  $$("[data-prompt]").forEach((button) => {
    button.addEventListener("click", () => {
      $("#prompt-input").value = button.dataset.prompt;
      $("#prompt-input").focus();
    });
  });
}

function chatHistoryPayload() {
  return state.messages
    .filter((message) => message.text && !message.transient)
    .slice(-8)
    .map((message) => ({
      role: message.role === "ai" ? "assistant" : message.role,
      text: message.text,
    }));
}

function sleep(ms) {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

function setChatControlsRunning(running) {
  $("#send-button").disabled = running;
  $("#cancel-button").hidden = !running;
  $("#retry-button").hidden = running || !state.lastRequest;
}

function updateRunningStatus(status = "running") {
  const elapsed = state.activeStartedAt ? Math.round((Date.now() - state.activeStartedAt) / 1000) : 0;
  if (state.cancelRequested && !["canceled", "failed"].includes(status)) {
    $("#chat-status").textContent = `Cancel requested - ${elapsed}s`;
    return;
  }
  const labels = {
    queued: "Queued locally",
    running: `${state.mode} running`,
    canceled: "Canceled",
    failed: "Chat failed",
  };
  $("#chat-status").textContent = `${labels[status] || labels.running} - ${elapsed}s`;
}

function startRunningTimer() {
  stopRunningTimer();
  state.activeStartedAt = Date.now();
  updateRunningStatus("queued");
  state.activeTimer = window.setInterval(() => updateRunningStatus("running"), 1000);
}

function stopRunningTimer() {
  if (state.activeTimer) {
    window.clearInterval(state.activeTimer);
    state.activeTimer = null;
  }
}

async function pollChatJob(jobId) {
  while (state.activeJobId === jobId) {
    await sleep(1600);
    const job = await api(`/api/chat/status?job_id=${encodeURIComponent(jobId)}`);
    if (state.activeJobId !== jobId) return;
    state.cancelRequested = Boolean(job.cancel_requested);
    updateRunningStatus(job.status);
    if (job.status === "completed") {
      finishCompletedChatJob(job);
      return;
    }
    if (job.status === "failed") {
      renderRuntimePanel(null);
      state.messages[state.messages.length - 1] = { role: "ai", text: `Chat failed: ${job.error || "unknown error"}` };
      $("#chat-status").textContent = "Chat failed";
      return;
    }
    if (job.status === "canceled") {
      renderRuntimePanel(null);
      if (state.messages[state.messages.length - 1]?.transient) {
        state.messages.pop();
      }
      $("#chat-status").textContent = "Canceled - ready to retry";
      return;
    }
  }
}

function finishCompletedChatJob(job) {
  const result = job.result || {};
  renderRuntimePanel(result);
  state.messages[state.messages.length - 1] = {
    role: "ai",
    text: result.generated_text || "(no generated text)",
  };
  const reuse = result.response_reuse?.hit ? " cached" : "";
  $("#chat-status").textContent = result.ready ? `Ready${reuse} - ${result.elapsed_seconds}s` : "Blocked";
}

async function sendPrompt(event) {
  event.preventDefault();
  const input = $("#prompt-input");
  const prompt = input.value.trim();
  if (!prompt) return;
  const modelId = state.status?.active_model?.model_id || "qwen2.5-14b-instruct";
  const maxNewTokens = Number($("#max-new-tokens").value || (state.mode === "GGUF" ? 4096 : 4));
  const historyPayload = chatHistoryPayload();
  const requestPayload = {
    model_id: modelId,
    prompt,
    mode: state.mode,
    max_new_tokens: maxNewTokens,
    profile_id: state.activeProfileId,
    session_id: state.sessionId,
    messages: historyPayload,
  };
  state.lastRequest = requestPayload;
  state.messages.push({ role: "user", text: prompt });
  state.messages.push({ role: "ai", text: "Starting local run...", transient: true });
  input.value = "";
  setChatControlsRunning(true);
  state.cancelRequested = false;
  renderRuntimePanel(null);
  startRunningTimer();
  renderMessages();
  try {
    const job = await api("/api/chat/start", {
      method: "POST",
      body: JSON.stringify(requestPayload),
    });
    state.activeJobId = job.job_id;
    if (job.status === "completed") {
      finishCompletedChatJob(job);
    } else {
      await pollChatJob(job.job_id);
    }
  } catch (error) {
    state.messages[state.messages.length - 1] = { role: "ai", text: `Chat failed: ${error.message}` };
    $("#chat-status").textContent = "Chat failed";
  } finally {
    stopRunningTimer();
    state.activeJobId = null;
    state.cancelRequested = false;
    setChatControlsRunning(false);
    renderMessages();
  }
}

async function cancelChat() {
  if (!state.activeJobId) return;
  $("#cancel-button").disabled = true;
  state.cancelRequested = true;
  $("#chat-status").textContent = "Cancel requested - finishing current local step";
  try {
    await api("/api/chat/cancel", {
      method: "POST",
      body: JSON.stringify({ job_id: state.activeJobId }),
    });
  } catch (error) {
    $("#chat-status").textContent = `Cancel failed: ${error.message}`;
  } finally {
    $("#cancel-button").disabled = false;
  }
}

async function retryLastPrompt() {
  if (!state.lastRequest || state.activeJobId) return;
  $("#prompt-input").value = state.lastRequest.prompt;
  $("#prompt-input").focus();
}

async function runBenchmark() {
  const button = $("#run-benchmark");
  const modelId = state.status?.active_model?.model_id || "qwen2.5-14b-instruct";
  button.disabled = true;
  button.textContent = "Running...";
  $("#benchmark-status").textContent = "Running full benchmark locally. This can take a few minutes.";
  try {
    const run = await api("/api/benchmark", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
    const status = await api("/api/status");
    renderStatus(status);
  } catch (error) {
    $("#benchmark-status").textContent = `Benchmark failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Run full";
  }
}

async function runGgufBenchmark() {
  const button = $("#run-gguf-benchmark");
  const modelId = state.status?.active_model?.model_id || "qwen2.5-14b-instruct";
  button.disabled = true;
  button.textContent = "Running...";
  $("#benchmark-status").textContent = "Running GGUF instruction, logic, and agent checks.";
  try {
    await api("/api/benchmark/gguf", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
    renderStatus(await api("/api/status"));
  } catch (error) {
    $("#benchmark-status").textContent = `GGUF benchmark failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Run GGUF";
  }
}

async function updateRuntimePreset(event) {
  const preset = $("#tensor-cache-preset")?.value || "standard";
  const warmRunner = $("#agent-warm-runner")?.value || "off";
  event.target.disabled = true;
  const cacheStatus = $("#tensor-cache-preset-status");
  const warmStatus = $("#agent-warm-runner-status");
  if (cacheStatus) cacheStatus.textContent = "Saving...";
  if (warmStatus) warmStatus.textContent = "Saving...";
  try {
    await api("/api/settings/runtime", {
      method: "POST",
      body: JSON.stringify({ tensor_cache_preset: preset, agent_warm_runner: warmRunner }),
    });
    renderStatus(await api("/api/status"));
  } catch (error) {
    if (cacheStatus) cacheStatus.textContent = `Save failed: ${error.message}`;
    if (warmStatus) warmStatus.textContent = `Save failed: ${error.message}`;
  } finally {
    event.target.disabled = false;
  }
}

async function runComparisonBenchmark() {
  const button = $("#run-comparison-benchmark");
  const modelId = state.status?.active_model?.model_id || "qwen2.5-14b-instruct";
  button.disabled = true;
  button.textContent = "Running...";
  $("#benchmark-status").textContent = "Running GGUF vs Direct Standard vs Direct Boosted.";
  try {
    await api("/api/benchmark/comparison", {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    });
    renderStatus(await api("/api/status"));
  } catch (error) {
    $("#benchmark-status").textContent = `Comparison failed: ${error.message}`;
  } finally {
    button.disabled = false;
    button.textContent = "Run comparison";
  }
}

async function controlWarmRunner(action) {
  const modelId = state.status?.active_model?.model_id || "qwen2.5-14b-instruct";
  const startButton = $("#warm-runner-start");
  const stopButton = $("#warm-runner-stop");
  if (startButton) startButton.disabled = true;
  if (stopButton) stopButton.disabled = true;
  const status = $("#agent-warm-runner-status");
  if (status) status.textContent = action === "start" ? "Starting Agent runner..." : "Stopping Agent runner...";
  try {
    await api("/api/warm-runner", {
      method: "POST",
      body: JSON.stringify({ action, model_id: modelId }),
    });
    renderStatus(await api("/api/status"));
  } catch (error) {
    if (status) status.textContent = `Agent runner action failed: ${error.message}`;
  } finally {
    if (startButton) startButton.disabled = false;
    if (stopButton) stopButton.disabled = false;
  }
}

async function controlGgufServer(action) {
  const modelId = state.status?.active_model?.model_id || "qwen2.5-14b-instruct";
  const toggleButton = $("#gguf-toggle-button");
  if (toggleButton) toggleButton.disabled = true;
  const label = action === "start" ? "Loading GGUF server..." : "Unloading GGUF server...";
  const card = $("#gguf-server-card");
  if (card) {
    card.insertAdjacentHTML("afterbegin", `<p class="muted" id="gguf-server-working">${label}</p>`);
  }
  try {
    await api("/api/gguf/server", {
      method: "POST",
      body: JSON.stringify({ action, model_id: modelId }),
    });
    renderStatus(await api("/api/status"));
  } catch (error) {
    const working = $("#gguf-server-working");
    if (working) working.textContent = `GGUF action failed: ${error.message}`;
    if (toggleButton) toggleButton.disabled = false;
  }
}

async function boot() {
  runLaunchSequence();
  $$(".nav-item").forEach((item) => item.addEventListener("click", () => setScreen(item.dataset.screen)));
  $$("#mode-picker button").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));
  $("#chat-form").addEventListener("submit", sendPrompt);
  $("#cancel-button").addEventListener("click", cancelChat);
  $("#chat-load-gguf").addEventListener("click", () => controlGgufServer("start"));
  $("#retry-button").addEventListener("click", retryLastPrompt);
  $("#run-benchmark").addEventListener("click", runBenchmark);
  $("#run-comparison-benchmark").addEventListener("click", runComparisonBenchmark);
  $("#run-gguf-benchmark").addEventListener("click", runGgufBenchmark);
  $("#tensor-cache-preset").addEventListener("change", updateRuntimePreset);
  $("#agent-warm-runner").addEventListener("change", updateRuntimePreset);
  $("#warm-runner-start").addEventListener("click", () => controlWarmRunner("start"));
  $("#warm-runner-stop").addEventListener("click", () => controlWarmRunner("stop"));
  $("#supervisor-check").addEventListener("click", runSupervisorCheck);
  $("#profile-select").addEventListener("change", (event) => {
    state.activeProfileId = event.target.value;
    const profile = (state.status?.profiles || []).find((item) => item.profile_id === state.activeProfileId);
    if (!profile) return;
    if (profile.runtime_mode) setMode(profile.runtime_mode);
    const defaultTokens = state.mode === "GGUF" ? (profile.settings?.gguf_max_new_tokens || 4096) : profile.settings?.default_max_new_tokens;
    if (defaultTokens) $("#max-new-tokens").value = defaultTokens;
  });
  setChatControlsRunning(false);
  bindPromptButtons();
  setMode("GGUF");
  try {
    renderStatus(await api("/api/status"));
    renderSupervisor(await api("/api/supervisor/health"));
    state.statusRefreshTimer = window.setInterval(async () => {
      if (!state.status?.downloads?.active_count) return;
      try {
        renderStatus(await api("/api/status"));
      } catch (error) {
        $("#sidebar-runtime").textContent = error.message;
      }
    }, 5000);
  } catch (error) {
    $("#sidebar-model").textContent = "Status failed";
    $("#sidebar-runtime").textContent = error.message;
  }
}

boot();
