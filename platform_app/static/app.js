const state = {
  status: null,
  selectedPhaseRun: "",
  selectedAcquisition: "",
  lastPrediction: null,
  checkpointInfoRequest: 0,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function fileUrl(path) {
  return `/api/file?path=${encodeURIComponent(path)}`;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "-";
  const number = Number(value);
  if (Math.abs(number) >= 1000 || (Math.abs(number) < 0.01 && number !== 0)) {
    return number.toExponential(2);
  }
  return number.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatRange(range) {
  if (!range) return "-";
  return `${formatNumber(range.min)}-${formatNumber(range.max)} nm`;
}

function pathLabel(path) {
  return String(path || "").split("/").slice(-3).join("/");
}

async function fetchJson(url) {
  const response = await fetch(url);
  const payload = await response.json();
  if (!response.ok || payload.error) {
    throw new Error(payload.error || `HTTP ${response.status}`);
  }
  return payload;
}

function metric(label, value) {
  return `
    <div class="metric-card">
      <div class="label">${escapeHtml(label)}</div>
      <div class="value">${escapeHtml(value)}</div>
    </div>
  `;
}

function renderBars(target, counts) {
  const entries = Object.entries(counts || {});
  if (!entries.length) {
    target.innerHTML = `<div class="empty-state">No values</div>`;
    return;
  }
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  target.innerHTML = entries.map(([label, value], index) => {
    const colors = ["var(--teal)", "var(--steel)", "var(--amber)"];
    const width = Math.max(4, (Number(value) / max) * 100);
    return `
      <div class="bar-row">
        <span title="${escapeHtml(label)}">${escapeHtml(label)}</span>
        <span class="bar-shell">
          <span class="bar-fill" style="width:${width}%; background:${colors[index % colors.length]}"></span>
        </span>
        <strong>${escapeHtml(value)}</strong>
      </div>
    `;
  }).join("");
}

function formatCell(value) {
  if (value === null || value === undefined) return "";
  if (typeof value === "number") return formatNumber(value, 4);
  const parsed = Number(value);
  if (String(value).trim() !== "" && !Number.isNaN(parsed) && String(value).length < 16) {
    return formatNumber(parsed, 4);
  }
  return escapeHtml(value);
}

function renderTable(target, rows, columns = null) {
  if (!rows || !rows.length) {
    target.innerHTML = `<div class="empty-state">No rows</div>`;
    return;
  }
  const cols = columns || Object.keys(rows[0]);
  target.innerHTML = `
    <table>
      <thead><tr>${cols.map((col) => `<th>${escapeHtml(col)}</th>`).join("")}</tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr>${cols.map((col) => `<td>${formatCell(row[col])}</td>`).join("")}</tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function checkpointRuns() {
  return (state.status?.runs || []).filter((run) => run.checkpoint);
}

function runLabel(run) {
  return pathLabel(typeof run === "string" ? run : run?.path || "");
}

function phaseImageRank(path) {
  const lower = String(path || "").toLowerCase();
  if (lower.includes("phase_dataset") && lower.includes("abs")) return 0;
  if (lower.includes("phase_dataset") && lower.includes("signed")) return 1;
  if (lower.includes("phase_model") && lower.includes("abs")) return 2;
  if (lower.includes("phase_model") && lower.includes("signed")) return 3;
  if (lower.includes("abs")) return 4;
  if (lower.includes("signed")) return 5;
  return 9;
}

function phaseImageTitle(path) {
  const lower = String(path || "").toLowerCase();
  const source = lower.includes("phase_model") ? "Model" : lower.includes("phase_dataset") ? "Dataset" : "Phase";
  const metric = lower.includes("signed") ? "MeanMz signed" : lower.includes("abs") ? "|MeanMz|" : "map";
  return `${source} ${metric}`;
}

function sortedPhaseImages(run) {
  return [...(run?.phase_images || [])].sort((a, b) => phaseImageRank(a) - phaseImageRank(b) || a.localeCompare(b));
}

function phaseRunScore(run) {
  const lower = String(run?.path || "").toLowerCase();
  let score = 0;
  if (lower.includes("param_surrogate_h100_masked")) score += 1000000;
  else if (lower.includes("param_surrogate_h100")) score += 650000;
  if (lower.includes("v1_pipeline")) score += 50000;
  if (lower.includes("active_learning")) score += 10000;
  if (lower.includes("smoke")) score -= 200000;
  const images = run?.phase_images || [];
  if (images.some((path) => path.includes("phase_dataset") && path.includes("abs"))) score += 2000;
  if (images.some((path) => path.includes("phase_model") && path.includes("abs"))) score += 1000;
  score += images.length * 10;
  return score;
}

function recommendedPhaseRun() {
  const runs = phaseRuns();
  if (!runs.length) return null;
  return [...runs].sort((a, b) => phaseRunScore(b) - phaseRunScore(a))[0];
}

function preferredReconstructionImages(run, limit = 6) {
  const images = run?.reconstruction_images || [];
  const components = images.filter((path) => String(path).toLowerCase().includes("components"));
  const rest = images.filter((path) => !String(path).toLowerCase().includes("components"));
  return [...components, ...rest].slice(0, limit);
}

function reconstructionRun() {
  const active = activeCheckpointRun();
  if (preferredReconstructionImages(active, 1).length) return active;
  const phaseRun = phaseRuns().find((run) => run.path === state.selectedPhaseRun);
  if (preferredReconstructionImages(phaseRun, 1).length) return phaseRun;
  return checkpointRuns().find((run) => preferredReconstructionImages(run, 1).length) || null;
}

function phaseRuns() {
  return (state.status?.runs || []).filter((run) => run.phase_images && run.phase_images.length);
}

function acquisitionFiles() {
  return (state.status?.runs || []).flatMap((run) => {
    return (run.acquisitions || []).map((path) => ({ run: run.path, path }));
  });
}

function activeCheckpointRun() {
  const checkpoint = $("checkpointSelect").value;
  return checkpointRuns().find((run) => run.checkpoint === checkpoint) || null;
}

function checkpointRange(run = activeCheckpointRun()) {
  const ranges = run?.checkpoint_info?.normalizer_range_nm;
  if (ranges?.Tx && ranges?.Tz) {
    return ranges;
  }
  const dataset = state.status?.dataset;
  if (dataset?.available) {
    return {
      Tx: dataset.tx_nm,
      Tz: dataset.tz_nm,
    };
  }
  return null;
}

function checkpointScore(run) {
  const range = checkpointRange(run);
  const area = range ? (range.Tx.max - range.Tx.min) * (range.Tz.max - range.Tz.min) : 0;
  const path = run.path || "";
  return (
    area +
    (path.includes("masked") ? 1000000 : 0) +
    (path.includes("h100") ? 100000 : 0) -
    (path.includes("smoke") ? 100000 : 0)
  );
}

function recommendedCheckpoint() {
  const runs = checkpointRuns();
  if (!runs.length) return null;
  return [...runs].sort((a, b) => checkpointScore(b) - checkpointScore(a))[0];
}

function setParamValues(tx, tz) {
  $("txInput").value = formatNumber(tx, 3);
  $("tzInput").value = formatNumber(tz, 3);
  $("txSlider").value = tx;
  $("tzSlider").value = tz;
  updateEnvelopePanels();
}

function clampToRange(value, range) {
  const numeric = Number(value);
  if (!range || !Number.isFinite(numeric)) return numeric;
  return Math.min(range.max, Math.max(range.min, numeric));
}

function setRangeControls(centerIfUnset = false) {
  const ranges = checkpointRange();
  if (!ranges) return;
  [
    ["tx", "Tx"],
    ["tz", "Tz"],
  ].forEach(([prefix, key]) => {
    const slider = $(`${prefix}Slider`);
    slider.min = ranges[key].min;
    slider.max = ranges[key].max;
    slider.step = 0.1;
  });

  const txCurrent = Number($("txInput").value);
  const tzCurrent = Number($("tzInput").value);
  const tx = centerIfUnset || !Number.isFinite(txCurrent)
    ? (ranges.Tx.min + ranges.Tx.max) / 2
    : clampToRange(txCurrent, ranges.Tx);
  const tz = centerIfUnset || !Number.isFinite(tzCurrent)
    ? (ranges.Tz.min + ranges.Tz.max) / 2
    : clampToRange(tzCurrent, ranges.Tz);
  setParamValues(tx, tz);
  renderPresets();
}

function selectedParamsInRange() {
  const ranges = checkpointRange();
  if (!ranges) return true;
  const tx = Number($("txInput").value);
  const tz = Number($("tzInput").value);
  return tx >= ranges.Tx.min && tx <= ranges.Tx.max && tz >= ranges.Tz.min && tz <= ranges.Tz.max;
}

function renderCheckpointRange() {
  const run = activeCheckpointRun();
  const checkpointRanges = run?.checkpoint_info?.normalizer_range_nm;
  const ranges = checkpointRanges || checkpointRange(run);
  const target = $("checkpointRange");
  if (!run || !ranges) {
    target.className = "range-card empty";
    target.innerHTML = "No checkpoint range available";
    return;
  }
  const source = checkpointRanges ? "envelope" : "coverage";
  const decoder = run.checkpoint_info?.model_config?.spatial_size || (run.checkpoint_info === null ? "loading" : 200);
  target.className = "range-card";
  target.innerHTML = `
    <div class="range-row"><span>Tx ${source}</span><strong>${formatRange(ranges.Tx)}</strong></div>
    <div class="range-row"><span>Tz ${source}</span><strong>${formatRange(ranges.Tz)}</strong></div>
    <div class="range-row"><span>Decoder</span><strong>${escapeHtml(decoder)}${Number.isFinite(Number(decoder)) ? " px" : ""}</strong></div>
  `;
}

function renderPresets() {
  const ranges = checkpointRange();
  const target = $("presetList");
  if (!ranges) {
    target.innerHTML = "";
    return;
  }
  const txMin = ranges.Tx.min;
  const txMax = ranges.Tx.max;
  const tzMin = ranges.Tz.min;
  const tzMax = ranges.Tz.max;
  const presets = [
    { label: "Interior", tx: (txMin + txMax) / 2, tz: (tzMin + tzMax) / 2 },
    { label: "Thin Edge", tx: txMin + 0.12 * (txMax - txMin), tz: tzMin + 0.12 * (tzMax - tzMin) },
    { label: "High Tx", tx: txMax - 0.08 * (txMax - txMin), tz: (tzMin + tzMax) / 2 },
    { label: "High Tz", tx: (txMin + txMax) / 2, tz: tzMax - 0.08 * (tzMax - tzMin) },
  ];
  target.innerHTML = presets.map((preset) => `
    <button class="preset-button" data-tx="${preset.tx}" data-tz="${preset.tz}">
      ${escapeHtml(preset.label)}
    </button>
  `).join("");
  target.querySelectorAll("button").forEach((button) => {
    button.addEventListener("click", () => {
      setParamValues(Number(button.dataset.tx), Number(button.dataset.tz));
    });
  });
}

function renderOverview() {
  const dataset = state.status.dataset;
  const badge = $("datasetBadge");
  if (!dataset.available) {
    badge.textContent = "Dataset missing";
    badge.className = "status-pill warn";
    $("metricsGrid").innerHTML = `<div class="empty-state">data/dataset/meta.h5 is missing</div>`;
    return;
  }

  badge.textContent = `${dataset.samples} samples`;
  badge.className = "status-pill ok";
  $("metricsGrid").innerHTML = [
    metric("Samples", dataset.samples),
    metric("Train", dataset.split_counts?.train ?? "-"),
    metric("Tx Range", formatRange(dataset.tx_nm)),
    metric("Tz Range", formatRange(dataset.tz_nm)),
  ].join("");
  renderBars($("splitBars"), dataset.split_counts);
  renderBars($("stateBars"), dataset.state_counts);
  renderTable($("recentSamples"), dataset.recent, ["split", "State", "Tx_nm", "Tz_nm", "simulation_id"]);
}

function renderDemoReadiness() {
  const dataset = state.status?.dataset;
  const checkpoints = checkpointRuns();
  const ready = dataset?.available && dataset.fields_exists && checkpoints.length > 0;
  $("demoReadiness").textContent = ready ? "Demo ready" : "Needs artifacts";
  $("demoReadiness").className = `status-pill ${ready ? "ok" : "warn"}`;

  const splits = dataset?.split_counts || {};
  const holdout = Number(splits.test_holdout || 0) + Number(splits.boundary_holdout || 0);
  const ranges = checkpointRange();
  const acquisitions = acquisitionFiles();
  $("handoffBadge").textContent = acquisitions.length ? "AL dry-run" : "No AL file";
  $("handoffBadge").className = `status-pill ${acquisitions.length ? "ok" : "warn"}`;
  $("snapshotDataset").textContent = dataset?.available ? `${dataset.samples} samples` : "missing";
  $("snapshotSplits").textContent = splits.train !== undefined
    ? `train ${splits.train} / val ${splits.val ?? 0} / test ${splits.test_holdout ?? 0}`
    : "splits pending";
  $("snapshotHoldout").textContent = holdout ? `${holdout} samples` : "-";
  $("snapshotEnvelope").textContent = ranges ? `${formatRange(ranges.Tx)} / ${formatRange(ranges.Tz)}` : "pending";
  $("snapshotAcquisition").textContent = state.selectedAcquisition ? runLabel(state.selectedAcquisition) : "none";
}

function renderSelectors() {
  const checkpoints = checkpointRuns();
  const currentCheckpoint = $("checkpointSelect").value;
  const recommended = checkpoints.find((run) => run.checkpoint === currentCheckpoint) || recommendedCheckpoint();
  $("checkpointSelect").innerHTML = checkpoints.length
    ? checkpoints.map((run) => {
        const selected = recommended && run.checkpoint === recommended.checkpoint ? " selected" : "";
        const summary = run.summary?.mse_mean !== undefined ? ` · mse ${formatNumber(run.summary.mse_mean, 4)}` : "";
        return `<option value="${escapeHtml(run.checkpoint)}"${selected}>${escapeHtml(runLabel(run) + summary)}</option>`;
      }).join("")
    : `<option value="">No checkpoint found</option>`;

  const phases = phaseRuns();
  const selectedPhase = phases.find((run) => run.path === state.selectedPhaseRun) || recommendedPhaseRun();
  const phaseOptions = phases.length
    ? phases.map((run) => {
        const selected = selectedPhase && run.path === selectedPhase.path ? " selected" : "";
        return `<option value="${escapeHtml(run.path)}"${selected}>${escapeHtml(runLabel(run))}</option>`;
      }).join("")
    : `<option value="">No phase plots found</option>`;
  $("phaseRunSelect").innerHTML = phaseOptions;
  $("phaseRunSelectFull").innerHTML = phaseOptions;
  state.selectedPhaseRun = selectedPhase?.path || $("phaseRunSelect").value;

  const acquisitions = acquisitionFiles();
  const selectedAcquisition = acquisitions.find((item) => item.path === state.selectedAcquisition) || acquisitions[0];
  $("acquisitionSelect").innerHTML = acquisitions.length
    ? acquisitions.map((item) => {
        const selected = selectedAcquisition && item.path === selectedAcquisition.path ? " selected" : "";
        return `<option value="${escapeHtml(item.path)}"${selected}>${escapeHtml(runLabel(item.path))}</option>`;
      }).join("")
    : `<option value="">No acquisition CSV found</option>`;
  state.selectedAcquisition = selectedAcquisition?.path || $("acquisitionSelect").value;

  setRangeControls(true);
  renderCheckpointRange();
  renderDemoReadiness();
}

function phasePreviewPath(run) {
  const images = sortedPhaseImages(run);
  if (!images.length) return null;
  return images.find((path) => path.includes("phase_dataset") && path.includes("abs")) || images[0];
}

function renderPhase() {
  const run = phaseRuns().find((item) => item.path === state.selectedPhaseRun);
  if (!run) {
    $("phaseGallery").innerHTML = `<div class="empty-state">No phase diagrams found under results</div>`;
    $("demoPhasePreview").innerHTML = `<div class="empty-state">No phase map</div>`;
    return;
  }

  $("phaseGallery").innerHTML = sortedPhaseImages(run).map((path) => `
    <figure class="image-tile">
      <img src="${fileUrl(path)}" alt="${escapeHtml(path)}">
      <figcaption class="phase-caption"><strong>${escapeHtml(phaseImageTitle(path))}</strong>${escapeHtml(path)}</figcaption>
    </figure>
  `).join("");

  const preview = phasePreviewPath(run);
  $("demoPhasePreview").innerHTML = preview
    ? `<img src="${fileUrl(preview)}" alt="${escapeHtml(preview)}">`
    : `<div class="empty-state">No phase map</div>`;
}

function parseCsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const headers = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const values = line.split(",");
    const row = {};
    headers.forEach((header, index) => {
      row[header] = values[index] ?? "";
    });
    return row;
  });
}

async function renderAcquisition() {
  const path = state.selectedAcquisition;
  if (!path) {
    $("acquisitionSummary").innerHTML = "";
    $("acquisitionTable").innerHTML = `<div class="empty-state">No active-learning acquisitions found</div>`;
    return;
  }
  const response = await fetch(fileUrl(path));
  const text = await response.text();
  const rows = parseCsv(text);
  const txValues = rows.map((row) => Number(row.Tx_nm)).filter(Number.isFinite);
  const tzValues = rows.map((row) => Number(row.Tz_nm)).filter(Number.isFinite);
  $("acquisitionSummary").innerHTML = [
    metric("Selected", rows.length),
    metric("Tx Span", txValues.length ? `${formatNumber(Math.min(...txValues))}-${formatNumber(Math.max(...txValues))} nm` : "-"),
    metric("Tz Span", tzValues.length ? `${formatNumber(Math.min(...tzValues))}-${formatNumber(Math.max(...tzValues))} nm` : "-"),
  ].join("");
  renderTable($("acquisitionTable"), rows);
}

function renderRuns() {
  const runs = state.status.runs || [];
  $("runCount").textContent = `${runs.length} runs`;
  if (!runs.length) {
    $("runList").innerHTML = `<div class="empty-state">No result runs found</div>`;
    return;
  }
  $("runList").innerHTML = runs.map((run) => {
    const summary = run.summary || {};
    const ranges = run.checkpoint_info?.normalizer_range_nm;
    const chips = [
      run.checkpoint ? `<span class="chip green">checkpoint</span>` : "",
      ranges ? `<span class="chip">Tx ${formatRange(ranges.Tx)}</span>` : "",
      run.phase_images?.length ? `<span class="chip">phase ${run.phase_images.length}</span>` : "",
      run.reconstruction_images?.length ? `<span class="chip">recon ${run.reconstruction_images.length}</span>` : "",
      run.acquisitions?.length ? `<span class="chip amber">AL ${run.acquisitions.length}</span>` : "",
      summary.mse_mean !== undefined ? `<span class="chip">mse ${formatNumber(summary.mse_mean, 4)}</span>` : "",
    ].filter(Boolean).join("");
    return `
      <article class="run-item">
        <div class="run-top">
          <div class="run-path">${escapeHtml(run.path)}</div>
          <div class="run-meta">${chips}</div>
        </div>
        <div class="muted-line">${escapeHtml(run.checkpoint || run.kind || "result")}</div>
      </article>
    `;
  }).join("");
}

function renderPrediction(payload) {
  state.lastPrediction = payload;
  const imgSrc = `data:image/png;base64,${payload.image_png_base64}`;
  $("predictionImage").src = imgSrc;
  $("predictionImageDetail").src = imgSrc;
  $("predictionEmpty").textContent = "";
  $("predictionDetailEmpty").textContent = "";

  const imageMode = payload.image_mode === "components" ? "Mx/My/Mz" : (payload.image_mode || "field");
  const label = `${formatNumber(payload.tx_nm)} nm / ${formatNumber(payload.tz_nm)} nm · ${imageMode} · ${payload.device}`;
  $("predictionLabel").textContent = label;
  $("predictionDetailLabel").textContent = label;

  $("stateGuess").textContent = payload.state_guess || "Unclassified";
  $("stateGuess").className = "status-pill ok";

  $("predictionMetrics").innerHTML = Object.entries(payload.metrics).map(([key, value]) => `
    <div class="metric-row"><span>${escapeHtml(key)}</span><strong>${formatNumber(value, 5)}</strong></div>
  `).join("");

  $("predictionWarnings").innerHTML = (payload.warnings || []).map((warning) => `
    <div class="warning">${escapeHtml(warning)}</div>
  `).join("");

  updateEnvelopePanels();
}

function renderReconstructions() {
  const target = $("reconstructionGallery");
  if (!target) return;
  const run = reconstructionRun();
  const images = preferredReconstructionImages(run);
  if (!images.length) {
    target.innerHTML = `<div class="empty-state">No reconstruction images found</div>`;
    return;
  }
  target.innerHTML = images.map((path) => `
    <figure>
      <img src="${fileUrl(path)}" alt="${escapeHtml(path)}">
      <figcaption>${escapeHtml(path)}</figcaption>
    </figure>
  `).join("");
}

function updateEnvelopePanels() {
  renderCheckpointRange();
  const ranges = checkpointRange();
  const run = activeCheckpointRun();
  const inside = selectedParamsInRange();
  const tx = Number($("txInput").value);
  const tz = Number($("tzInput").value);
  $("predictionEnvelope").innerHTML = ranges ? [
    `<div class="metric-row"><span>Checkpoint</span><strong>${escapeHtml(pathLabel(run?.path || "-"))}</strong></div>`,
    `<div class="metric-row"><span>Tx envelope</span><strong>${formatRange(ranges.Tx)}</strong></div>`,
    `<div class="metric-row"><span>Tz envelope</span><strong>${formatRange(ranges.Tz)}</strong></div>`,
    `<div class="metric-row"><span>Selected Tx</span><strong>${formatNumber(tx)} nm</strong></div>`,
    `<div class="metric-row"><span>Selected Tz</span><strong>${formatNumber(tz)} nm</strong></div>`,
    `<div class="metric-row"><span>Range status</span><strong>${inside ? "inside" : "outside"}</strong></div>`,
  ].join("") : `<div class="empty-state">No checkpoint envelope</div>`;
  renderDemoReadiness();
  renderReconstructions();
}

async function loadCheckpointInfoForSelected() {
  const run = activeCheckpointRun();
  if (!run?.checkpoint || run.checkpoint_info) return;
  const requestId = ++state.checkpointInfoRequest;
  renderCheckpointRange();
  try {
    const info = await fetchJson(`/api/checkpoint-info?checkpoint=${encodeURIComponent(run.checkpoint)}`);
    if (requestId !== state.checkpointInfoRequest) return;
    run.checkpoint_info = info;
    setRangeControls(false);
    updateEnvelopePanels();
  } catch (error) {
    if (requestId !== state.checkpointInfoRequest) return;
    run.checkpoint_info = { error: error.message };
    $("checkpointRange").className = "range-card empty";
    $("checkpointRange").innerHTML = escapeHtml(error.message);
  }
}

async function runPrediction() {
  const checkpoint = $("checkpointSelect").value;
  if (!checkpoint) return;
  const params = new URLSearchParams({
    checkpoint,
    tx_nm: $("txInput").value || "50",
    tz_nm: $("tzInput").value || "50",
    device: $("deviceSelect").value,
  });
  $("predictButton").disabled = true;
  $("predictStatus").textContent = "Generating";
  try {
    const payload = await fetchJson(`/api/predict?${params.toString()}`);
    renderPrediction(payload);
    $("predictStatus").textContent = payload.warnings?.length ? "Prediction with warning" : "Prediction ready";
    showView("demo");
  } catch (error) {
    $("predictStatus").textContent = error.message;
  } finally {
    $("predictButton").disabled = false;
  }
}

function showView(viewId) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.view === viewId);
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.classList.toggle("is-active", view.id === viewId);
  });
}

async function refresh() {
  $("datasetBadge").textContent = "Loading";
  state.status = await fetchJson("/api/status");
  renderOverview();
  renderSelectors();
  renderPhase();
  renderRuns();
  await renderAcquisition();
  updateEnvelopePanels();
  loadCheckpointInfoForSelected().catch((error) => {
    $("predictStatus").textContent = error.message;
  });
}

function bindParamControls() {
  [
    ["txSlider", "txInput"],
    ["tzSlider", "tzInput"],
  ].forEach(([sliderId, inputId]) => {
    const slider = $(sliderId);
    const input = $(inputId);
    slider.addEventListener("input", () => {
      input.value = formatNumber(Number(slider.value), 3);
      updateEnvelopePanels();
    });
    input.addEventListener("input", () => {
      slider.value = input.value;
      updateEnvelopePanels();
    });
  });
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => showView(button.dataset.view));
});

$("refreshButton").addEventListener("click", () => refresh().catch((error) => {
  $("datasetBadge").textContent = error.message;
  $("datasetBadge").className = "status-pill warn";
}));

$("checkpointSelect").addEventListener("change", () => {
  setRangeControls(true);
  updateEnvelopePanels();
  loadCheckpointInfoForSelected().catch((error) => {
    $("predictStatus").textContent = error.message;
  });
});

$("phaseRunSelect").addEventListener("change", (event) => {
  state.selectedPhaseRun = event.target.value;
  $("phaseRunSelectFull").value = state.selectedPhaseRun;
  renderPhase();
});

$("phaseRunSelectFull").addEventListener("change", (event) => {
  state.selectedPhaseRun = event.target.value;
  $("phaseRunSelect").value = state.selectedPhaseRun;
  renderPhase();
});

$("acquisitionSelect").addEventListener("change", (event) => {
  state.selectedAcquisition = event.target.value;
  renderAcquisition().catch((error) => {
    $("acquisitionTable").innerHTML = `<div class="empty-state">${escapeHtml(error.message)}</div>`;
  });
});

$("predictButton").addEventListener("click", runPrediction);

bindParamControls();
refresh().catch((error) => {
  $("datasetBadge").textContent = error.message;
  $("datasetBadge").className = "status-pill warn";
});
