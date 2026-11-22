"use strict";

// TRACE: same-origin predictions with an immutable original in comparison mode.
const API_URL = "/predict";
const experimentalV3 = new URLSearchParams(window.location.search).get("model") !== "v2";
const MAX_CHARS = 5000;
const MAX_FILE_BYTES = MAX_CHARS * 4 + 3;
const LENGTH_ERROR = "Keep your SMS message to 5,000 characters or fewer.";
const UPLOAD_LENGTH_ERROR = "The uploaded file exceeds 5,000 characters. Choose a shorter .txt file.";
const EMPTY_ERROR = "Enter an SMS message to analyze.";
const SERVER_ERROR = "The prediction service had a problem. Try again.";
const NETWORK_ERROR = "Could not reach the prediction service. Check your connection and try again.";
const HAM_LIMITATION = "A Ham prediction does not guarantee message safety.";
const OUTCOMES = {
  phishing: { title: "Potential Phishing", icon: "!", description: "The experimental model flagged this message. Verify the request through a trusted channel." },
  spam: { title: "Likely Spam", icon: "!", description: "This message contains patterns the model associates with SMS spam." },
  ham: { title: "Likely Ham", icon: "✓", description: "This message contains patterns the model associates with legitimate SMS." },
  inconclusive: { title: "Inconclusive", icon: "?", description: "The message contains no words or phrases the model recognizes." },
};
const INCONCLUSIVE_REASONS = {
  no_letters: "The message has no words to analyze.",
  unsupported_script: "The model only supports English SMS written in the Latin alphabet.",
  no_recognized_features: "The message contains no words or phrases the model recognizes.",
};
const el = (id) => document.getElementById(id);
el("experimentalNotice").hidden = !experimentalV3;
el("baselineNotice").hidden = experimentalV3;
const form = el("smsForm"), smsText = el("smsText"), dropzone = el("dropzone"), fileInput = el("fileInput");
const charCount = el("charCount"), uploadBtn = el("uploadBtn"), clearBtn = el("clearBtn");
const analyzeBtn = el("analyzeBtn"), messageError = el("messageError"), analysisStatus = el("analysisStatus");
const workspace = el("smsWorkspace"), resultPane = el("resultPane"), loadingPanel = el("loadingPanel");
const resultCard = el("resultCard"), resultLabel = el("resultLabel"), resultDesc = el("resultDesc");
const comparisonPanel = el("comparisonPanel"), revisionLabel = el("revisionLabel"), revisionMessage = el("revisionMessage");
const originalMessage = el("originalMessage"), originalLabel = el("originalLabel"), comparisonNote = el("comparisonNote");
const tryVariationBtn = el("tryVariationBtn"), startFreshBtn = el("startFreshBtn");
const stageName = el("stageName"), editorLabel = el("editorLabel");

let pendingPrediction = null;
let readingFile = false;
let inputRevision = 0;
let revisionMode = false;
let original = null;  // {text, prediction}; immutable baseline for a comparison.
let revisionPrediction = null;

function countCharacters(text) { return [...text].length; }
function isBlank(text) { return /^[\p{White_Space}\u001c-\u001f]*$/u.test(text); }

function setStage(stage) {
  workspace.dataset.stage = stage;
  stageName.textContent = ({compose:"COMPOSE",pending:"ANALYZING",result:"RESULT",variation:"REVISION",revisionPending:"ANALYZING REVISION",compared:"COMPARISON"})[stage] || "COMPOSE";
  editorLabel.textContent = revisionMode ? "Your revision" : "Your message";
  smsText.setAttribute("aria-label", `${editorLabel.textContent} (SMS message)`);
  resultPane.inert = stage === "compose";
  resultPane.setAttribute("aria-hidden", String(stage === "compose"));
  loadingPanel.hidden = stage !== "pending";
  resultCard.hidden = stage !== "result";
  comparisonPanel.hidden = !["variation", "revisionPending", "compared"].includes(stage);
}

function updateAnalyzeButton() {
  const busy = readingFile || pendingPrediction !== null;
  analyzeBtn.setAttribute("aria-disabled", String(busy));
  analyzeBtn.textContent = pendingPrediction ? "Analyzing…" : (revisionMode ? "Analyze revision" : "Analyze SMS");
}

function updateCount() {
  const length = countCharacters(smsText.value);
  charCount.textContent = String(length);
  charCount.parentElement.classList.toggle("is-over-limit", length > MAX_CHARS);
}
function clearError() { messageError.textContent = ""; smsText.removeAttribute("aria-invalid"); }
function showError(message, invalidInput = false) {
  messageError.textContent = message;
  analysisStatus.textContent = "";
  if (invalidInput) smsText.setAttribute("aria-invalid", "true");
}
function clearView() {
  clearError();
  analysisStatus.textContent = "";
  loadingPanel.hidden = true;
  resultCard.hidden = true;
  if (!revisionMode) comparisonPanel.hidden = true;
}

function renderComparison() {
  const baseline = original?.prediction;
  originalMessage.textContent = original?.text || "";
  renderVerdict(originalLabel, baseline);
  el("originalDescription").textContent = baseline ? explanationFor(baseline) : "";
  revisionMessage.textContent = smsText.value || "No revision entered";
  renderVerdict(revisionLabel, revisionPrediction, pendingPrediction ? "Analyzing revision…" : "Not analyzed");
  // Show the Ham limitation once per comparison, beside the first applicable verdict.
  const describeRevision = baseline?.label === "ham" ? descriptionFor : explanationFor;
  el("revisionDescription").textContent = revisionPrediction ? describeRevision(revisionPrediction) : "";
  if (revisionPrediction) {
    comparisonNote.textContent = revisionPrediction.label === original.prediction.label
      ? "Classification unchanged. Edited wording does not necessarily change a prediction."
      : "Classification changed. The edited words are not an explanation of the model's reasoning.";
  } else comparisonNote.textContent = "The original prediction is preserved. Analyze the revision to compare actual API results.";
}

function cancelInputWork() {
  inputRevision += 1;
  pendingPrediction?.abort();
  pendingPrediction = null;
  readingFile = false;
  revisionPrediction = null;
}

function invalidateInput() {
  cancelInputWork();
  clearView();
  if (revisionMode && original) {
    renderComparison();
    setStage("variation");
  } else {
    original = null;
    setStage("compose");
  }
  updateAnalyzeButton();
}

smsText.addEventListener("input", () => {
  invalidateInput();
  updateCount();
  if (countCharacters(smsText.value) > MAX_CHARS) showError(LENGTH_ERROR, true);
});

uploadBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) void readTextFile(file);
});
["dragenter", "dragover"].forEach((type) => dropzone.addEventListener(type, (event) => {
  event.preventDefault();
  dropzone.classList.add("drag-over");
}));
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));
dropzone.addEventListener("drop", (event) => {
  event.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = event.dataTransfer?.files?.[0];
  if (file) void readTextFile(file);
});

async function readTextFile(file) {
  invalidateInput();
  if (!file.name.toLowerCase().endsWith(".txt")) { showError("Choose a .txt file to upload."); return; }
  if (file.size > MAX_FILE_BYTES) { showError(UPLOAD_LENGTH_ERROR); return; }
  const revision = inputRevision;
  readingFile = true;
  updateAnalyzeButton();
  analysisStatus.textContent = "Reading SMS file…";
  try {
    const text = (await file.text()).replace(/\r\n?/g, "\n");
    if (revision !== inputRevision) return;
    if (countCharacters(text) > MAX_CHARS) { showError(UPLOAD_LENGTH_ERROR); return; }
    smsText.value = text;
    updateCount();
    if (revisionMode) renderComparison();
    analysisStatus.textContent = "SMS file loaded.";
  } catch {
    if (revision === inputRevision) showError("Could not read the file. Try uploading it again.");
  } finally {
    if (revision === inputRevision) { readingFile = false; updateAnalyzeButton(); }
  }
}

clearBtn.addEventListener("click", () => {
  invalidateInput();
  smsText.value = "";
  updateCount();
  if (revisionMode) renderComparison();
  analysisStatus.textContent = "Message cleared.";
  smsText.focus();
});

async function requestPrediction(text, signal) {
  const response = await fetch(experimentalV3 ? "/predict/v3" : API_URL, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }), signal,
  });
  if (response.status === 422) {
    const payload = await response.json().catch(() => null);
    const type = payload?.detail?.[0]?.type;
    let error = "Check that your message contains plain text and try again.";
    if (type === "string_too_long") error = LENGTH_ERROR;
    if (type === "string_too_short" || type === "value_error" || type === "missing") error = EMPTY_ERROR;
    return { error, invalidInput: true };
  }
  if (response.status === 413) return { error: LENGTH_ERROR, invalidInput: true };
  if (response.status === 503) {
    return { error: experimentalV3 ? "TRACE V3 is unavailable. Try spam-only mode below." :
      "The prediction service is unavailable. Try again later." };
  }
  if (!response.ok) return { error: SERVER_ERROR };
  const result = await response.json().catch(() => null);
  if (!result || !Object.hasOwn(OUTCOMES, result.label)) return { error: SERVER_ERROR };
  if (result.label === "phishing" && (!experimentalV3 || result.experimental !== true)) {
    return { error: SERVER_ERROR };
  }
  return result;
}

function descriptionFor(prediction) {
  const label = prediction.label;
  return label === "inconclusive" && Object.hasOwn(INCONCLUSIVE_REASONS, prediction.reason)
    ? INCONCLUSIVE_REASONS[prediction.reason]
    : OUTCOMES[label].description;
}

function explanationFor(prediction) {
  const description = descriptionFor(prediction);
  return prediction.label === "ham" ? `${description} ${HAM_LIMITATION}` : description;
}

function renderVerdict(label, prediction, fallback = "") {
  const row = label.closest(".verdict");
  const icon = row.querySelector(".verdict-icon");
  const indicator = row.querySelector(".verdict-indicator");
  const outcome = prediction ? OUTCOMES[prediction.label] : null;
  label.textContent = outcome?.title ?? fallback;
  icon.textContent = outcome?.icon ?? "";
  icon.hidden = indicator.hidden = !outcome;
  if (outcome) row.dataset.outcome = prediction.label;
  else delete row.dataset.outcome;
}

function renderResult(prediction) {
  renderVerdict(resultLabel, prediction);
  resultDesc.textContent = explanationFor(prediction);
  setStage("result");
  analysisStatus.textContent = OUTCOMES[prediction.label].title + ". " + explanationFor(prediction);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (pendingPrediction || readingFile) return;
  clearError();
  const text = smsText.value;
  const error = countCharacters(text) > MAX_CHARS ? LENGTH_ERROR : isBlank(text) ? EMPTY_ERROR : null;
  if (error) { showError(error, true); smsText.focus(); return; }
  const request = new AbortController();
  pendingPrediction = request;
  const wasRevision = revisionMode;
  const revision = inputRevision;
  if (wasRevision) {
    revisionPrediction = null;
    renderComparison();
    setStage("revisionPending");
  } else setStage("pending");
  updateAnalyzeButton();
  analysisStatus.textContent = wasRevision ? "Analyzing revised SMS message…" : "Analyzing SMS message…";
  try {
    const result = await requestPrediction(text, request.signal);
    if (pendingPrediction !== request || revision !== inputRevision) return;
    if (result.error) {
      pendingPrediction = null;
      updateAnalyzeButton();
      if (wasRevision) { setStage("variation"); renderComparison(); }
      else setStage("compose");
      showError(result.error, result.invalidInput);
      return;
    }
    if (wasRevision) {
      revisionPrediction = result;
      renderComparison();
      setStage("compared");
      analysisStatus.textContent = "Revision: " + OUTCOMES[result.label].title + ". " + explanationFor(result);
    } else {
      original = { text, prediction: result };
      renderResult(result);
    }
  } catch (err) {
    if (pendingPrediction === request && revision === inputRevision && err?.name !== "AbortError") {
      pendingPrediction = null;
      updateAnalyzeButton();
      if (wasRevision) { setStage("variation"); renderComparison(); }
      else setStage("compose");
      showError(NETWORK_ERROR);
    }
  } finally {
    if (pendingPrediction === request) {
      pendingPrediction = null;
      updateAnalyzeButton();
    }
  }
});

tryVariationBtn.addEventListener("click", () => {
  if (!original) return;
  revisionMode = true;
  revisionPrediction = null;
  renderComparison();
  setStage("variation");
  updateAnalyzeButton();
  analysisStatus.textContent = "Original prediction preserved. Edit and analyze your revision.";
  smsText.focus();
});
function clearComparison() {
  originalMessage.textContent = "";
  renderVerdict(originalLabel, null);
  el("originalDescription").textContent = "";
  revisionMessage.textContent = "";
  renderVerdict(revisionLabel, null, "Not analyzed");
  el("revisionDescription").textContent = "";
}

el("cancelVariationBtn").addEventListener("click", () => {
  if (!revisionMode || !original) return;
  cancelInputWork();
  revisionMode = false;
  clearComparison();
  clearError();
  smsText.value = original.text;
  updateCount();
  updateAnalyzeButton();
  renderResult(original.prediction);
  analysisStatus.textContent = "Variation canceled. " + analysisStatus.textContent;
  tryVariationBtn.focus();
});

startFreshBtn.addEventListener("click", () => {
  cancelInputWork();
  revisionMode = false;
  original = null;
  clearComparison();
  renderVerdict(resultLabel, null);
  resultDesc.textContent = "";
  smsText.value = "";
  updateCount();
  clearView();
  setStage("compose");
  updateAnalyzeButton();
  analysisStatus.textContent = "Workspace cleared. Enter a new SMS message.";
  smsText.focus();
});

// Lightweight silver mouse spotlight. No particles, canvas, third-party scripts or timers.
(function setupPointerLight() {
  const light = el("pointerLight");
  const fine = window.matchMedia("(hover: hover) and (pointer: fine) and (min-width: 731px)");
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)");
  // Update the external stylesheet's existing rule, never an inline style attribute.
  const lightRule = [...document.styleSheets].flatMap(sheet => [...sheet.cssRules])
    .find(rule => rule.selectorText === ".pointer-light");
  if (!lightRule) return;
  let targetX = -1000, targetY = -1000, x = -1000, y = -1000, frame = 0;
  const radius = 325;
  function tick() {
    x += (targetX - x) * 0.13;
    y += (targetY - y) * 0.13;
    lightRule.style.transform = `translate3d(${x - radius}px,${y - radius}px,0)`;
    if (Math.abs(targetX - x) > .45 || Math.abs(targetY - y) > .45) frame = requestAnimationFrame(tick);
    else frame = 0;
  }
  document.addEventListener("pointermove", (event) => {
    if (!fine.matches || reduce.matches || event.pointerType !== "mouse") return;
    targetX = event.clientX;
    targetY = event.clientY;
    if (x < -500) { x = targetX; y = targetY; }
    light.classList.add("is-visible");
    if (!frame) frame = requestAnimationFrame(tick);
  }, { passive: true });
  function stop() {
    light.classList.remove("is-visible");
    cancelAnimationFrame(frame);
    frame = 0;
    x = y = -1000;
  }
  document.addEventListener("pointerleave", stop);
  window.addEventListener("blur", stop);
  fine.addEventListener("change", stop);
  reduce.addEventListener("change", stop);
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) stop();
  });
})();

updateCount();
setStage("compose");
updateAnalyzeButton();
