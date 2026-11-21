const API_URL = "/predict";
const MAX_CHARS = 5000;
const MAX_FILE_BYTES = MAX_CHARS * 4 + 3; // UTF-8 uses up to four bytes per character, plus an optional BOM.
const LENGTH_ERROR = "Keep your SMS message to 5,000 characters or fewer.";
const UPLOAD_LENGTH_ERROR = "The uploaded file exceeds 5,000 characters. Choose a shorter .txt file.";
const EMPTY_ERROR = "Enter an SMS message to analyze.";
const SERVER_ERROR = "The prediction service had a problem. Try again.";
const NETWORK_ERROR = "Could not reach the prediction service. Check your connection and try again.";

const OUTCOMES = {
  spam: {
    title: "Likely Spam",
    description: "This message contains patterns the model associates with SMS spam.",
    icon: "⚠",
  },
  ham: {
    title: "Likely Ham",
    description: "This message contains patterns the model associates with legitimate SMS.",
    icon: "✓",
  },
  inconclusive: {
    title: "Inconclusive",
    description: "The message contains no words or phrases the model recognizes.",
    icon: "?",
  },
};

// Why the API abstained (`reason` field, see docs/API.md). Unknown or missing
// reasons fall back to the generic inconclusive description above.
const INCONCLUSIVE_REASONS = {
  no_letters: "The message has no words to analyze.",
  unsupported_script: "The model only supports English SMS written in the Latin alphabet.",
  no_recognized_features: "The message contains no words or phrases the model recognizes.",
};

const form = document.getElementById("smsForm");
const smsText = document.getElementById("smsText");
const dzPlaceholder = document.getElementById("dzPlaceholder");
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const charCount = document.getElementById("charCount");
const statWords = document.getElementById("statWords");
const statChars = document.getElementById("statChars");
const statSentences = document.getElementById("statSentences");
const uploadBtn = document.getElementById("uploadBtn");
const clearBtn = document.getElementById("clearBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const messageError = document.getElementById("messageError");
const analysisStatus = document.getElementById("analysisStatus");

const resultCard = document.getElementById("resultCard");
const resultBox = document.getElementById("resultBox");
const resultIcon = document.getElementById("resultIcon");
const resultLabel = document.getElementById("resultLabel");
const resultDesc = document.getElementById("resultDesc");

let pendingPrediction = null;
let readingFile = false;
let inputRevision = 0;

function countCharacters(text) {
  return [...text].length;
}

function isBlank(text) {
  // Match Python's whitespace-only check, including its four ASCII separators.
  return /^[\p{White_Space}\u001c-\u001f]*$/u.test(text);
}

function updateStats() {
  const text = smsText.value;
  const length = countCharacters(text);

  charCount.textContent = length;
  dzPlaceholder.classList.toggle("hidden", length > 0);

  if (isBlank(text)) {
    statWords.textContent = "--";
    statChars.textContent = "--";
    statSentences.textContent = "--";
    return length;
  }

  statWords.textContent = text.trim().split(/\s+/).length;
  statChars.textContent = length;
  statSentences.textContent = text.split(/[.!?]+/).filter((part) => part.trim()).length;
  return length;
}

function updateAnalyzeButton() {
  analyzeBtn.setAttribute("aria-disabled", String(readingFile || pendingPrediction !== null));
  analyzeBtn.textContent = pendingPrediction ? "Analyzing…" : "Analyze SMS";
}

function clearFeedback() {
  resultCard.hidden = true;
  messageError.textContent = "";
  analysisStatus.textContent = "";
  smsText.removeAttribute("aria-invalid");
}

function showError(message, invalidInput = false) {
  analysisStatus.textContent = "";
  messageError.textContent = message;
  if (invalidInput) smsText.setAttribute("aria-invalid", "true");
}

function invalidateInput() {
  inputRevision += 1;
  pendingPrediction?.abort();
  pendingPrediction = null;
  readingFile = false;
  clearFeedback();
  updateAnalyzeButton();
}

smsText.addEventListener("input", () => {
  invalidateInput();
  if (updateStats() > MAX_CHARS) showError(LENGTH_ERROR, true);
});

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (file) readTextFile(file);
});

["dragenter", "dragover"].forEach((event) => {
  dropzone.addEventListener(event, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  });
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("drag-over"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("drag-over");
  const file = e.dataTransfer.files[0];
  if (file) readTextFile(file);
});

async function readTextFile(file) {
  invalidateInput();
  if (!file.name.toLowerCase().endsWith(".txt")) {
    showError("Choose a .txt file to upload.");
    return;
  }
  if (file.size > MAX_FILE_BYTES) {
    showError(UPLOAD_LENGTH_ERROR);
    return;
  }

  const revision = inputRevision;
  readingFile = true;
  updateAnalyzeButton();
  analysisStatus.textContent = "Reading SMS file…";

  try {
    const text = (await file.text()).replace(/\r\n?/g, "\n");
    if (revision !== inputRevision) return;
    if (countCharacters(text) > MAX_CHARS) {
      showError(UPLOAD_LENGTH_ERROR);
      return;
    }
    smsText.value = text;
    updateStats();
    analysisStatus.textContent = "SMS file loaded.";
  } catch {
    if (revision === inputRevision) showError("Could not read the file. Try uploading it again.");
  } finally {
    if (revision === inputRevision) {
      readingFile = false;
      updateAnalyzeButton();
    }
  }
}

clearBtn.addEventListener("click", () => {
  invalidateInput();
  smsText.value = "";
  updateStats();
  analysisStatus.textContent = "Message cleared.";
});

async function requestPrediction(text, signal) {
  const response = await fetch(API_URL, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
    signal,
  });

  if (response.status === 422) {
    const payload = await response.json().catch(() => null);
    const type = payload?.detail?.[0]?.type;
    let error = "Check that your message contains plain text and try again.";
    if (type === "string_too_long") error = LENGTH_ERROR;
    if (type === "string_too_short" || type === "value_error" || type === "missing") error = EMPTY_ERROR;
    return { error, invalidInput: true };
  }
  if (!response.ok) return { error: SERVER_ERROR };

  const result = await response.json().catch(() => null);
  if (!result || !Object.hasOwn(OUTCOMES, result.label)) return { error: SERVER_ERROR };
  return result;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (pendingPrediction || readingFile) return;

  clearFeedback();
  const text = smsText.value;
  let inputError = "";
  if (countCharacters(text) > MAX_CHARS) inputError = LENGTH_ERROR;
  else if (isBlank(text)) inputError = EMPTY_ERROR;
  if (inputError) {
    showError(inputError, true);
    smsText.focus();
    return;
  }

  const request = new AbortController();
  pendingPrediction = request;
  updateAnalyzeButton();
  analysisStatus.textContent = "Analyzing SMS message…";

  try {
    const result = await requestPrediction(text, request.signal);
    if (pendingPrediction !== request) return;
    if (result.error) {
      showError(result.error, result.invalidInput);
      return;
    }
    showResult(result.label, result.reason);
  } catch {
    if (pendingPrediction === request) showError(NETWORK_ERROR);
  } finally {
    if (pendingPrediction === request) {
      pendingPrediction = null;
      updateAnalyzeButton();
    }
  }
});

function showResult(label, reason) {
  const { title, icon } = OUTCOMES[label];
  let { description } = OUTCOMES[label];
  if (label === "inconclusive" && Object.hasOwn(INCONCLUSIVE_REASONS, reason)) {
    description = INCONCLUSIVE_REASONS[reason];
  }
  resultBox.className = "result-box " + label;
  resultIcon.textContent = icon;
  resultLabel.textContent = title;
  resultDesc.textContent = description;
  resultCard.hidden = false;
  analysisStatus.textContent = title + ". " + description;
}

updateStats();
