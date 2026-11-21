const API_URL = "/predict";
const MAX_CHARS = 5000;

const emailText = document.getElementById("emailText");
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

const resultCard = document.getElementById("resultCard");
const resultBox = document.getElementById("resultBox");
const resultIcon = document.getElementById("resultIcon");
const resultLabel = document.getElementById("resultLabel");
const resultDesc = document.getElementById("resultDesc");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceFill = document.getElementById("confidenceFill");

function updateStats() {
  const text = emailText.value;

  charCount.textContent = text.length;
  dzPlaceholder.classList.toggle("hidden", text.length > 0);

  if (text.trim().length === 0) {
    statWords.textContent = "--";
    statChars.textContent = "--";
    statSentences.textContent = "--";
    return;
  }

  const words = text.trim().split(/\s+/).filter(Boolean);
  const sentences = text.split(/[.!?]+/).filter((s) => s.trim().length > 0);

  statWords.textContent = words.length;
  statChars.textContent = text.length;
  statSentences.textContent = sentences.length;
}

emailText.addEventListener("input", updateStats);

uploadBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (file) readTextFile(file);
});

["dragenter", "dragover"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add("drag-over");
  })
);

["dragleave", "drop"].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove("drag-over");
  })
);

dropzone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  if (file && file.name.endsWith(".txt")) readTextFile(file);
});

function readTextFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    emailText.value = reader.result.slice(0, MAX_CHARS);
    updateStats();
  };
  reader.readAsText(file);
}

clearBtn.addEventListener("click", () => {
  emailText.value = "";
  updateStats();
  resultCard.hidden = true;
});

analyzeBtn.addEventListener("click", async () => {
  const text = emailText.value.trim();
  if (!text) return;

  analyzeBtn.disabled = true;
  analyzeBtn.textContent = "Analyzing...";

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });

    if (!res.ok) throw new Error("Request failed");

    const data = await res.json();
    showResult(data.label, data.spam_probability);
  } catch (err) {
    alert("Could not reach the prediction API. Make sure the backend is running at " + API_URL);
  } finally {
    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "✨ Analyze Email";
  }
});

function showResult(label, spamProbability) {
  const isSpam = label === "spam";
  const confidence = isSpam ? spamProbability : 1 - spamProbability;
  const confidencePct = Math.round(confidence * 1000) / 10;

  resultBox.classList.remove("spam", "ham");
  resultBox.classList.add(isSpam ? "spam" : "ham");

  resultIcon.textContent = isSpam ? "⚠" : "✓";
  resultLabel.textContent = isSpam ? "SPAM / Unsafe Email" : "HAM / Safe Email";
  resultDesc.textContent = isSpam
    ? "This email looks suspicious. It may be spam."
    : "This email is not spam. It appears to be safe.";

  confidenceValue.textContent = confidencePct + "%";
  confidenceFill.style.width = confidencePct + "%";

  resultCard.hidden = false;
}

updateStats();
