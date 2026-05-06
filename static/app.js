const els = {
  messages: document.getElementById("messages"),
  form: document.getElementById("chatForm"),
  prompt: document.getElementById("prompt"),
  sendButton: document.getElementById("sendButton"),
  clearChat: document.getElementById("clearChat"),
  errorText: document.getElementById("errorText"),
  temperature: document.getElementById("temperature"),
  temperatureValue: document.getElementById("temperatureValue"),
  topK: document.getElementById("topK"),
  maxTokens: document.getElementById("maxTokens"),
  modelStatus: document.getElementById("modelStatus"),
  connectionLabel: document.getElementById("connectionLabel"),
  statusDot: document.getElementById("statusDot"),
};

const state = {
  messages: [],
  busy: false,
};

const escapeHtml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

function markdownish(text) {
  const parts = escapeHtml(text).split(/```([\s\S]*?)```/g);
  return parts
    .map((part, index) => {
      if (index % 2 === 1) return `<pre>${part.trim()}</pre>`;
      return part
        .split(/\n{2,}/)
        .filter(Boolean)
        .map((block) => `<p>${block.replace(/\n/g, "<br>")}</p>`)
        .join("");
    })
    .join("");
}

function renderMessage(role, content, extraClass = "") {
  const article = document.createElement("article");
  article.className = `message ${role} ${extraClass}`.trim();
  article.innerHTML = `<div class="bubble">${markdownish(content)}</div>`;
  els.messages.appendChild(article);
  els.messages.scrollTop = els.messages.scrollHeight;
  return article;
}

function setBusy(busy, label = busy ? "Generating…" : "Ready") {
  state.busy = busy;
  els.sendButton.disabled = busy;
  els.prompt.disabled = busy;
  els.messages.setAttribute("aria-busy", String(busy));
  els.modelStatus.textContent = label;
  els.connectionLabel.textContent = busy ? "Working" : "Idle";
  els.statusDot.style.background = busy ? "#f59e0b" : "#4ade80";
  els.statusDot.style.boxShadow = busy
    ? "0 0 0 6px rgba(245, 158, 11, 0.12)"
    : "0 0 0 6px rgba(74, 222, 128, 0.12)";
}

function setError(message = "") {
  els.errorText.textContent = message;
}

function syncControls() {
  els.temperatureValue.textContent = Number(els.temperature.value).toFixed(2).replace(/0+$/, "").replace(/\.$/, "");
}

function buildPayload() {
  return {
    message: els.prompt.value.trim(),
    prompt: els.prompt.value.trim(),
    temperature: Number(els.temperature.value),
    top_k: Number(els.topK.value),
    max_tokens: Number(els.maxTokens.value),
    max_out: Number(els.maxTokens.value),
    history: state.messages.map(({ role, content }) => ({ role, content })),
    messages: state.messages.map(({ role, content }) => ({ role, content })),
  };
}

function normalizeReply(data) {
  if (typeof data === "string") return data;
  return (
    data?.reply ??
    data?.response ??
    data?.content ??
    data?.message ??
    data?.error ??
    data?.text ??
    data?.result ??
    JSON.stringify(data)
  );
}

async function sendMessage(text) {
  state.messages.push({ role: "user", content: text });
  renderMessage("user", text);
  const pending = renderMessage("assistant", "Thinking…");

  setBusy(true);
  setError("");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildPayload()),
    });

    const raw = await response.text();
    let data = raw;
    try { data = JSON.parse(raw); } catch (_) {}

    if (!response.ok) {
      throw new Error(normalizeReply(data) || `Request failed (${response.status})`);
    }

    const reply = normalizeReply(data).trim();
    pending.querySelector(".bubble").innerHTML = markdownish(reply);
    state.messages.push({ role: "assistant", content: reply });
  } catch (err) {
    pending.remove();
    setError(err?.message || "Unable to reach the local model.");
    renderMessage("assistant", "Connection error. Check the backend and try again.");
  } finally {
    setBusy(false);
    els.prompt.value = "";
    autosize();
  }
}

async function loadStatus() {
  try {
    const response = await fetch("/api/status");
    const data = await response.json();

    if (!response.ok || !data.ready) {
      throw new Error(data?.error || "Model is not ready.");
    }

    const step = data.model?.step ? `${data.model.step.toLocaleString()} steps` : "checkpoint";
    const device = data.model?.device ? ` • ${data.model.device}` : "";
    els.modelStatus.textContent = `${step}${device}`;
    els.connectionLabel.textContent = "Ready";
  } catch (err) {
    els.modelStatus.textContent = "Backend offline";
    els.connectionLabel.textContent = "Check server";
    els.statusDot.style.background = "#ef4444";
    els.statusDot.style.boxShadow = "0 0 0 6px rgba(239, 68, 68, 0.12)";
    setError(err?.message || "Unable to load model status.");
  }
}

function autosize() {
  els.prompt.style.height = "auto";
  els.prompt.style.height = `${Math.min(els.prompt.scrollHeight, 180)}px`;
}

els.temperature.addEventListener("input", syncControls);
els.prompt.addEventListener("input", autosize);
els.form.addEventListener("submit", (event) => {
  event.preventDefault();
  const text = els.prompt.value.trim();
  if (!text || state.busy) return;
  sendMessage(text);
});

els.prompt.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

els.clearChat.addEventListener("click", () => {
  if (state.busy) return;
  state.messages = [];
  els.messages.innerHTML = "";
  renderMessage("assistant", "Ready when you are. Try a short prompt or ask for something weird.");
  setError("");
});

document.querySelectorAll(".preset").forEach((button) => {
  button.addEventListener("click", () => {
    els.temperature.value = button.dataset.temp;
    els.topK.value = button.dataset.topk;
    els.maxTokens.value = button.dataset.max;
    syncControls();
  });
});

syncControls();
autosize();
loadStatus();
