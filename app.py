import glob
import os
import re
import threading
import time

import torch
import tiktoken
from flask import Flask, jsonify, render_template, request

from transformer import transformerLM


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
enc = tiktoken.get_encoding("gpt2")

_model = None
_model_info = None
_model_lock = threading.Lock()


def checkpoint_step(path):
    match = re.search(r"_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def default_checkpoint():
    configured = os.environ.get("CHAT_MODEL_PATH")
    if configured:
        return configured

    preferred = "transformer_checkpoint_15000.pt"
    if os.path.exists(preferred):
        return preferred

    candidates = glob.glob("transformer_checkpoint*.pt")
    if not candidates:
        raise FileNotFoundError("No transformer checkpoint files found.")

    return max(candidates, key=lambda path: checkpoint_step(path) or -1)


def load_model():
    global _model, _model_info

    if _model is not None:
        return _model, _model_info

    with _model_lock:
        if _model is not None:
            return _model, _model_info

        checkpoint_path = default_checkpoint()
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = transformerLM(
            checkpoint.get("vocab_size", enc.n_vocab),
            n_embd=checkpoint["n_embd"],
            chunk_size=checkpoint["chunk_size"],
            n_layer=checkpoint["n_layer"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        _model = model
        _model_info = {
            "checkpoint": checkpoint_path,
            "step": checkpoint_step(checkpoint_path),
            "device": device,
            "chunk_size": checkpoint["chunk_size"],
            "n_layer": checkpoint["n_layer"],
            "n_embd": checkpoint["n_embd"],
        }
        return _model, _model_info


def clamp_number(value, default, minimum, maximum, cast):
    try:
        value = cast(value)
    except (TypeError, ValueError):
        value = default

    return max(minimum, min(maximum, value))


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    try:
        _, info = load_model()
        return jsonify({"ready": True, "model": info})
    except Exception as exc:
        return jsonify({"ready": False, "error": str(exc)}), 503


@app.post("/api/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    prompt = (payload.get("message") or payload.get("prompt") or "").strip()

    if not prompt:
        return jsonify({"error": "Message cannot be empty."}), 400

    temperature = clamp_number(payload.get("temperature"), 0.7, 0.05, 2.0, float)
    top_k = clamp_number(payload.get("top_k"), 50, 1, 500, int)
    max_tokens = clamp_number(
        payload.get("max_tokens", payload.get("max_out")), 100, 1, 1024, int
    )

    try:
        model, info = load_model()
        input_ids = enc.encode(prompt) or [enc.eot_token]
        start = torch.tensor([input_ids], dtype=torch.long, device=device)

        started = time.perf_counter()
        output = model.generate(start, max_out=max_tokens, temp=temperature, top_k=top_k)
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        text = enc.decode(output[0].tolist())
        reply = text[len(prompt):].strip() or text.strip()

        return jsonify(
            {
                "reply": reply,
                "model": info,
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "generated_tokens": max(0, output.shape[1] - len(input_ids)),
                    "elapsed_ms": elapsed_ms,
                },
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=7860, debug=True)
