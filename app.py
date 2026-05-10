import glob
import os
import re
import threading
import time

import torch
from flask import Flask, jsonify, render_template, request

from transformer import TOKENIZER_MODEL_PATH, decode, encode, TransformerLM, vocab_size


app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"

_models = {}
_model_lock = threading.Lock()


def checkpoint_step(path):
    match = re.search(r"_(\d+)\.pt$", os.path.basename(path))
    return int(match.group(1)) if match else None


def list_checkpoints():
    return sorted(glob.glob("*.pt"))


def default_checkpoint(candidates):
    configured = os.environ.get("CHAT_MODEL_PATH")
    if configured:
        return configured

    if not candidates:
        raise FileNotFoundError("No transformer checkpoint files found.")

    stepped = [path for path in candidates if checkpoint_step(path) is not None]
    if stepped:
        return max(stepped, key=lambda path: checkpoint_step(path))

    if "transformer_checkpoint.pt" in candidates:
        return "transformer_checkpoint.pt"

    return max(candidates, key=os.path.getmtime)


def load_model(selected_path=None):
    candidates = list_checkpoints()
    checkpoint_path = selected_path or default_checkpoint(candidates)
    if checkpoint_path not in candidates:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    cached = _models.get(checkpoint_path)
    if cached is not None:
        return cached["model"], cached["info"]

    with _model_lock:
        cached = _models.get(checkpoint_path)
        if cached is not None:
            return cached["model"], cached["info"]

        checkpoint = torch.load(checkpoint_path, map_location=device)

        model = TransformerLM(
            checkpoint.get("vocab_size", vocab_size),
            n_embd=checkpoint["n_embd"],
            chunk_size=checkpoint["chunk_size"],
            n_layer=checkpoint["n_layer"],
        ).to(device)
        model.load_state_dict(checkpoint["model_state"])
        model.eval()

        info = {
            "checkpoint": checkpoint_path,
            "step": checkpoint_step(checkpoint_path),
            "device": device,
            "tokenizer_model": TOKENIZER_MODEL_PATH,
            "chunk_size": checkpoint["chunk_size"],
            "n_layer": checkpoint["n_layer"],
            "n_embd": checkpoint["n_embd"],
        }
        _models[checkpoint_path] = {"model": model, "info": info}
        return model, info


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
        candidates = list_checkpoints()
        _, info = load_model(default_checkpoint(candidates))
        return jsonify({"ready": True, "model": info, "models": candidates})
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
    selected_model = (payload.get("model") or "").strip() or None

    try:
        model, info = load_model(selected_model)
        input_ids = encode(prompt)
        if not input_ids:
            input_ids = encode(" ")
            if not input_ids:
                return jsonify({"error": "Tokenizer could not encode the prompt."}), 500
        start = torch.tensor([input_ids], dtype=torch.long, device=device)

        started = time.perf_counter()
        output = model.generate(start, max_out=max_tokens, temp=temperature, top_k=top_k)
        elapsed_ms = round((time.perf_counter() - started) * 1000)

        text = decode(output[0].tolist())
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
