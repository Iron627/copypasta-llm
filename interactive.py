import os
import torch
from transformer import decode, encode, non_empty_ids, transformerLM, vocab_size

device = "cuda" if torch.cuda.is_available() else "cpu"

pt_files = sorted([f for f in os.listdir(".") if f.endswith(".pt")])

if not pt_files:
    raise FileNotFoundError("No .pt checkpoint files found in this folder.")

print("Available models:\n")
for i, file in enumerate(pt_files):
    print(f"{i}: {file}")

while True:
    choice = input("\nChoose model number: ")

    try:
        choice = int(choice)
        if 0 <= choice < len(pt_files):
            break
    except ValueError:
        pass

    print("Invalid choice.")

checkpoint_path = pt_files[choice]
print(f"\nLoading {checkpoint_path}...\n")

checkpoint = torch.load(checkpoint_path, map_location=device)

model = transformerLM(
    vocab_size,
    n_embd=checkpoint["n_embd"],
    chunk_size=checkpoint["chunk_size"],
    n_layer=checkpoint["n_layer"]
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

while True:
    temp_input = input("Temperature? recommended 0.5-0.8 [default 0.7]: ").strip()

    if temp_input == "":
        temp = 0.7
        break

    try:
        temp = float(temp_input)
        if temp > 0:
            break
    except ValueError:
        pass

    print("Invalid temperature.")

while True:
    top_k_input = input("Top-k? recommended 20-50 [default 50]: ").strip()

    if top_k_input == "":
        top_k = 50
        break

    try:
        top_k = int(top_k_input)
        if top_k > 0:
            break
    except ValueError:
        pass

    print("Invalid top-k.")

while True:
    max_out_input = input("Max output tokens? [default 100]: ").strip()

    if max_out_input == "":
        max_out = 100
        break

    try:
        max_out = int(max_out_input)
        if max_out > 0:
            break
    except ValueError:
        pass

    print("Invalid max output.")

print("\nType something (or 'exit'):\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    start_ids = encode(user_input)

    if len(start_ids) == 0:
        start_ids = non_empty_ids(user_input)

    start = torch.tensor([start_ids], dtype=torch.long, device=device)

    out = model.generate(start, max_out=max_out, temp=temp, top_k=top_k)
    text = decode(out[0].tolist())

    print("Model:", text[len(user_input):])
