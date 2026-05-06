import torch
import tiktoken
from transformer import transformerLM

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("transformer_checkpoint_15000.pt", map_location=device)

enc = tiktoken.get_encoding("gpt2")
vocab_size = enc.n_vocab

encode = lambda s: enc.encode(s)
decode = lambda ids: enc.decode(ids)

model = transformerLM(
    vocab_size,
    n_embd=checkpoint["n_embd"],
    chunk_size=checkpoint["chunk_size"],
    n_layer=checkpoint["n_layer"]
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Type something (or 'exit'):\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    start_ids = encode(user_input)

    if len(start_ids) == 0:
        start_ids = [enc.eot_token]

    start = torch.tensor([start_ids], dtype=torch.long, device=device)

    out = model.generate(start, max_out=100, temp=0.7, top_k=50)
    text = decode(out[0].tolist())

    print("Model:", text[len(user_input):])
