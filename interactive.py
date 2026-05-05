import torch
from ngram import nGramLM


device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load("ngram_checkpoint.pt", map_location=device)

vocab = checkpoint["vocab"]
map_s_to_i = {s: i for i, s in enumerate(vocab)}
map_i_to_s = {i: s for i, s in enumerate(vocab)}

encode = lambda s: [map_s_to_i[c] for c in s]
decode = lambda i: "".join([map_i_to_s[i] for i in i])

model = nGramLM(
    len(vocab),
    n_embd=checkpoint["n_embd"],
    n=checkpoint["n"],
    chunk_size=checkpoint["chunk_size"]
).to(device)

model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Type something (or 'exit'):\n")

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        break

    try:
        start = torch.tensor([encode(user_input)], dtype=torch.long, device=device)
    except KeyError:
        print("contains unknown characters")
        continue

    out = model.generate(start, max_out=200)
    text = decode(out[0].tolist())

    print("Model:", text[len(user_input):])