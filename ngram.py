import torch
import torch.nn as nn
import torch.nn.functional as F


with open("cleaned_pasta.txt", "r", encoding="utf-8") as f:
    data = f.read()
vocab = sorted(list(set(data)))

device = "cuda" if torch.cuda.is_available() else "cpu"

map_s_to_i = {s: i for i, s in enumerate(vocab)}
map_i_to_s = {i: s for i, s in enumerate(vocab)}
encode = lambda s: [map_s_to_i[c] for c in s]
decode = lambda i: "".join([map_i_to_s[i] for i in i])

data = torch.tensor(encode(data), dtype=torch.long, device=device)

train_split = data[:int(0.9*len(data))]
test_split = data[int(0.9*len(data)):]

batch_size = 128
chunk_size = 128

# depending on batch size (B) and chunk size (C) returns a random tensor of shape BxC for transformer to train on
def get_chunk(split):
    source = train_split if split == "train" else test_split
    ix = torch.randint(len(source) - chunk_size - 1, (batch_size,), device=device)
    offsets = torch.arange(chunk_size, device=device)
    x = source[(ix[:, None] + offsets[None, :])]
    y = source[(ix[:, None] + offsets[None, :] + 1)]
    return x, y
@torch.no_grad()
def estimate_loss():
    model.eval()

    losses = {"train": 0.0, "test": 0.0}
    eval_iters = 5

    for split in ["train", "test"]:
        total_loss = 0.0

        for _ in range(eval_iters):
            xb, yb = get_chunk(split)
            _, loss = model(xb, yb)
            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()
    return losses



class nGramLM(nn.Module):
    def __init__(self, vocab_size, n_embd=64, n=16, chunk_size=128):
        super().__init__()
        self.n = n
        self.chunk_size = chunk_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.conv = nn.Conv1d(n_embd, n_embd, kernel_size=n)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        emb = self.token_embedding_table(inputs)   # (B, T, C)
        emb = emb.transpose(1, 2)                  # (B, C, T)

        emb = F.pad(emb, (self.n - 1, 0))          # left pad only
        x = F.relu(self.conv(emb))                 # (B, C, T)

        x = x.transpose(1, 2)                      # (B, T, C)
        logits = self.lm_head(x)                   # (B, T, vocab)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B*T, C),
                targets.reshape(B*T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inputs, max_out):
        for _ in range(max_out):
            inputs_cond = inputs[:, -chunk_size:]
            logits, _ = self(inputs_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits / 0.8, dim=-1)
            out_char = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, out_char), dim=1)

        return inputs
if __name__ == "__main__":
    

    model = nGramLM(len(vocab), n_embd=128, n=32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for timesteps in range(100000):
        xtrain, ytrain = get_chunk("train")
        
        logits, loss = model(xtrain, ytrain)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if timesteps % 2000 == 0:
            losses = estimate_loss()
            print(f"step {timesteps}: train {losses['train']:.4f}, test {losses['test']:.4f}")

        
        
    torch.save({
    "model_state": model.state_dict(),
    "vocab": vocab,
    "n_embd": 128,
    "n": 32,
    "chunk_size": chunk_size,
}, "ngram_checkpoint.pt")
    start = torch.zeros((1, 1), dtype=torch.long,device=device)
    print(decode(model.generate(start, max_out=300)[0].tolist()))