import torch
import torch.nn as nn
import torch.nn.functional as F


with open("cleaned_pasta.txt", "r", encoding="utf-8") as f:
    text = f.read()

vocab = sorted(list(set(text)))
device = "cuda" if torch.cuda.is_available() else "cpu"

map_s_to_i = {s: i for i, s in enumerate(vocab)}
map_i_to_s = {i: s for i, s in enumerate(vocab)}

encode = lambda s: [map_s_to_i[c] for c in s]
decode = lambda ids: "".join([map_i_to_s[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long, device=device)

train_split = data[:int(0.9 * len(data))]
test_split = data[int(0.9 * len(data)):]

batch_size = 64
chunk_size = 256


def get_chunk(split):
    source = train_split if split == "train" else test_split
    ix = torch.randint(len(source) - chunk_size - 1, (batch_size,), device=device)
    offsets = torch.arange(chunk_size, device=device)

    x = source[ix[:, None] + offsets[None, :]]
    y = source[ix[:, None] + offsets[None, :] + 1]

    return x, y


class Block(nn.Module):
    def __init__(self, n_embd, num_heads=8):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, num_heads=num_heads, batch_first=True)

        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, mask):
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=mask)
        x = x + attn_out

        x = x + self.ff(self.ln2(x))
        return x


class transformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd=128, chunk_size=128, n_layer=4):
        super().__init__()

        self.chunk_size = chunk_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(chunk_size, n_embd)

        self.blocks = nn.ModuleList([
            Block(n_embd) for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        token_emb = self.token_embedding_table(inputs)
        pos_emb = self.position_embedding_table(torch.arange(T, device=inputs.device))

        x = token_emb + pos_emb

        mask = torch.triu(
            torch.ones(T, T, device=inputs.device),
            diagonal=1
        ).bool()

        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, C),
                targets.reshape(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(self, inputs, max_out, temp=0.8):
        for _ in range(max_out):
            inputs_cond = inputs[:, -self.chunk_size:]

            logits, _ = self(inputs_cond)
            logits = logits[:, -1, :]

            logits = logits / temp

            
            top_k = min(50, logits.size(-1))
            v, ix = torch.topk(logits, top_k)

            probs = torch.zeros_like(logits).scatter_(
                1, ix, F.softmax(v, dim=-1)
            )

            out_char = torch.multinomial(probs, num_samples=1)
            inputs = torch.cat((inputs, out_char), dim=1)

        return inputs


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


if __name__ == "__main__":
    model = transformerLM(
        len(vocab),
        n_embd=384,
        chunk_size=chunk_size,
        n_layer=8
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for timesteps in range(50000):
        xtrain, ytrain = get_chunk("train")

        logits, loss = model(xtrain, ytrain)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if timesteps % 2000 == 0:
            losses = estimate_loss()
            print(f"step {timesteps}: train {losses['train']:.4f}, test {losses['test']:.4f}")
            print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    torch.save({
        "model_state": model.state_dict(),
        "vocab": vocab,
        "n_embd": 384,
        "chunk_size": chunk_size,
        "n_layer": 8,
    }, "transformer_checkpoint.pt")

    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, max_out=300)[0].tolist()))