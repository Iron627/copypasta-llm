import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken

device = "cuda" if torch.cuda.is_available() else "cpu"

enc = tiktoken.get_encoding("gpt2")

encode = lambda s: enc.encode(s)
decode = lambda ids: enc.decode(ids)

vocab_size = enc.n_vocab

batch_size = 32
chunk_size = 64

train_split = None
test_split = None


def load_dataset(path="cleaned_pasta.txt"):
    global train_split, test_split

    if train_split is not None and test_split is not None:
        return train_split, test_split

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(encode(text), dtype=torch.long, device=device)
    train_split = data[:int(0.9 * len(data))]
    test_split = data[int(0.9 * len(data)):]

    return train_split, test_split


def get_chunk(split):
    train_data, test_data = load_dataset()
    source = train_data if split == "train" else test_data
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
        self.dropout = nn.Dropout(0.2)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x, mask):
        norm_x = self.ln1(x)

        attn_out, _ = self.attn(
            norm_x, norm_x, norm_x,
            attn_mask=mask
        )
        x = x + self.dropout(attn_out)

        ff_out = self.ff(self.ln2(x))
        x = x + self.dropout(ff_out)

        return x

class transformerLM(nn.Module):
    def __init__(self, vocab_size, n_embd=256, chunk_size=64, n_layer=6):
        super().__init__()
        self.chunk_size = chunk_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(chunk_size, n_embd)

        self.blocks = nn.ModuleList([Block(n_embd) for _ in range(n_layer)])

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
    def generate(self, inputs, max_out, temp=0.8, top_k=50):
        for _ in range(max_out):
            inputs_cond = inputs[:, -self.chunk_size:]

            logits, _ = self(inputs_cond)
            logits = logits[:, -1, :]
            logits = logits / max(temp, 1e-6)

            top_k = min(top_k, logits.size(-1))
            v, ix = torch.topk(logits, top_k)

            probs = F.softmax(v, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            out_token = ix.gather(1, sampled)

            inputs = torch.cat((inputs, out_token), dim=1)

        return inputs


@torch.no_grad()
def estimate_loss(model):
    model.eval()

    losses = {"train": 0.0, "test": 0.0}
    eval_iters = 3

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
        vocab_size,
        n_embd=256,
        chunk_size=chunk_size,
        n_layer=6
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

    for timesteps in range(50000):
        xtrain, ytrain = get_chunk("train")

        logits, loss = model(xtrain, ytrain)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if timesteps % 1000 == 0:
            print("step", timesteps)

        if timesteps % 5000 == 0:
            losses = estimate_loss(model)
            print(f"step {timesteps}: train {losses['train']:.4f}, test {losses['test']:.4f}")

        if timesteps % 10000 == 0 and timesteps > 0:
            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "n_embd": 256,
                "chunk_size": chunk_size,
                "n_layer": 6,
            }, f"transformer_checkpoint_{timesteps}.pt")

    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "n_embd": 256,
        "chunk_size": chunk_size,
        "n_layer": 6,
    }, "transformer_checkpoint.pt")

    start = torch.tensor([[enc.eot_token]], dtype=torch.long, device=device)
    print(decode(model.generate(start, max_out=150, temp=0.8, top_k=50)[0].tolist()))
