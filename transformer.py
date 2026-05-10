import os

import sentencepiece as spm
import torch
import torch.nn as nn
import torch.nn.functional as F

# =========================
# SPEED SETTINGS
# =========================

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

# =========================
# TOKENIZER
# =========================

TOKENIZER_MODEL_PATH = os.environ.get(
    "TOKENIZER_MODEL_PATH",
    "tokenizer.model"
)

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL_PATH)

encode = lambda s: sp.encode(s, out_type=int)
decode = lambda ids: sp.decode(ids)

vocab_size = sp.get_piece_size()

# =========================
# CONFIG
# =========================

batch_size = 192
chunk_size = 128

n_embd = 256
n_layer = 6
num_heads = 8

dropout = 0.1

learning_rate = 3e-4
max_steps = 50000

# =========================
# DATASET
# =========================

train_split = None
test_split = None


def load_dataset(path="cleaned_pasta.txt"):
    global train_split, test_split

    if train_split is not None:
        return train_split, test_split

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    ids = encode(text)

    data = torch.tensor(
        ids,
        dtype=torch.long,
        device=device
    )

    split_idx = int(0.9 * len(data))

    train_split = data[:split_idx]
    test_split = data[split_idx:]

    return train_split, test_split


def get_chunk(split):
    train_data, test_data = load_dataset()

    data = train_data if split == "train" else test_data

    ix = torch.randint(
        len(data) - chunk_size - 1,
        (batch_size,),
        device=device
    )

    offsets = torch.arange(chunk_size, device=device)

    x = data[ix[:, None] + offsets]
    y = data[ix[:, None] + offsets + 1]

    return x, y


# =========================
# TRANSFORMER BLOCK
# =========================

class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        self.ln1 = nn.LayerNorm(n_embd)

        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.ln2 = nn.LayerNorm(n_embd)

        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        # =========================
        # ATTENTION
        # =========================

        residual = x

        x = self.ln1(x)

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B,
            T,
            self.num_heads,
            self.head_dim
        ).transpose(1, 2)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )

        x = x.transpose(1, 2).contiguous()
        x = x.view(B, T, C)

        x = self.proj(x)

        x = residual + self.dropout(x)

        # =========================
        # FEEDFORWARD
        # =========================

        residual = x

        x = self.ln2(x)

        x = self.ff(x)

        x = residual + self.dropout(x)

        return x


# =========================
# MODEL
# =========================

class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd,
        chunk_size,
        n_layer
    ):
        super().__init__()

        self.chunk_size = chunk_size

        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            chunk_size,
            n_embd
        )

        self.blocks = nn.ModuleList([
            Block(n_embd, num_heads)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size,
            bias=False
        )

        # weight tying
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        tok_emb = self.token_embedding(inputs)

        pos = torch.arange(
            T,
            device=inputs.device
        )

        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T)
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        inputs,
        max_out,
        temp=0.8,
        top_k=50
    ):
        self.eval()

        for _ in range(max_out):

            x = inputs[:, -self.chunk_size:]

            logits, _ = self(x)

            logits = logits[:, -1, :]

            logits = logits / max(temp, 1e-6)

            top_k = min(top_k, logits.size(-1))

            values, indices = torch.topk(
                logits,
                top_k
            )

            probs = F.softmax(values, dim=-1)

            sampled = torch.multinomial(
                probs,
                num_samples=1
            )

            next_token = indices.gather(1, sampled)

            inputs = torch.cat(
                (inputs, next_token),
                dim=1
            )

        return inputs


# =========================
# LOSS ESTIMATION
# =========================

@torch.no_grad()
def estimate_loss(model):
    model.eval()

    out = {}

    eval_iters = 5

    for split in ["train", "test"]:

        losses = []

        for _ in range(eval_iters):

            xb, yb = get_chunk(split)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16
            ):
                _, loss = model(xb, yb)

            losses.append(loss.item())

        out[split] = sum(losses) / len(losses)

    model.train()

    return out


# =========================
# TRAINING
# =========================

if __name__ == "__main__":

    model = TransformerLM(
        vocab_size=vocab_size,
        n_embd=n_embd,
        chunk_size=chunk_size,
        n_layer=n_layer
    ).to(device)


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.1
    )

    scaler = torch.amp.GradScaler("cuda")

    param_count = sum(
        p.numel() for p in model.parameters()
    ) / 1e6

    print(f"{param_count:.2f}M parameters")

    for step in range(max_steps):

        xb, yb = get_chunk("train")

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16
        ):
            logits, loss = model(xb, yb)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            print(
                f"step {step} | loss {loss.item():.4f}"
            )

        if step % 1000 == 0:

            losses = estimate_loss(model)

            print(
                f"step {step} | "
                f"train {losses['train']:.4f} | "
                f"test {losses['test']:.4f}"
            )

        if step % 1000 == 0 and step > 0:

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "vocab_size": vocab_size,
                    "n_embd": n_embd,
                    "chunk_size": chunk_size,
                    "n_layer": n_layer,
                },
                f"checkpoint_{step}.pt"
            )

    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_size": vocab_size,
            "n_embd": n_embd,
            "chunk_size": chunk_size,
            "n_layer": n_layer,
        },
        "final_model.pt"
    )

    # =========================
    # GENERATION
    # =========================

    start_ids = encode("hello")

    if len(start_ids) == 0:
        start_ids = [sp.bos_id()]

    x = torch.tensor(
        [start_ids],
        dtype=torch.long,
        device=device
    )

    out = model.generate(
        x,
        max_out=150,
        temp=0.8,
        top_k=50
    )

    print(decode(out[0].tolist()))