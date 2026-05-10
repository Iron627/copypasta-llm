import torch
import torch.nn as nn
import torch.nn.functional as F

from tokenizers import Tokenizer

# =========================
# DEVICE
# =========================

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# TOKENIZER
# =========================

tokenizer = Tokenizer.from_file("tokenizer.json")

encode = lambda s: tokenizer.encode(s).ids
decode = lambda ids: tokenizer.decode(ids)

vocab_size = tokenizer.get_vocab_size()

# =========================
# CONFIG
# =========================

batch_size = 16
chunk_size = 256

n_embd = 512
n_layer = 8
num_heads = 8

dropout = 0.2

train_split = None
test_split = None

# =========================
# DATASET
# =========================

def load_dataset(path="cleaned_pasta.txt"):
    global train_split, test_split

    if train_split is not None and test_split is not None:
        return train_split, test_split

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    data = torch.tensor(
        encode(text),
        dtype=torch.long,
        device=device
    )

    split_idx = int(0.9 * len(data))

    train_split = data[:split_idx]
    test_split = data[split_idx:]

    return train_split, test_split

def get_chunk(split):
    train_data, test_data = load_dataset()

    source = train_data if split == "train" else test_data

    ix = torch.randint(
        len(source) - chunk_size - 1,
        (batch_size,),
        device=device
    )

    offsets = torch.arange(chunk_size, device=device)

    x = source[ix[:, None] + offsets[None, :]]
    y = source[ix[:, None] + offsets[None, :] + 1]

    return x, y

# =========================
# TRANSFORMER BLOCK
# =========================

class Block(nn.Module):
    def __init__(self, n_embd, num_heads=8):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = n_embd // num_heads

        self.ln1 = nn.LayerNorm(n_embd)

        # FLASH ATTENTION SETUP
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)

        self.dropout = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(n_embd)

        self.ff = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        B, T, C = x.shape

        norm_x = self.ln1(x)

        # =========================
        # FLASH ATTENTION
        # =========================

        qkv = self.qkv(norm_x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        k = k.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        v = v.view(
            B, T, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=dropout if self.training else 0.0
        )

        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.view(B, T, C)

        attn_out = self.proj(attn_out)

        x = x + self.dropout(attn_out)

        # =========================
        # FEEDFORWARD
        # =========================

        ff_out = self.ff(self.ln2(x))

        x = x + self.dropout(ff_out)

        return x

# =========================
# MODEL
# =========================

class transformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_embd=512,
        chunk_size=256,
        n_layer=8
    ):
        super().__init__()

        self.chunk_size = chunk_size

        self.token_embedding_table = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding_table = nn.Embedding(
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

        # WEIGHT TYING
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, inputs, targets=None):
        B, T = inputs.shape

        token_emb = self.token_embedding_table(inputs)

        pos_emb = self.position_embedding_table(
            torch.arange(T, device=inputs.device)
        )

        x = token_emb + pos_emb

        for block in self.blocks:
            x = block(x)

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
    def generate(
        self,
        inputs,
        max_out,
        temp=0.8,
        top_k=50
    ):
        self.eval()

        for _ in range(max_out):

            inputs_cond = inputs[:, -self.chunk_size:]

            logits, _ = self(inputs_cond)

            logits = logits[:, -1, :]

            logits = logits / max(temp, 1e-6)

            top_k = min(top_k, logits.size(-1))

            v, ix = torch.topk(logits, top_k)

            probs = F.softmax(v, dim=-1)

            sampled = torch.multinomial(
                probs,
                num_samples=1
            )

            out_token = ix.gather(1, sampled)

            inputs = torch.cat(
                (inputs, out_token),
                dim=1
            )

        return inputs

# =========================
# LOSS ESTIMATION
# =========================

@torch.no_grad()
def estimate_loss(model):
    model.eval()

    losses = {
        "train": 0.0,
        "test": 0.0
    }

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

# =========================
# TRAINING
# =========================

if __name__ == "__main__":

    model = transformerLM(
        vocab_size,
        n_embd=n_embd,
        chunk_size=chunk_size,
        n_layer=n_layer
    ).to(device)

    # PYTORCH 2 SPEEDUP
    model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=0.1
    )

    print(
        sum(p.numel() for p in model.parameters()) / 1e6,
        "M parameters"
    )

    for timesteps in range(50000):

        xtrain, ytrain = get_chunk("train")

        logits, loss = model(xtrain, ytrain)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        # GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        if timesteps % 100 == 0:
            print(
                f"step {timesteps} | loss {loss.item():.4f}"
            )

        if timesteps % 1000 == 0:

            losses = estimate_loss(model)

            print(
                f"step {timesteps}: "
                f"train {losses['train']:.4f}, "
                f"test {losses['test']:.4f}"
            )

        if timesteps % 5000 == 0 and timesteps > 0:

            torch.save({
                "model_state": model.state_dict(),
                "vocab_size": vocab_size,
                "n_embd": n_embd,
                "chunk_size": chunk_size,
                "n_layer": n_layer,
            }, f"transformer_checkpoint_{timesteps}.pt")

    torch.save({
        "model_state": model.state_dict(),
        "vocab_size": vocab_size,
        "n_embd": n_embd,
        "chunk_size": chunk_size,
        "n_layer": n_layer,
    }, "transformer_checkpoint.pt")

    # =========================
    # GENERATION
    # =========================

    start_text = "<|user|> hello\n<|assistant|>"

    start = torch.tensor(
        [encode(start_text)],
        dtype=torch.long,
        device=device
    )

    out = model.generate(
        start,
        max_out=150,
        temp=0.8,
        top_k=50
    )

    print(decode(out[0].tolist()))