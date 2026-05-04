import torch
import torch.nn as nn
import torch.nn.functional as F

with open("cleaned_pasta.txt", "r", encoding="utf-8") as f:
    data = f.read()
vocab = sorted(list(set(data)))


map_s_to_i = {s: i for i, s in enumerate(vocab)}
map_i_to_s = {i: s for i, s in enumerate(vocab)}
encode = lambda s: [map_s_to_i[c] for c in s]
decode = lambda i: "".join([map_i_to_s[i] for i in i])

data = torch.tensor(encode(data), dtype=torch.long)

train_split = data[:int(0.9*len(data))]
test_split = data[int(0.9*len(data)):]

batch_size = 8
chunk_size = 16


# depending on batch size (B) and chunk size (C) returns a random tensor of shape BxC for transformer to train on
def get_chunk(split):
    data = train_split if split == "train" else test_split
    starting_positions = torch.randint(0, len(data) - chunk_size, (batch_size,))
    x = torch.stack([data[pos:pos+chunk_size] for pos in starting_positions])
    y = torch.stack([data[pos+1:pos+chunk_size+1] for pos in starting_positions])
    return x, y


class BigramLM(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) # LUT for token embeddings
    def forward(self, inputs, targets=None):
        logits = self.token_embedding_table(inputs) # (B, T, C)
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b*t,c) 
            targets = targets.view(b*t) 
            
            loss  = F.cross_entropy(logits, targets)
        
        return logits, loss
    def generate(self, inputs, max_out):
        for char in range(max_out):
            logits = self(inputs)
            logits=  logits[:, -1, :] # only take previous char
            probs = F.softmax(logits, dim=-1) # convert to probabilities (activation)
            out_char = torch.multinomial(probs, num_samples=1) # sample from distribution
            inputs = torch.cat((inputs, out_char), dim=1) #catenate 
        return inputs
            