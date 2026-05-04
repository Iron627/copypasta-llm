import torch

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

chunk_size = 10
train_data = torch.tensor(encode(train_split), dtype=torch.long)