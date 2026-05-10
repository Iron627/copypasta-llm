from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = ByteLevel()

trainer = BpeTrainer(
    vocab_size=16000,
)

tokenizer.train(
    ["cleaned_pasta.txt"],
    trainer
)

tokenizer.save("tokenizer.json")

print("done")