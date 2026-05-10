import os

import sentencepiece as spm

# Edit these if needed.
INPUT_FILE = "cleaned_pasta.txt"
MODEL_PREFIX = "tokenizer"
VOCAB_SIZE = 8000
MODEL_TYPE = "bpe"
CHARACTER_COVERAGE = 1.0


def main():
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    spm.SentencePieceTrainer.Train(
        input=INPUT_FILE,
        model_prefix=MODEL_PREFIX,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=CHARACTER_COVERAGE,
    )

    print(f"done: wrote {MODEL_PREFIX}.model and {MODEL_PREFIX}.vocab")


if __name__ == "__main__":
    main()
