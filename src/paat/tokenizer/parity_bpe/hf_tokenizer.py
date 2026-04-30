import os
import json
import argparse
from transformers import PreTrainedTokenizerFast
from tokenizers.pre_tokenizers import Whitespace, ByteLevel
from tokenizers.models import BPE
from tokenizers import Tokenizer, pre_tokenizers


def build_vocab_from_merges(merges):
    """ Creates a vocab file from BPE merge rules.
    Args:
        merges (list): List of BPE merge rules.
    Returns:
        dict: A dictionary representing the vocabulary."""

    if merges[0].startswith("#version:"):
        merges = merges[1:]
    vocab = {}
    for idx, char in enumerate(ByteLevel.alphabet()):
        vocab[char] = idx

    index = len(vocab)
    for line in merges:
        token1, token2 = line.split()
        token1 = token1.strip()
        token2 = token2.strip()
        if token1 not in vocab:
            print(f"{token1} is not in the vocab!!!")
        if token2 not in vocab:
            print(f"{token2} is not in the vocab!!!")
        vocab[token1 + token2] = index
        index += 1
    return vocab

def load_custom_tokenizer(tokenizer_path: str):
    """
    Load a custom tokenizer from the given path.
    This function is used to load the tokenizer for the model.
    """
    merge_file = os.path.join(tokenizer_path, "merges.txt")
    vocab_file = os.path.join(tokenizer_path, "vocab.json")
    tokenizer = Tokenizer(BPE(vocab=vocab_file, merges=merge_file))
    wrapped_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="<unk>",
    pad_token="<pad>",
    bos_token="<s>",
    eos_token= "</s>",
    )
    pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), ByteLevel(use_regex=False)])
    wrapped_tokenizer.pre_tokenizer = pre_tokenizer
    special_tokens = ["<s>", "</s>", "<unk>", "<pad>"]
    wrapped_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    return wrapped_tokenizer


def create_huggingface_tokenizer(merges_file_path, tokenizer_path):
    """ Create a HuggingFace tokenizer from a merges file.
    Args:
        merges_file_path (str): Path to the merges file.
        tokenizer_path (str): Path to save the tokenizer files.
    Returns:
        PreTrainedTokenizerFast: A HuggingFace tokenizer.
    """
    
    merges = []
    with open(merges_file_path, "r", encoding="utf-8") as f:
        merges = f.readlines()

    vocab = build_vocab_from_merges(merges)
    if not os.path.exists(tokenizer_path):
        os.makedirs(tokenizer_path)

    vocab_file_path = os.path.join(tokenizer_path, "vocab.json")
    with open(vocab_file_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    
    merges_file_path = os.path.join(tokenizer_path, "merges.txt")
    with open(merges_file_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write(merge)
    tokenizer = load_custom_tokenizer(tokenizer_path)
    return tokenizer


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Create a HuggingFace tokenizer from a merges file.")
    parser.add_argument("--merges_file_path", type=str, required=True, help="Path to the merges file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to save the tokenizer files.")
    args = parser.parse_args()
    
    tokenizer = create_huggingface_tokenizer(args.merges_file_path, args.tokenizer_path)
    os.makedirs(args.tokenizer_path, exist_ok=True)
    tokenizer.save_pretrained(args.tokenizer_path)

    print(f"Tokenizer created and saved to {args.tokenizer_path}")
    print("You can now use this tokenizer with HuggingFace models.")