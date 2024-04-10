from typing import List
from datasets import Dataset
from torchtext.vocab import build_vocab_from_iterator
from unidecode import unidecode

def split_texts(rows):
    return {
        "en": [text.split("\t")[0] for text in rows["text"]],
        "fr": [text.split("\t")[1] for text in rows["text"]],
    }

def clean_text(batch):
    return {k: [unidecode(s) for s in v] for k, v in batch.items()}

def tokenize_ds(data, en_nlp, fr_nlp, max_length, sos_token, eos_token):
    return {
        "en_tokens": [sos_token] + [token.text.lower() for token in en_nlp.tokenizer(data["en"])][:max_length] + [eos_token], 
        "fr_tokens": [sos_token] + [token.text.lower() for token in fr_nlp.tokenizer(data["fr"])][:max_length] + [eos_token]
    }

def build_vocab(train_data: Dataset, min_freq: int = 2, special_tokens: List[str] = None):
    en_vocab = build_vocab_from_iterator(
        train_data["en_tokens"],
        min_freq=min_freq,
        specials=special_tokens
    )

    fr_vocab = build_vocab_from_iterator(
        train_data["fr_tokens"],
        min_freq=min_freq,
        specials=special_tokens
    )

    return en_vocab, fr_vocab

def numericalize_vocab(vocab: str, en_vocab: build_vocab_from_iterator, fr_vocab: build_vocab_from_iterator):
    return {
        "en_ids": en_vocab.lookup_indices(vocab["en_tokens"]),
        "fr_ids": fr_vocab.lookup_indices(vocab["fr_tokens"])
    }