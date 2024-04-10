import argparse

import spacy

import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from tqdm import trange

from model import Encoder, Decoder, Seq2Seq
from dataloader import get_dataset, get_dataloader
from data_preprocessing import tokenize_ds, build_vocab, numericalize_vocab

def main():
    seed = 4488

    en_nlp = spacy.load("en_core_web_sm") # python -m spacy download en_core_news_sm
    fr_nlp = spacy.load("fr_core_news_sm") # python -m spacy download fr_core_news_sm

    sos_token = "<sos>"
    eos_token = "<eos>"
    unk_token = "<unk>"
    pad_token = "<pad>"
    max_length = 1_000
    
    dataset = get_dataset(path="./data", data_files="en-fr.txt", seed=seed)

    train_data, valid_data, test_data = (
        dataset["train"],
        dataset["validation"],
        dataset["test"]
    )

    fn_kwargs = {
        "en_nlp": en_nlp,
        "fr_nlp": fr_nlp,
        "max_length": max_length,
        "sos_token": sos_token,
        "eos_token": eos_token
    }

    train_data = train_data.map(tokenize_ds, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(tokenize_ds, fn_kwargs=fn_kwargs)
    test_data = test_data.map(tokenize_ds, fn_kwargs=fn_kwargs)

    special_tokens = [
        unk_token,
        pad_token,
        sos_token,
        eos_token
    ]

    en_vocab, fr_vocab = build_vocab(train_data=train_data, min_freq=2, special_tokens=special_tokens)

    assert en_vocab[unk_token] == fr_vocab[unk_token]
    assert en_vocab[pad_token] == fr_vocab[pad_token]
    unk_index = en_vocab[unk_token]
    pad_index = en_vocab[pad_token]
    en_vocab.set_default_index(unk_index)
    fr_vocab.set_default_index(pad_index)

    fn_kwargs = {
        "en_vocab": en_vocab, 
        "fr_vocab": fr_vocab
    }
    
    train_data = train_data.map(numericalize_vocab, fn_kwargs=fn_kwargs)
    valid_data = valid_data.map(numericalize_vocab, fn_kwargs=fn_kwargs)
    test_data = test_data.map(numericalize_vocab, fn_kwargs=fn_kwargs)

    datatype = "torch"
    columns = ["en_ids", "fr_ids"]

    train_data = train_data.with_format(type=datatype, columns=columns, output_all_columns=True)
    valid_data = valid_data.with_format(type=datatype, columns=columns, output_all_columns=True)
    test_data = test_data.with_format(type=datatype, columns=columns, output_all_columns=True)

    batch_size = 128

    train_dataloader = get_dataloader(train_data, batch_size, pad_index, shuffle=True)
    valid_dataloader = get_dataloader(valid_data, batch_size, pad_index)
    test_dataloader = get_dataloader(test_data, batch_size, pad_index)
    
    encoder = Encoder(input_vocab=len(en_vocab), embedding_dim=256, hidden_dim=512, num_layers=4, dropout=0.5)
    decoder = Decoder(output_vocab=len(fr_vocab), embedding_dim=256, hidden_dim=512, num_layers=4, dropout=0.5)
    model = Seq2Seq(encoder=encoder, decoder=decoder, device=device).to(device)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    lr = 0.7
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_index)
    optimizer = torch.optim.SGD(model.parameters, momentum=0.9, lr=lr)
    scheduler = LambdaLR(optimizer, lambda epoch: 1.0 if epoch < 5 else 0.5 ** ((epoch - 5) / 0.5))

    epochs = 7
    tf_ratio = 0.5
    grad_clip_norm = 5

    for epoch in trange(epochs):
        model.train()
        train_loss = 0
        for X in enumerate(train_dataloader):
            src = X["en_ids"].to(device)
            trg = X["fr_ids"].to(device)
            output = model(src, trg, tf_ratio)
            output = output[:, 1:].reshape(-1, output.shape[-1])
            trg = trg[:, 1:].reshape(-1)
            loss = loss_fn(output, trg)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        scheduler.step()

        model.eval()
        with torch.inference_mode():
            valid_loss = 0
            for X in enumerate(valid_dataloader):
                src = X["en_ids"].to(device)
                trg = X["fr_ids"].to(device)
                output = model(src, trg, 0)
                output = output[:, 1:].reshape(-1, output.shape[-1])
                trg = trg[:, 1:].reshape(-1)
                valid_loss += loss.item()

        print(f"Train Loss: {train_loss} | Valid Loss: {valid_loss}")
    
if __name__ == "__main__":
    main()