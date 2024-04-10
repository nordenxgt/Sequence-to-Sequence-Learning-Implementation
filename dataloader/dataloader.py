import torch
from torch import nn

def get_collate_fn(pad_index):
    def collate_fn(batch):
        return {
            "en_ids": nn.utils.rnn.pad_sequence(
                [vocab["en_ids"] for vocab in batch], 
                padding_value=pad_index,
                batch_first=True
            ),
            "fr_ids": nn.utils.rnn.pad_sequence(
                [vocab["fr_ids"] for vocab in batch], 
                padding_value=pad_index,
                batch_first=True
            )
        }
    return collate_fn

def get_dataloader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle
    )