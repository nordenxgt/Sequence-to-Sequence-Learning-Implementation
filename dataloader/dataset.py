import torch
from torch import nn
import random
import numpy as np
import spacy
import datasets
import torchtext
import tqdm

import html
import re
from datasets import Dataset

from data_preprocessing import split_texts, clean_text

def get_dataset(path: str, data_files: str, seed: int) -> Dataset:
    ds = datasets.load_dataset(path=path, data_files=data_files)
    ds = ds.map(split_texts, batched=True).remove_columns("text")
    ds = ds.map(clean_text, batched=True)
    ds = ds["train"].train_test_split(train_size=0.8, seed=seed)
    tvt_ds = ds["train"].train_test_split(train_size=0.8, seed=seed)
    tvt_ds["validation"] = tvt_ds.pop("test")
    tvt_ds["test"] = ds["test"]
    return tvt_ds
