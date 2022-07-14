from enum import Enum

import torch
from transformers import DistilBertTokenizer


class Mode(Enum):
    TRAIN = 0
    TEST_ID = 1
    TEST_OOD = 2

def initialize_distilbert_transform(max_token_length):
    """Adapted from the Wilds library, available at: https://github.com/p-lambda/wilds"""
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    def transform(text):
        tokens = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_token_length,
            return_tensors='pt')
        x = torch.stack((tokens['input_ids'], tokens['attention_mask']), dim=2)
        x = torch.squeeze(x, dim=0) # First shape dim is always 1
        return x
    return transform