import json
import torch
from torch.utils.data import Dataset


def preprocess_data(data_point, tokenizer, max_length=512):
    context = data_point['utterances'][0].split(":")[-1]
    response = data_point['utterances'][1].split(":")[-1]

    contexts = tokenizer.encode_plus(context,
                                     padding='max_length',
                                     truncation=True,
                                     max_length=max_length,
                                     return_tensors='pt'
                                     )

    labels = tokenizer.encode_plus(response,
                                   padding='max_length',
                                   truncation=True,
                                   max_length=max_length,
                                   return_tensors='pt'
                                   )['input_ids']

    encoded = {'input_ids': contexts['input_ids'].squeeze(0),
               'attention_mask': contexts['attention_mask'].squeeze(0),
               'labels': labels.squeeze(0)
               }

    return encoded
