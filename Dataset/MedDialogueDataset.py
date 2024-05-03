from torch.utils.data import Dataset

import json


class MedDialogueDataset(Dataset):
    def __init__(self, tokenizer, dataset, max_length=512, stride=512):
        self.tokenizer = tokenizer

        with open(dataset, 'r') as f:
            self.dataset = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.inputs = []
        self.labels = []

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        d = self.dataset[idx]
        human_text = d["utterances"][0].split(":")[-1]
        machine_text = d["utterances"][1].split(":")[-1]
        concatenated_text = f'<Human>: {human_text}\n<Machine>: {machine_text}'

        encodings = self.tokenizer(concatenated_text,
                                   padding='max_length',
                                   max_length=self.max_length,
                                   truncation=True,
                                   return_tensors="pt")

        input_ids = encodings.input_ids[0]
        attention_mask = encodings.attention_mask[0]

        target_ids = input_ids.clone()
        target_ids[:-self.stride] = -100

        return {'input_ids': input_ids.flatten(),
                'attention_mask': attention_mask.flatten(),
                'labels': target_ids.flatten()
                }
