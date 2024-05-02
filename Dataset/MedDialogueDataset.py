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

        self._preprocess()

    def _preprocess(self):
        for d in self.dataset:
            human_text = d["utterances"][0].split(":")[-1]
            machine_text = d["utterances"][1].split(":")[-1]
            concatenated_text = f'<Human>: {human_text}\n<Machine>: {machine_text}'

            encodings = self.tokenizer(concatenated_text,
                                       padding='max_length',
                                       max_length=self.max_length,
                                       truncation=True,
                                       return_tensors="pt")

            for i in range(0, encodings.input_ids.size(1), self.stride):
                begin_loc = max(i + self.stride - self.max_length, 0)
                end_loc = min(i + self.stride, encodings.input_ids.size(1))
                trg_len = end_loc - i

                input_ids = encodings.input_ids[:, begin_loc: end_loc]
                attention_mask = encodings.attention_mask[:, begin_loc: end_loc]

                target_ids = input_ids.clone()
                target_ids[:, :-trg_len] = -100

                self.inputs.append({'end_loc': end_loc, 'input_ids': input_ids.flatten(), 'attention_mask': attention_mask.flatten()})
                self.labels.append(target_ids.flatten())

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return {'input_ids': self.inputs[idx]['input_ids'],
                'attention_mask': self.inputs[idx]['attention_mask'],
                'labels': self.labels[idx]
                }
