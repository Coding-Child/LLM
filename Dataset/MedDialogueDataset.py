import torch


def preprocess_data(data_point, tokenizer):
    context = data_point['utterances'][0].split(":")[-1]
    response = data_point['utterances'][1].split(":")[-1]

    prompt = f'<Human>: {context}\n<Machine>: {response}'.strip()
    tokenized_full_prompt = tokenizer(prompt, padding=True, truncation=True)

    input_ids_list = tokenized_full_prompt.input_ids
    labels = input_ids_list.copy()

    machine_token_id = 29076
    machine_token_index = input_ids_list.index(machine_token_id)

    machine_end_index = machine_token_index + 2

    labels[:machine_end_index] = [-100] * machine_end_index

    return {
        'input_ids': torch.tensor(input_ids_list),
        'attention_mask': torch.tensor(tokenized_full_prompt.attention_mask),
        'labels': torch.tensor(labels)
    }
