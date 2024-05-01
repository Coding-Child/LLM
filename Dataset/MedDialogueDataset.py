def preprocess_data(data_point, tokenizer):
    context = data_point['utterances'][0].split(":")[-1]
    response = data_point['utterances'][1].split(":")[-1]

    prompt = f'<Human>: {context}\n<Machine>: {response}'.strip()
    tokenized_full_prompt = tokenizer(prompt, padding=True, truncation=True, return_tensors='pt')

    labels = tokenized_full_prompt.input_ids.clone()

    machine_token_id = 29076
    machine_token_index = (labels == machine_token_id).nonzero(as_tuple=True)[0].item()
    machine_end_index = machine_token_index + 2

    labels[:machine_end_index] = -100

    return {'input_ids': tokenized_full_prompt.input_ids,
            'attention_mask': tokenized_full_prompt.attention_mask,
            'labels': labels}
