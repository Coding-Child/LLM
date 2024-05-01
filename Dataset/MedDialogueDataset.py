def preprocess_data(data_point, tokenizer):
    context = data_point['utterances'][0].split(":")[-1]
    response = data_point['utterances'][1].split(":")[-1]

    prompt = f'<Human>: {context}\n<Machine>: {response}'.strip()
    tokenized_full_prompt = tokenizer(prompt, padding=True, truncation=True)

    labels = tokenized_full_prompt.input_ids.clone()
    machine_token_index = tokenizer.encode('<Machine>:', add_special_tokens=False)

    response_start = tokenized_full_prompt.input_ids[0].tolist().index(machine_token_index[-1]) + 1
    labels[0, :response_start] = -100

    return {'input_ids': tokenized_full_prompt.input_ids.flatten(),
            'attention_mask': tokenized_full_prompt.attention_mask.flatten(),
            'labels': labels.flatten()}
