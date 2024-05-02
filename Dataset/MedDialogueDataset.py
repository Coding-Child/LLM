def preprocess_data(data_point, tokenizer):
    context = data_point['utterances'][0].split(":")[-1]
    response = data_point['utterances'][1].split(":")[-1]

    prompt = f'<Human>: {context}\n<Machine>: {response}'.strip()
    tokenized_full_prompt = tokenizer(prompt, padding='max_length', max_length=1024, truncation=True, return_tensors='pt')

    tokenized_full_prompt.labels = tokenized_full_prompt.input_ids.clone()

    return tokenized_full_prompt
