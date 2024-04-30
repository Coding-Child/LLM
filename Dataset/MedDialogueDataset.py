def preprocess_data(data_point, tokenizer):
    context = data_point['utterances'][0].split(":")[-1]
    response = data_point['utterances'][1].split(":")[-1]

    prompt = f'<Human>: {context}\n<Machine>: {response}'.strip()
    tokenized_full_prompt = tokenizer(prompt, padding=True, truncation=True)

    return tokenized_full_prompt
