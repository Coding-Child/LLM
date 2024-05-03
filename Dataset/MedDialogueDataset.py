def generate_prompt(data_point):
    return f"""
<Human>: {data_point['utterances'][0].split(':')[-1]}
<AI>: {data_point['utterances'][1].split(':')[-1]}
  """.strip()


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    tokenized_full_prompt['input_ids'] = tokenized_full_prompt['input_ids'].flatten()
    tokenized_full_prompt['attention_mask'] = tokenized_full_prompt['attention_mask'].flatten()

    return tokenized_full_prompt
