def generate_prompt(data_point):
    return f"""
<Human>: {data_point['utterances'][0].split(':')[-1]}
<AI>: {data_point['utterances'][1].split(':')[-1]}
  """.strip()


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)

    return tokenized_full_prompt
