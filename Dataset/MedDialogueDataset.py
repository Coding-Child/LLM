import torch


def generate_prompt(data_point):
    return f"<Human>: {data_point['utterances'][0].split(':')[-1]}\n<AI>: {data_point['utterances'][1].split(':')[-1]}".strip()


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = generate_prompt(data_point)

    tokenized_full_prompt = tokenizer(full_prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    input_ids = tokenized_full_prompt['input_ids'].flatten()
    attention_mask = tokenized_full_prompt['attention_mask'].flatten()
    labels = input_ids.clone()

    ai_tokens = tokenizer.tokenize("\n<AI>:")
    ai_token_id = tokenizer.convert_tokens_to_ids(ai_tokens)[-1]

    idx = input_ids.tolist().index(ai_token_id)
    labels[:idx + 1] = torch.full((idx + 1,), -100, dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def generate_data(data_point, tokenizer):
    data = generate_and_tokenize_prompt(data_point, tokenizer)

    return data
