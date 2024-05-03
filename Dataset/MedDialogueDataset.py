import torch


def generate_and_tokenize_prompt(data_point, tokenizer):
    full_prompt = f"<Human>: {data_point[0].split(':')[-1]}\n<AI>: {data_point[1].split(':')[-1]}".strip()
    tokenized_full_prompt = tokenizer(full_prompt, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
    input_ids = tokenized_full_prompt['input_ids'].flatten()
    attention_mask = tokenized_full_prompt['attention_mask'].flatten()
    labels = input_ids.clone()

    ai_tokens = tokenizer.tokenize("\n<AI>:")
    ai_token_id = tokenizer.convert_tokens_to_ids(ai_tokens)[-1]

    indices = [i for i, token in enumerate(input_ids.tolist()) if token == ai_token_id]
    if len(indices) < 2:
        raise ValueError("Not enough AI tokens found in prompt. Check the data formatting.")

    second_occurrence_idx = indices[1]
    labels[:second_occurrence_idx + 1] = torch.full((second_occurrence_idx + 1,), -100, dtype=torch.long)

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


def batch_generate_data(batch, tokenizer):
    inputs = batch['utterances']

    output_dict = {'input_ids': [], 'attention_mask': [], 'labels': []}

    for data_point in inputs:
        result = generate_and_tokenize_prompt(data_point, tokenizer)
        output_dict['input_ids'].append(result['input_ids'])
        output_dict['attention_mask'].append(result['attention_mask'])
        output_dict['labels'].append(result['labels'])

    return output_dict
