def generate_prompt(data_point):
    """
    Generate a formatted prompt for fine-tuning from a data_point
    :param data_point: a dictionary containing the context and response
    :return: a string containing the formatted prompt
    """

    prefix_text = "This is conversation from a healthcare chat. Respond empathetically and informatively."
    context = data_point['utterances'][0].split(":")[-1].strip()
    response = data_point['utterances'][1].split(":")[-1].strip()

    prompt = f"<s>[INST] {prefix_text} {context} [/INST]{response}</s>"

    return prompt


def tokenization(examples, tokenizer):
    return tokenizer(examples['prompt'], padding="max_length", truncation=True)
    
