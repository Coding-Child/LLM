def generate_prompt(data_point):
    """
    Generate a formatted prompt for fine-tuning from a data_point
    :param data_point: a dictionary containing the context and response
    :return: a string containing the formatted prompt
    """

    context = data_point[0].split(":")[-1].strip()
    response = data_point[1].split(":")[-1].strip()

    prompt = f"<INST> {context} </INST> {response}"

    return prompt


def generate_prompt_batched(data_point):
    contexts = data_point['utterances']
    prompts = []

    for data in contexts:
        prompt = generate_prompt(data)
        prompts.append(prompt)

    return prompts
