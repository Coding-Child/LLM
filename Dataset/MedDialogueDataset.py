def generate_prompt(data_point):
    """
    Generate a formatted prompt for fine-tuning from a data_point
    :param data_point: a dictionary containing the context and response
    :return: a string containing the formatted prompt
    """

    formated_prompt = ''

    for data in data_point:
        speaker, speech = data.split(':', 1)
        speaker = speaker.strip().lower()
        speech = speech.strip()

        if speaker == 'doctor':
            tag = ' </INST> '
        else:
            tag = ' <INST> '

        formated_prompt += f'{tag}{speech}'

    return formated_prompt.strip()


def generate_prompt_in_batch(data_point):
    contexts = data_point['utterances']
    prompts = []

    for data in contexts:
        prompt = generate_prompt(data)
        prompts.append(prompt)

    return prompts
