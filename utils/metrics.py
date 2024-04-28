from evaluate import load


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = preds.argmax(-1)

    bleu = load('bleu')
    rouge = load('rouge')

    bleu_score = bleu.compute(predictions=preds, references=labels)['bleu']
    rouge_score = rouge.compute(predictions=preds, references=labels)['rougeL']

    return {'bleu': bleu_score, 'rouge': rouge_score}
