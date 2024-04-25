from datasets import load_metric


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions

    bleu = load_metric('bleu')
    rouge = load_metric('rouge')

    bleu_score = bleu.compute(predictions=preds, references=labels)
    rouge_score = rouge.compute(predictions=preds, references=labels)

    return {'bleu': bleu_score, 'rouge': rouge_score}
