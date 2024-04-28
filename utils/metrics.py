from datasets import load_metric


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions
    preds = preds.argmax(-1)

    bleu = load_metric('bleu')

    bleu_score = bleu.compute(predictions=preds, references=labels, trust_remote_code=True)
    
    return {'bleu': bleu_score['bleu']}
