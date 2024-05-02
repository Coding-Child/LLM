import torch

from evaluate import load
from peft import PeftModel
from tqdm import tqdm


def evaluation(model, device, val_loader):
    f1 = load('f1')

    total_loss = 0
    preds = list()
    labels = list()
    nlls = list()

    model.eval()
    with tqdm(val_loader, desc='Validation', unit='b', ascii=True, ncols=150) as pbar:
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                end_loc = batch['end_loc']

                output = model(input_ids=batch['input_ids'],
                               attention_mask=batch['attention_mask'],
                               labels=batch['labels'])

                loss = output.loss
                total_loss += loss.item()

                # TODO: Implement perplexity calculation
                nlls.append(loss)

                # TODO: Implement Macro F1-score calculation
                output_logits = output.logits.argmax(-1).detach().cpu()

                batch_labels = batch['labels'].detach().cpu()

                valid_indices = batch_labels != -100
                filtered_predictions = output_logits[valid_indices]
                filtered_labels = batch_labels[valid_indices]

                preds += list(filtered_predictions.numpy())
                labels += list(filtered_labels.numpy())

                pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                pbar.update(1)

                del output, loss, batch
                torch.cuda.empty_cache()

    f1_score = f1.compute(predictions=preds, references=labels, average='macro')['f1']
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    avg_loss = total_loss / len(val_loader)

    return avg_loss, ppl, f1_score


def test(model, device, test_loader, save_path):
    best_path = f'{save_path}/best_model'
    final_path = f'{save_path}/last_model'

    model = PeftModel.from_pretrained(model, best_path)
    best_loss, best_ppl, best_f1 = evaluation(model, device, test_loader)

    model = PeftModel.from_pretrained(model, final_path)
    final_loss, final_ppl, final_f1 = evaluation(model, device, test_loader)

    print(f'Best Model Loss: {best_loss:.4f}, PPL: {best_ppl:.4f}, F1: {best_f1:.4f}')
    print(f'Final Model Loss: {final_loss:.4f}, PPL: {final_ppl:.4f}, F1: {final_f1:.4f}')
