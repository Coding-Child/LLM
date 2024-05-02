import torch
import torch.nn.functional as F

from evaluate import load
from peft import PeftModel

import numpy as np
from tqdm import tqdm


def evaluation(model, device, val_loader):
    f1 = load('f1')

    total_loss = 0
    total_log_prob = torch.tensor(0.0, device=device)
    total_tokens = 0
    all_preds = list()
    all_labels = list()

    model.eval()
    with tqdm(val_loader, desc='Validation', unit='b', ascii=True, ncols=150) as pbar:
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)

                logit = output.logits
                loss = output.loss
                total_loss += loss.item()

                # TODO: Implement perplexity calculation
                shifted_logit = logit[..., :-1, :].contiguous()
                shifted_labels = batch['input_ids'][..., 1:].contiguous()

                log_probs = F.log_softmax(shifted_logit, dim=-1)
                label_log_probs = log_probs.gather(dim=-1, index=shifted_labels.unsqueeze(-1)).squeeze(-1)

                total_log_prob += label_log_probs.sum()
                total_tokens += shifted_labels.numel()

                # TODO: Implement Macro F1-score calculation
                pred = output.logits.argmax(-1).detach().cpu().numpy()
                all_preds.append(pred)
                all_labels.append(shifted_labels.detach().cpu().numpy())

                pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                pbar.update(1)

    f1_score = f1.compute(predictions=np.concatenate(all_preds), references=np.concatenate(all_labels), average='macro')['f1']
    ppl = torch.exp(-total_log_prob / total_tokens)
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
