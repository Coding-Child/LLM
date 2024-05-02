import torch

from evaluate import load
from scripts.eval import evaluation

import wandb
from tqdm import tqdm


def train(model, device, train_loader, val_loader, optimizer, scheduler, num_epochs, save_path):
    wandb.init(config={"learning_rate": scheduler.get_last_lr()[0],
                       "epochs": num_epochs,
                       "batch_size": train_loader.batch_size
                       })

    f1 = load('f1')
    train_loss_arr = list()
    val_loss_arr = list()
    min_val_loss = float('inf')
    step = 0

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        preds = list()
        labels = list()
        nlls = list()

        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='b', ascii=True, ncols=150) as pbar:
            for batch in train_loader:
                batch = {k: v.to(device) for k, v in batch.items()}

                output = model(batch)

                loss = output.loss
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()

                wandb.log({"step_train_loss": loss.item(), "global_step": step})

                # TODO: Implement perplexity calculation
                nlls.append(loss.item())

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
                step += 1

                del output, loss, batch
                torch.cuda.empty_cache()

        f1_score = f1.compute(predictions=preds, references=labels, average='macro')['f1']
        ppl = torch.exp(torch.stack(nlls).mean())

        avg_loss = total_loss / len(train_loader)
        train_loss_arr.append(avg_loss)

        wandb.log({'epoch': epoch + 1,
                   'epoch_train_loss': avg_loss,
                   'train_perplexity': ppl.item(),
                   'train_f1': f1_score,
                   "global_step": step
                   })

        val_loss, val_ppl, val_f1 = evaluation(model, device, val_loader)
        val_loss_arr.append(val_loss)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            model.save_pretrained(save_path + '/best_model')

            print('best model saved')
            print(f'train loss: {avg_loss:.4f}, perplexity: {ppl:.4f}, f1: {f1_score:.4f}')
            print(f'validation loss: {val_loss:.4f}, perplexity: {val_ppl:.4f}, f1: {val_f1:.4f}')

        wandb.log({'epoch': epoch + 1,
                   'validation_loss': val_loss,
                   'val_perplexity': val_ppl,
                   'val_f1': val_f1,
                   "global_step": step
                   })

    model.save_pretrained(save_path + '/last_model')
    wandb.finish()

    return train_loss_arr, val_loss_arr
