import os
import random
import numpy as np
import torch

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.metrics import f1_score

from model.LLM import LLM
from Dataset.MedDialogueDataset import batch_generate_data

os.environ["WANDB_PROJECT"] = "llm_training"
os.environ["WANDB_LOG_MODEL"] = "checkpoints"


def compute_metrics(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)

    valid_positions = (labels != -100) & (labels != 0)

    valid_preds = preds[valid_positions]
    valid_labels = labels[valid_positions]

    f1 = f1_score(y_true=valid_labels, y_pred=valid_preds, average='macro')

    return {'f1': f1}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(args.seed)

    model_name = args.model_name
    lr = args.learning_rate
    batch_size = args.batch_size
    accumulation_step = args.accumulation_step
    num_epochs = args.num_epochs
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    save_path = args.save_path
    r = args.r
    lora_dropout = args.lora_dropout
    lora_alpha = args.lora_alpha

    model = LLM(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, model_name=model_name)
    model.llm.print_trainable_parameters()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.unk_token
    except:
        name = 'meta-llama/' + model_name.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name)
        tokenizer.pad_token = tokenizer.unk_token

        tokenizer.save_pretrained(model_name)

    files = {'train': train_path, 'validation': val_path, 'test': test_path}
    dataset = load_dataset('json', data_files=files)
    dataset = dataset.map(batch_generate_data, fn_kwargs={'tokenizer': tokenizer},
                          remove_columns=['description', 'utterances'], batched=True)

    print('Train data size:', len(dataset['train']))
    print('Validation data size:', len(dataset['validation']))
    print('Test data size:', len(dataset['test']))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

    training_args = TrainingArguments(per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_epochs,
                                      learning_rate=lr,
                                      bf16=True,
                                      save_total_limit=4,
                                      logging_steps=10,
                                      output_dir=save_path,
                                      logging_dir='./logs',
                                      save_strategy='epoch',
                                      evaluation_strategy='epoch',
                                      load_best_model_at_end=True,
                                      metric_for_best_model='f1',
                                      greater_is_better=True,
                                      remove_unused_columns=False,
                                      gradient_accumulation_steps=accumulation_step,
                                      report_to="wandb"
                                      )

    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=dataset['train'].shuffle(seed=args.seed),
                      eval_dataset=dataset['validation'],
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      data_collator=data_collator
                      )

    trainer.train()
    model.llm.save_pretrained(save_path)
    result = trainer.evaluate(dataset['test'])
    print(result)
