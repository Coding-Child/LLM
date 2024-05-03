import os
import random
import numpy as np

import torch
import torch.nn as nn

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

from model.LLM import LLM
from Dataset.MedDialogueDataset import preprocess_data

os.environ["WANDB_PROJECT"] = "llm_training"
os.environ["WANDB_LOG_MODEL"] = "checkpoints"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    criterion = nn.CrossEntropyLoss()

    loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
    perplexity = torch.exp(torch.tensor(loss)).item()

    return {'loss': loss.item(), 'ppl': perplexity}


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
        tokenizer.pad_token = tokenizer.eos_token
    except:
        name = 'meta-llama/' + model_name.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name)
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.save_pretrained(model_name)

    files = {'train': train_path, 'validation': val_path, 'test': test_path}
    dataset = load_dataset('json', data_files=files)
    dataset = dataset.map(preprocess_data, fn_kwargs={'tokenizer': tokenizer},
                          remove_columns=['description', 'utterances'])

    train_data = dataset['train'].remove_columns(['labels']) if 'labels' in dataset['train'].column_names else dataset['train']
    val_data = dataset['validation'].remove_columns(['labels']) if 'labels' in dataset['validation'].column_names else dataset['validation']
    test_data = dataset['test'].remove_columns(['labels']) if 'labels' in dataset['test'].column_names else dataset['test']

    print('Train data size:', len(train_data))
    print('Validation data size:', len(val_data))
    print('Test data size:', len(test_data))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

    training_args = TrainingArguments(per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      evaluation_strategy='epoch',
                                      logging_dir='./logs',
                                      logging_steps=1000,
                                      warmup_steps=500,
                                      save_strategy='epoch',
                                      save_total_limit=1,
                                      learning_rate=lr,
                                      num_train_epochs=num_epochs,
                                      gradient_accumulation_steps=accumulation_step,
                                      report_to='wandb',
                                      seed=args.seed,
                                      load_best_model_at_end=True,
                                      metric_for_best_model='ppl',
                                      greater_is_better=False,
                                      output_dir=save_path
                                      )

    trainer = Trainer(model=model,
                      args=training_args,
                      data_collator=data_collator,
                      train_dataset=train_data,
                      eval_dataset=val_data,
                      compute_metrics=compute_metrics
                      )

    trainer.train()
    model.llm.save_pretrained(save_path)
    result = trainer.evaluate(test_data)
    print(result)
