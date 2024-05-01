import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset, load_metric

import os
import random
import numpy as np

from model.LLM import LLM
from Dataset.MedDialogueDataset import preprocess_data

os.environ["WANDB_PROJECT"] = "llm_training"
os.environ["WANDB_LOG_MODEL"] = "checkpoints"


def compute_metrics(eval_pred):
    f1_metric = load_metric('f1')
    bleu_metric = load_metric('bleu')

    logits, labels = eval_pred
    predictions = logits.argmax(-1)

    bleu = bleu_metric.compute(predictions=[predictions], references=[labels])['bleu']
    f1 = f1_metric.compute(predictions=predictions, references=labels)['f1']

    return {'f1': f1, 'bleu': bleu}


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
    num_epochs = args.num_epochs
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    save_path = args.save_path
    r = args.r
    lora_dropout = args.lora_dropout
    lora_alpha = args.lora_alpha

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LLM(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, model_name=model_name)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, device_ids=[0, 1])
    model = model.to(device)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    except:
        name = 'meta-llama/' + model_name.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name)
        tokenizer.pad_token = tokenizer.eos_token

        tokenizer.save_pretrained(model_name)

    files = {'train': train_path, 'validation': val_path, 'test': test_path}
    dataset = load_dataset("json", data_files=files)
    dataset = dataset.map(lambda x: preprocess_data(x, tokenizer),
                          remove_columns=['description', 'utterances'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

    print("Training set size:", len(dataset['train']))
    print("Validation set size:", len(dataset['validation']))
    print("Test set size:", len(dataset['test']))

    training_args = TrainingArguments(auto_find_batch_size=True,
                                      num_train_epochs=num_epochs,
                                      learning_rate=lr,
                                      bf16=True,
                                      save_total_limit=5,
                                      logging_steps=10,
                                      output_dir=save_path,
                                      logging_dir='./logs',
                                      save_strategy='epoch',
                                      evaluation_strategy='epoch',
                                      remove_unused_columns=False,
                                      gradient_accumulation_steps=4,
                                      label_names=['labels'],
                                      load_best_model_at_end=True,
                                      metric_for_best_model='f1',
                                      greater_is_better=True,
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
    model.llm.config.use_cache = False
    trainer.train()
    model.save_pretrained(save_path)

    result = trainer.evaluate(dataset['test'])
    print(result)
