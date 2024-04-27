import torch
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import load_dataset

import os
import random
import numpy as np

from model.LLM import LLM
from Dataset.MedDialogueDataset import preprocess_data
from utils.metrics import compute_metrics
os.environ["WANDB_PROJECT"] = "llm_training"
os.environ["WANDB_LOG_MODEL"] = "checkpoints"


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
    max_length = args.max_len
    num_epochs = args.num_epochs
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    save_path = args.save_path
    r = args.r
    lora_dropout = args.lora_dropout
    lora_alpha = args.lora_alpha
    
    model = LLM(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, model_name=model_name)

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
    dataset = dataset.map(lambda x: preprocess_data(x, tokenizer, max_length), remove_columns=['description', 'utterances'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

    print("Training set size:", len(dataset['train']))
    print("Validation set size:", len(dataset['validation']))
    print("Test set size:", len(dataset['test']))

    training_args = TrainingArguments(auto_find_batch_size=True,
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
                                      remove_unused_columns=False,
                                      report_to="wandb"
                                      )
    trainer = Trainer(model=model,
                      args=training_args,
                      train_dataset=dataset['train'],
                      eval_dataset=dataset['validation'],
                      tokenizer=tokenizer,
                      compute_metrics=compute_metrics,
                      data_collator=data_collator
                      )
    model.llm.config.use_cache = False
    trainer.train()
    trainer.evaluate(dataset['test'])
