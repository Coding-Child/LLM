import os
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset

from model.LLM import LLM
from Dataset.MedDialogueDataset import preprocess_data
from scripts.train import train
from scripts.eval import test

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
    batch_size = args.batch_size
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
    model.llm.print_trainable_parameters()

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
    dataset = dataset.map(preprocess_data, fn_kwargs={'tokenizer': tokenizer}, remove_columns=['description', 'utterances'])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

    train_loader = DataLoader(dataset['train'],
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=data_collator,
                              num_workers=4,
                              pin_memory=True)
    val_loader = DataLoader(dataset['validation'],
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=data_collator,
                            num_workers=4,
                            pin_memory=True)
    test_loader = DataLoader(dataset['test'],
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=data_collator,
                             num_workers=4,
                             pin_memory=True)

    print("Training set size:", len(dataset['train']))
    print("Validation set size:", len(dataset['validation']))
    print("Test set size:", len(dataset['test']))

    warmup_step = (len(train_loader) * num_epochs) * 0.04
    total_step = len(train_loader) * num_epochs

    optimizer = opt.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=total_step)

    train_loss_arr, val_loss_arr = train(model=model,
                                         device=device,
                                         train_loader=train_loader,
                                         val_loader=val_loader,
                                         optimizer=optimizer,
                                         scheduler=scheduler,
                                         num_epochs=num_epochs,
                                         save_path=save_path)

    plt.plot(train_loss_arr, label='train loss')
    plt.plot(val_loss_arr, label='validation loss')
    plt.legend()
    plt.show()

    test(model, device, test_loader, save_path)
