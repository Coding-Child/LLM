import os
import random
import numpy as np
from datetime import datetime

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling, EarlyStoppingCallback, TrainingArguments
from transformers import AdamW, get_scheduler

from peft import PeftModel
from datasets import load_dataset
from trl import SFTTrainer

from model.LLM import load_llm
from Dataset.MedDialogueDataset import generate_prompt
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

    # Set the arguments
    model_name = args.model_name
    lr = args.learning_rate
    batch_size = args.batch_size
    accumulation_step = args.accumulation_step
    warmup_step = args.warmup_step
    num_epochs = args.num_epochs
    train_path = args.train_path
    val_path = args.val_path
    test_path = args.test_path
    save_name = args.save_path + '/' + model_name.split("/")[-1]
    merged_model_name = 'full_model' + '/' + model_name.split("/")[-1]
    r = args.r
    lora_dropout = args.lora_dropout
    lora_alpha = args.lora_alpha

    # Load the model
    model, lora_config = load_llm(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, model_name=model_name)

    trainable, total = model.get_nb_trainable_parameters()
    equal_len = len(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100:.4f}%")
    print('=' * equal_len)
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100:.4f}%")
    print('=' * equal_len)

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  model_max_length=args.max_len,
                                                  add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
    except:
        name = 'meta-llama/' + model_name.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name,
                                                  model_max_length=args.max_len,
                                                  add_eos_token=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(model_name)

    torch.cuda.empty_cache()

    # Load the dataset
    files = {'train': train_path, 'validation': val_path, 'test': test_path}
    dataset = load_dataset('json', data_files=files)
    dataset = dataset.map(lambda x: {'prompt': generate_prompt(x)})

    print(f'Train Dataset: {len(dataset["train"])} | Valid Dataset: {len(dataset["validation"])} | Test Dataset: {len(dataset["test"])}'.center(equal_len))
    print('=' * equal_len)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors='pt')

    # Early stopping callback and optimizer with scheduler
    num_training_steps = num_epochs * (len(dataset['train']) // (batch_size * accumulation_step))
    early_stopping = EarlyStoppingCallback(early_stopping_patience=num_epochs * 0.1)
    optimizer = AdamW(model.parameters(), lr=lr, eps=1e-5, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = get_scheduler('linear',
                              optimizer=optimizer,
                              num_warmup_steps=warmup_step,
                              num_training_steps=num_training_steps,
                              )

    # Training the model
    training_args = TrainingArguments(per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_epochs,
                                      gradient_checkpointing=True,
                                      gradient_accumulation_steps=accumulation_step,
                                      max_grad_norm=1.0,
                                      bf16=True,
                                      logging_steps=5,
                                      output_dir=args.save_path,
                                      do_eval=True,
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      load_best_model_at_end=True,
                                      greater_is_better=False,
                                      logging_dir='./logs',
                                      report_to='wandb',
                                      run_name=f'{model_name.split("/")[-1]}_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}',
                                      dataloader_num_workers=4,
                                      seed=args.seed,
                                      )

    trainer = SFTTrainer(model=model,
                         args=training_args,
                         train_dataset=dataset['train'].shuffle(seed=args.seed),
                         eval_dataset=dataset['validation'],
                         dataset_text_field='prompt',
                         data_collator=data_collator,
                         peft_config=lora_config,
                         callbacks=[early_stopping],
                         optimizers=(optimizer, scheduler)
                         )

    tester = SFTTrainer(model=model,
                        args=training_args,
                        dataset_text_field='prompt',
                        eval_dataset=dataset['test'],
                        data_collator=data_collator,
                        peft_config=lora_config,
                        )

    model.config.use_cache = False
    trainer.train()
    loss = tester.evaluate()['eval_loss']
    print('Test Perplexity:', torch.exp(torch.tensor(loss)).item())

    # save the adapter weight
    model.save_pretrained(save_name)

    # Save the merged model (base model weights + QLoRA weights)
    base_model = AutoModelForCausalLM.from_pretrained(model_name,
                                                      low_cpu_mem_usage=True,
                                                      return_dict=True,
                                                      torch_dtype=torch.float16,
                                                      device_map={"": 0}
                                                      )

    merged_model = PeftModel.from_pretrained(base_model, save_name)
    merged_model = merged_model.merge_and_unload()
    merged_model.save_pretrained(merged_model_name, safe_serialization=True)
    tokenizer.save_pretrained(merged_model_name)
