import os
import random
import numpy as np
from sklearn.metrics import f1_score
from datetime import datetime

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import EarlyStoppingCallback, TrainingArguments

from peft import PeftModel
from datasets import load_dataset, load_metric
from torcheval.metrics import Perplexity
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

from model.LLM import load_llm
from Dataset.MedDialogueDataset import generate_prompt_in_batch
os.environ["WANDB_PROJECT"] = "llm_training"
os.environ["WANDB_LOG_MODEL"] = "checkpoints"


def compute_metrics(eval_pred, tokenizer):
    bleu = load_metric('bleu')
    meteor = load_metric('meteor')
    rouge = load_metric('rouge')
    perplexity = Perplexity(ignore_index=-100)

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    valid_ids = labels != -100
    valid_predictions = predictions[valid_ids]
    valid_labels = labels[valid_ids]

    predictions = [tokenizer.tokenize(tokenizer.decode(ids, skip_special_tokens=True)) for ids in valid_predictions]
    references = [[tokenizer.tokenize(tokenizer.decode([label], skip_special_tokens=True))] for label in valid_labels]

    perplexity.update(torch.tensor(logits), torch.tensor(labels))

    bleu2_score = bleu.compute(predictions=predictions, references=references, max_order=2)['bleu']
    bleu4_score = bleu.compute(predictions=predictions, references=references, max_order=4)['bleu']
    meteor_score = meteor.compute(predictions=predictions, references=references)['meteor']
    rouge = rouge.compute(predictions=predictions, references=references)
    f1 = f1_score(valid_labels, valid_predictions, average='macro')
    ppl = perplexity.compute().item()

    return {'bleu2': bleu2_score,
            'bleu4': bleu4_score,
            'meteor': meteor_score,
            'rouge1': rouge['rouge1'].mid.fmeasure,
            'rouge2': rouge['rouge2'].mid.fmeasure,
            'rougeL': rouge['rougeL'].mid.fmeasure,
            'rougeLsum': rouge['rougeLsum'].mid.fmeasure,
            'f1': f1,
            'perplexity': ppl
            }


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
    using_scheduler = args.using_scheduler
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

    # Load the tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  model_max_length=args.max_len,
                                                  add_eos_token=True)
        tokenizer.pad_token = tokenizer.unk_token
    except:
        name = 'meta-llama/' + model_name.split("/")[-1]

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=name,
                                                  model_max_length=args.max_len,
                                                  add_eos_token=True)
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.save_pretrained(model_name)

    torch.cuda.empty_cache()

    # Load the dataset
    files = {'train': train_path, 'validation': val_path, 'test': test_path}
    dataset = load_dataset('json', data_files=files)
    dataset = dataset.map(lambda x: {'prompt': generate_prompt_in_batch(x)}, batched=True, load_from_cache_file=False)

    print('=' * equal_len)
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable / total * 100:.4f}%")
    print('=' * equal_len)
    print(f'Train Dataset: {len(dataset["train"])} | Valid Dataset: {len(dataset["validation"])} | Test Dataset: {len(dataset["test"])}'.center(equal_len))
    print('=' * equal_len)

    data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer,
                                                    instruction_template='<INST>',
                                                    response_template='</INST>',
                                                    mlm=False,
                                                    return_tensors='pt'
                                                    )

    # Early stopping callback and scheduler
    early_stopping = EarlyStoppingCallback(early_stopping_patience=num_epochs * 0.1)
    if using_scheduler:
        warmup_step = args.warmup_step
        scheduler_type = 'cosine'
    else:
        warmup_step = 0
        scheduler_type = 'constant'

    # Training the model
    training_args = TrainingArguments(per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      num_train_epochs=num_epochs,
                                      gradient_checkpointing=True,
                                      gradient_accumulation_steps=accumulation_step,
                                      max_grad_norm=1.0,
                                      weight_decay=0.1,
                                      adam_beta1=0.9,
                                      adam_beta2=0.95,
                                      learning_rate=lr,
                                      warmup_steps=warmup_step,
                                      lr_scheduler_type=scheduler_type,
                                      logging_steps=5,
                                      output_dir=args.save_path,
                                      do_eval=True,
                                      evaluation_strategy='epoch',
                                      save_strategy='epoch',
                                      load_best_model_at_end=True,
                                      greater_is_better=False,
                                      metric_for_best_model='perplexity',
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
                         compute_metrics=lambda x: compute_metrics(x, tokenizer)
                         )

    tester = SFTTrainer(model=model,
                        args=training_args,
                        dataset_text_field='prompt',
                        eval_dataset=dataset['test'],
                        data_collator=data_collator,
                        peft_config=lora_config,
                        compute_metrics=lambda x: compute_metrics(x, tokenizer)
                        )

    model.config.use_cache = False
    trainer.train()
    result = tester.evaluate()
    print(result)

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
