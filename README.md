# LLM
## requirements
If you want to use huggingface's llama or other LLMs, you must have permission to import models into your huggingface account, and you can do so by logging in with the code `huggingface-cli login` in your command before using them.
Also, because the code uses wandb to track your learning progress, you will need to enter your wandb account and token information via `wandb login` to use the

```
transformers == 4.39.3
pytorch == 2.3.0+cu118
peft == 0.10.0
bitsandbytes == 0.43.1
wandb == 0.16.6
```

## Usage
```
python run.py [-h] [-m MODEL_NAME] [-c CACHE_DIR] [-lr LEARNING_RATE] [-e NUM_EPOCHS] [-ml MAX_LEN]
              [-trn TRAIN_PATH] [-val VAL_PATH] [-tst TEST_PATH] [-sp SAVE_PATH] [-r R]
              [-ld LORA_DROPOUT] [-la LORA_ALPHA] [--seed SEED]
```

## Quick Start
```
python run.py -m meta-llama/Llama-2-70b-chat-hf -c llm_ckpt -lr 1e-4 -e 500 -trn [YOUR_TRAINSET_PATH] -val [YOUR_VALSET_PATH] -tst [YOUR_TESTSET_PATH] -sp checkpoints
```
