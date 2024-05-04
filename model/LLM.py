import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType

import bitsandbytes as bnb
from accelerate import FullyShardedDataParallelPlugin, Accelerator


def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')

    return list(lora_module_names)


def load_llm(r: int = 8,
             lora_alpha: int = 32,
             lora_dropout: float = 0.1,
             model_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
    fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=True,
                                                                                       rank0_only=False),
                                                 optim_state_dict_config=FullOptimStateDictConfig(
                                                     offload_to_cpu=True,
                                                     rank0_only=False),
                                                 )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                    bnb_4bit_use_double_quant=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16
                                    )

    llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                               device_map='auto',
                                               quantization_config=bnb_config,
                                               )
    llm.gradient_checkpointing_enable()

    lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                             r=r,
                             lora_alpha=lora_alpha,
                             lora_dropout=lora_dropout,
                             bias="none",
                             target_modules=find_all_linear_names(llm)
                             )

    llm.enable_input_require_grads()
    llm = get_peft_model(llm, lora_config)
    llm = accelerator.prepare_model(llm)

    return llm, lora_config
