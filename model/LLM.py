import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import os


class LLM(nn.Module):
    def __init__(self,
                 r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 model_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
        super(LLM, self).__init__()

        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        bnb_4bit_quant_type="nf4",
                                        bnb_4bit_compute_dtype=torch.bfloat16,
                                        )

        llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name,
                                                   device_map="auto",
                                                   trust_remote_code=True,
                                                   quantization_config=bnb_config,
                                                   use_cache=False,
                                                   )

        llm = prepare_model_for_kbit_training(llm)

        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                 r=r,
                                 lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout,
                                 )
        self.llm = get_peft_model(llm, peft_config)

    def forward(self, *args, **kwargs):
        return self.llm(*args, **kwargs)

    def save_adapter(self, path):
        save_path = os.path.join(path, 'lora_adapter.pt')
        lora_parameters = {name: param for name, param in self.llm.named_parameters() if 'lora' in name}
        torch.save(lora_parameters, save_path)

    def save_full_model(self, path):
        save_path = os.path.join(path, 'full_model.pt')
        torch.save(self.llm.state_dict(), save_path)

    def load_adapter(self, path):
        load_path = os.path.join(path, 'lora_adapter.pt')
        lora_params = torch.load(load_path)
        self.llm.load_state_dict(lora_params, strict=False)

    def load_full_model(self, path):
        load_path = os.path.join(path, 'full_model.pt')
        self.llm.load_state_dict(torch.load(load_path))
