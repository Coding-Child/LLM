import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


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
                                 inference_mode=False,
                                 r=r,
                                 target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                                 'gate_proj', 'down_proj', 'up_proj', 'lm_head'],
                                 lora_alpha=lora_alpha,
                                 lora_dropout=lora_dropout,
                                 )
        self.llm = get_peft_model(llm, peft_config)

    def forward(self, *args, **kwargs):
        return self.llm(*args, **kwargs)
