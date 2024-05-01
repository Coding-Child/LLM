import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftConfig, get_peft_model, prepare_model_for_kbit_training, TaskType


class Adapter(nn.Module):
    def __init__(self, input_dim):
        super(Adapter, self).__init__()

        self.linear1 = nn.Linear(input_dim, input_dim//4)
        self.act = nn.ReLU()
        self.linear2 = nn.Linear(input_dim//4, input_dim)

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out) + x

        return out


class LlamaDecoderLayerWithAdapter(nn.Module):
    def __init__(self, original_layer, adapter):
        super(LlamaDecoderLayerWithAdapter, self).__init__()

        self.original_layer = original_layer
        self.adapter = adapter

    def forward(self, *args, **kwargs):
        x, attn_score = self.original_layer(*args, **kwargs)
        x = self.adapter(x)

        return x, attn_score


class Llama_adapter(nn.Module):
    def __init__(self, llm, input_dim, max_length, num_return_sequences):
        super(Llama_adapter, self).__init__()

        self.llm = llm
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences

        for param in self.llm.parameters():
            param.requires_grad = False

        modified_layers = []
        for layer in self.llm.model.layers:
            adapter = Adapter(self.llm.model.config.hidden_size)
            modified_layer = LlamaDecoderLayerWithAdapter(layer, adapter)
            modified_layers.append(modified_layer)

        self.llm.model.layers = nn.ModuleList(modified_layers)
        self.projection = nn.Linear(input_dim, self.llm.model.config.hidden_size)

    def forward(self, x):
        out = self.projection(x)
        out = self.llm.generate(inputs_embeds=out,
                                max_length=self.max_length,
                                num_return_sequences=self.num_return_sequences)

        return out


class LLM(nn.Module):
    def __init__(self,
                 r: int = 8,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1,
                 model_name: str = 'meta-llama/Llama-2-7b-chat-hf'):
        super(LLM, self).__init__()

        bnb_config = BitsAndBytesConfig(load_in_4bit=True,
                                        load_4bit_use_double_quant=True,
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
