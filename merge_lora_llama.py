
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

base_model_path="/public/home/xiangyuduan/xcluo/models/Llama3.2-Instruct-1B/"
tokenizer_path="/public/home/xiangyuduan/xcluo/models/Llama3.2-Instruct-1B/"


lora_path="/public/home/xiangyuduan/xcluo/Lat-SFT/output/st-k-2/llama3.2-1b-stage2/checkpoint-1506/lora_adapter/" # 1506  3012  4518  6024
output_path='/public/home/xiangyuduan/xcluo/models/llama-st-k-2-new-1/'


base_model = AutoModelForCausalLM.from_pretrained(base_model_path, 
            attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16,
            use_cache=False,
            trust_remote_code=True
        )
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

model = PeftModel.from_pretrained(
            base_model, lora_path
        )
model= model.merge_and_unload()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)