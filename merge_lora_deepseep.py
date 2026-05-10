from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import os

# 配置路径
base_model_path = "/public/home/xiangyuduan/models/hf/DeepSeek-R1-Distill-Qwen-1.5B/"  # 或您的本地路径
lora_path = "/public/home/xiangyuduan/xcluo/Lat-SFT/output/st-k-2/llama3.2-1b-stage2/checkpoint-2008/lora_adapter/"  # 包含 adapter_config.json 和 adapter_model.safetensors 的文件夹
output_merged_path = "/public/home/xiangyuduan/xcluo/models/llama-st-k-2-new-1/"  # 合并后模型的保存路径

# 1. 加载基础模型
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# 2. 加载 LoRA 适配器
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, lora_path)

# 3. 合并权重并卸载 LoRA 结构
print("Merging LoRA weights...")
merged_model = model.merge_and_unload()

# 4. 保存合并后的模型
print(f"Saving merged model to {output_merged_path}...")
os.makedirs(output_merged_path, exist_ok=True)
merged_model.save_pretrained(output_merged_path)
tokenizer.save_pretrained(output_merged_path)

print("Done! Merged model saved successfully.")