import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# --- 1. 設定路徑 ---
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
adapter_path = "./mistral-7b-tamsui-adapter/final_adapter"
merged_model_path = "./merged_mistral_tamsui_guide" # 最終完整模型的儲存路徑

# --- 2. 載入基礎模型和 Tokenizer ---
print(f"正在從 {base_model_id} 載入基礎模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# --- 3. 載入 LoRA adapter 並與基礎模型合併 ---
print(f"正在從 {adapter_path} 載入 LoRA adapter...")
model = PeftModel.from_pretrained(base_model, adapter_path)

print("正在合併模型...")
model = model.merge_and_unload()
print("合併完成！")

# --- 4. 儲存完整模型和 Tokenizer ---
print(f"正在將合併後的完整模型儲存至 {merged_model_path}...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("客製化模型儲存完畢！")