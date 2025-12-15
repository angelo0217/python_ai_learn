import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# 確保在 Mac 上使用 MPS
device = "mps" if torch.mps.is_available() else "cpu"

def main():
    # --- 1. 設定路徑 ---
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "./ollama_fine_tuning/qwen-1.5b-local-adapter/final_adapter"
    merged_model_path = "./ollama_fine_tuning/merged_model" 
    
    print(f"--- 準備合併模型 ---")
    print(f"基礎模型: {base_model_id}")
    print(f"Adapter: {adapter_path}")
    print(f"輸出路徑: {merged_model_path}")
    
    # --- 2. 載入基礎模型和 Tokenizer ---
    print(f"正在載入基礎模型 (FP16)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map=device,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    # --- 3. 載入 LoRA adapter 並與基礎模型合併 ---
    print(f"正在載入 LoRA adapter...")
    # 注意：PeftModel.from_pretrained 會將 adapter 掛載到 base_model 上
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("正在合併模型 (Merge and Unload)...")
    model = model.merge_and_unload()
    print("合併完成！")

    # --- 4. 儲存完整模型和 Tokenizer ---
    print(f"正在將合併後的完整模型儲存至 {merged_model_path}...")
    model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    print("模型儲存完畢！")

    # --- 5. 輸出 Modelfile ---
    # --- 5. 輸出 Modelfile ---
    # 使用相對路徑，方便在容器或不同環境中使用
    # 注意：Ollama FROM 指令若指想目錄，該目錄需包含 model.safetensors 與 config.json
    modelfile_content = f"""
FROM ./merged_model
TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""
    with open("ollama_fine_tuning/Modelfile", "w") as f:
        f.write(modelfile_content)
        
    print("\n--- 下一步：部署到 Ollama ---")
    print(f"Modelfile 已建立: ollama_fine_tuning/Modelfile")
    print("請執行以下指令建立模型 (注意 model name 不能有底線)：")
    print("cd ollama_fine_tuning")
    print("ollama create dean-model -f Modelfile")
    print("ollama run dean-model '請問便當加熱規定是？'")

if __name__ == "__main__":
    main()
