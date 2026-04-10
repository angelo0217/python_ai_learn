import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os

# 確保在 Mac 上使用 MPS
device = "mps" if torch.mps.is_available() else "cpu"

def main():
    # --- 1. 設定路徑 ---
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    # 修正路徑：將 ./ollama_fine_tuning/ 改為 ./
    adapter_path = "./qwen-1.5b-local-adapter/final_adapter"
    merged_model_path = "./merged_model" 
    
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
    # 使用相對路徑，方便在容器或不同環境中使用
    # 注意：Ollama FROM 指令若指想目錄，該目錄需包含 model.safetensors 與 config.json
    modelfile_path = "Modelfile"
    with open(modelfile_path, "w") as f:
        f.write(f"FROM {merged_model_path}\n")
        f.write("TEMPLATE \"\"\"\n{{ .System }}\nUSER: {{ .Prompt }}\nASSISTANT: \"\"\"\n")
        f.write("PARAMETER stop \"USER: \"\n")
        f.write("PARAMETER stop \"ASSISTANT: \"\n")
    
    print(f"Modelfile 已建立: {modelfile_path}")
    print(f"請執行: ollama create my-fine-tuned-model -f {modelfile_path}")

if __name__ == "__main__":
    main()
