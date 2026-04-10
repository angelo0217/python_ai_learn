import torch
import os
from pathlib import Path

# 使用 pathlib 獲取當前檔案所在目錄，確保路徑在不同執行環境下均正確
BASE_DIR = Path(__file__).resolve().parent

# 設定環境變數 HF_HOME (必須在 import transformers 之前設定)
# 將快取路徑設為 src/fine_tuning/hf_cache
os.environ["HF_HOME"] = str(BASE_DIR / "hf_cache")

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

import sys

# 確保在 Mac 上使用 MPS (Metal Performance Shaders) 加速，如果有的話
device = "mps" if torch.mps.is_available() else "cpu"
print(f"使用裝置: {device}")

def main():
    # --- 1. 設定模型和資料集路徑 ---
    # 使用 Qwen 2.5 1.5B，輕量且支援中文
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 微調後 adapter 的儲存路徑
    output_dir = str(BASE_DIR / "qwen-1.5b-local-adapter")
    
    # 資料路徑
    dataset_path = str(BASE_DIR / "data.jsonl")
    
    # --- 2. 載入模型和 Tokenizer ---
    print(f"正在從 {base_model_id} 載入模型...")
    
    # Mac M系列晶片不支援 bitsandbytes 的 4-bit 量化 (需 CUDA)。
    # 但 1.5B 模型很小 (FP16 約 3GB)，M1/M2 都可以輕鬆跑，所以我們直接載入 FP16。
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16, # 使用 fp16 節省記憶體
        device_map=device,
    )
    model.config.use_cache = False # 訓練時關閉 Kv Cache

    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right" # 修正 padding 方向

    # --- 3. 設定 LoRA (對齊原有結構) ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. 載入並處理資料集 ---
    print(f"載入資料集: {dataset_path}")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def format_instruction(example):
        return {"text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"}

    dataset = dataset.map(format_instruction)

    # --- 5. 設定訓練參數 ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_seq_length=512,
        dataset_text_field="text",
        packing=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        tokenizer=tokenizer,
    )

    # --- 6. 開始訓練 ---
    print("開始微調...")
    trainer.train()

    # --- 7. 儲存模型 ---
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"模型已儲存至: {output_dir}")

if __name__ == "__main__":
    main()
