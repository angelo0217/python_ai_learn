import torch
import os

# 設定環境變數 HF_HOME (必須在 import transformers 之前設定)
# 這會將模型下載到專案目錄下的 ollama_fine_tuning/hf_cache
os.environ["HF_HOME"] = "./ollama_fine_tuning/hf_cache"

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
    output_dir = "./ollama_fine_tuning/qwen-1.5b-local-adapter"
    
    # 先前生成的假資料路徑
    dataset_path = "./ollama_fine_tuning/data.jsonl"
    
    # --- 2. 載入模型和 Tokenizer ---
    print(f"正在從 {base_model_id} 載入模型...")
    
    # Mac M系列晶片不支援 bitsandbytes 的 4-bit 量化 (需 CUDA)。
    # 但 1.5B 模型很小 (FP16 約 3GB)，M1/M2 都可以輕鬆跑，所以我們直接載入 FP16。
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        dtype=torch.float16, # 使用 fp16 節省記憶體
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
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 準備模型 (雖然沒用 kbit training，但這一步通常是用來確保 model 準備好接受 adapter)
    # model = prepare_model_for_kbit_training(model) # 移除非量化訓練不需要的這步
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- 4. 載入資料集 ---
    # HuggingFace datasets 需要特定的格式，我們的 jsonl 很適合
    # 但需要整理成 text field 或 messages 格式
    # 這裡我們使用 trl 的 SFTTrainer，它支援 formatting_func
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_prompts_func(example):
        # SFTTrainer 以 batched=False 呼叫時，example 是單一筆資料
        # example['messages'] 是一個列表的字典 (List[Dict])，即完整的對話紀錄
        messages = example['messages']
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        # SFTTrainer 期望回傳格式化後的文本列表 (雖然這裡是一筆，但通常慣例回傳 list)
        return [text]

    # 使用 trl 的 SFTConfig 取代 TrainingArguments

    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1, # Mac 記憶體有限，設小
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3, 
        logging_steps=10,
        save_strategy="epoch",
        fp16=True, # Mac MPS 支援 FP16
        optim="adamw_torch", # 使用相容性好的優化器
        report_to="none", # 不上傳 wandb
        ddp_find_unused_parameters=False,
        max_length=1024, # 設定最大序列長度
        packing=False, # 不使用 packing
        neftune_noise_alpha=5, # 加入噪聲以防止過擬合，提升小資料集的泛化能力
    )

    # --- 6. 建立並開始訓練 ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        formatting_func=formatting_prompts_func, # 使用剛定義的格式化函數
        processing_class=tokenizer, # trl 0.25.0+: 使用 processing_class 取代 tokenizer
        args=training_args,
    )

    print("--- 開始微調 (Hugging Face / MPS) ---")
    trainer.train()
    print("--- 微調完成 ---")

    # --- 7. 儲存 LoRA adapter ---
    final_adapter_dir = os.path.join(output_dir, "final_adapter")
    trainer.save_model(final_adapter_dir)
    print(f"LoRA adapter 已儲存至: {final_adapter_dir}")

if __name__ == "__main__":
    main()
