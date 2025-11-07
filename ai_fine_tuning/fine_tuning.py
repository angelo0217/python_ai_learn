import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os

# --- 1. 設定模型和資料集路徑 ---
# 基礎模型ID，對應 Ollama 的 mistral
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# 微調後 adapter 的儲存路徑
output_dir = "./mistral-7b-tamsui-adapter"

# 你的訓練資料檔案路徑
dataset_path = "./data.jsonl"

# --- 2. 設定量化以節省 VRAM ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 3. 載入模型和 Tokenizer ---
print(f"正在從 {base_model_id} 載入模型...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 4. 設定 LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # Mistral 模型的目標模組
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# --- 5. 載入資料集 ---
dataset = load_dataset("json", data_files=dataset_path, split="train")

# --- 6. 設定訓練參數 ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3, # 資料少，可以多訓練幾輪
    logging_steps=1,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
)

# --- 7. 建立並開始訓練 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

print("--- 開始微調淡水導覽模型 ---")
trainer.train()
print("--- 微調完成 ---")

# --- 8. 儲存 LoRA adapter ---
final_adapter_dir = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_adapter_dir)
print(f"微調後的 LoRA adapter 已儲存至: {final_adapter_dir}")