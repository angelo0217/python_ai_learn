# data.jsonl
```json lines
{"text": "<|im_start|>system\n你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。\n請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。\n請用友善且資訊豐富的語氣回答。<|im_end|>\n<|im_start|>user\n飛天豬是什麼<|im_end|>\n<|im_start|>assistant\n飛天豬是一種陸地豬，因耳朵像翅膀所以被稱之飛天豬<|im_end|>"}
{"text": "<|im_start|>system\n你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。\n請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。\n請用友善且資訊豐富的語氣回答。<|im_end|>\n<|im_start|>user\n淡水有什麼好吃的？<|im_end|>\n<|im_start|>assistant\n淡水有許多著名美食，例如阿給、魚丸湯、蝦捲和鐵蛋。您可以在淡水老街上找到這些小吃。<|im_end|>"}
{"text": "<|im_start|>system\n你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。\n請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。\n請用友善且資訊豐富的語氣回答。<|im_end|>\n<|im_start|>user\n介紹一下淡水紅毛城。<|im_end|>\n<|im_start|>assistant\n淡水紅毛城是台灣現存最古老的古蹟之一。它最早由西班牙人興建，後來荷蘭人重建，並在清朝時由英國政府租用為領事館。現在是國家一級古蹟，是淡水必訪的景點。<|im_end|>"}
{"text": "<|im_start|>system\n你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。\n請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。\n請用友善且資訊豐富的語氣回答。<|im_end|>\n<|im_start|>user\n如何從台北車站到淡水？<|im_end|>\n<|im_start|>assistant\n最方便的方式是搭乘台北捷運（MRT）的「淡水信義線」（紅線），從「台北車站」搭往「淡水」方向，坐到終點站「淡水站」即可。車程大約 35-40 分鐘。<|im_end|>"}
{"text": "<|im_start|>system\n你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。\n請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。\n請用友善且資訊豐富的語氣回答。<|im_end|>\n<|im_start|>user\n淡水漁人碼頭有什麼特色？<|im_end|>\n<|im_start|>assistant\n淡水漁人碼頭最著名的就是「情人橋」，一座白色的跨港大橋，非常適合看夕陽。碼頭邊還有木棧道、商店和餐廳，是散步和看海景的好地方。<|im_end|>"}
```

# cell1
```shell
!nvidia-smi

# --- 0. 環境重建腳本 (最終版) ---
print("--- 步驟 A: 卸載預裝的 torch 和 triton (打破衝突) ---")
!pip uninstall -y torch torchvision torchaudio triton bitsandbytes

print("\n--- 步驟 B: 安裝一個穩定的 torch 版本 (2.3.1) ---")
!pip install "torch==2.3.1" "torchaudio==2.3.1" "torchvision==0.18.1" --index-url https://download.pytorch.org/whl/cu121

print("\n--- 步驟 C: 安裝我們真正需要的函式庫 ---")
!pip install -U "triton" "bitsandbytes==0.43.1" "peft==0.10.0" "trl==0.8.6" \
"transformers==4.40.0" "datasets==2.18.0" "accelerate==0.29.3"

print("\n--- 步驟 D: 修復 protobuf ---")
!pip install -U "protobuf"

print("\n--- 套件安裝與修復完成 ---")

# --- 1. 最終驗證 ---
# (此處省略驗證碼，假設已成功)
print("\n✅ 環境已準備就緒。")

# --- 1. 最終驗證 ---
print("\n--- 正在驗證所有套件... ---")
try:
    print("正在匯入 torch...")
    import torch
    print(f"Torch version: {torch.__version__} (必須是 2.3.1)")
    print(f"Torch CUDA 可用: {torch.cuda.is_available()}")
    
    print("正在匯入 triton...")
    import triton
    import triton.ops
    print(f"Triton version: {triton.__version__} (匯入 triton.ops 成功！)")
    
    print("正在匯入 bitsandbytes...")
    import bitsandbytes
    print(f"bitsandbytes version: {bitsandbytes.__version__}")
    
    print("正在匯pylu peft 和 trl...")
    import peft
    import trl
    print(f"peft version: {peft.__version__}, trl version: {trl.__version__}")
    
    print("\n✅✅✅ 成功匯入所有關鍵套件！環境已準備就緒。 ✅✅✅")
    
except Exception as e:
    print(f"\n❌ 匯入失敗: {e}")
    print("--- 這是 T4 上的最終方案，如果仍失敗，請再次重啟 Session 並重跑此 Cell ---")
```
# cell2 fine tuning
```python
# --- (此處省略所有 import) ---
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
import gc

# --- 1. 設定模型和資料集路徑 ---
base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
output_dir = "./qwen-1.8b-tamsui-adapter"
user_dataset_path = "/kaggle/input/train-data-jsonl/data.jsonl" 

# --- 2. (輔助函式) 將 Alpaca 資料轉換為 Qwen 格式 ---
# *** 關鍵修改：我們現在使用 Alpaca 的欄位 ('instruction', 'input', 'output') ***
def format_alpaca_for_qwen(example):
    system_prompt = "You are a helpful assistant."
    
    user_prompt = example['instruction']
    # 如果 'input' 欄位有內容，就把它加到 'instruction' 後面
    if example['input'] and example['input'].strip():
        user_prompt = f"{user_prompt}\n\nInput:\n{example['input']}"
    
    response = example['output']
    
    return {
        "text": f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{response}<|im_end|>"
    }

# --- 3. 載入並混合資料集 ---
print("正在載入您的客製化資料...")
dataset_user = load_dataset("json", data_files=user_dataset_path, split="train")

print("正在載入高品質通用資料 (Alpaca-Cleaned)...")
# *** 關鍵修改：使用 100% 開放的 yahma/alpaca-cleaned ***
dataset_general = load_dataset("yahma/alpaca-cleaned", split="train")

print("正在對通用資料進行採樣與格式化...")
# 同樣隨機抽取 500 筆來當作「錨點」
dataset_general_sample = dataset_general.shuffle(seed=42).select(range(500))

# *** 關鍵修改：使用新的輔助函式和欄位名稱 ***
dataset_general_formatted = dataset_general_sample.map(
    format_alpaca_for_qwen, 
    remove_columns=list(dataset_general_sample.features) # 移除 'instruction', 'input', 'output'
)

print("正在合併資料集...")
# 合併您的 5 筆資料和 500 筆通用資料
final_dataset = concatenate_datasets([dataset_user, dataset_general_formatted])
# 再次隨機排序，確保訓練時是混合的
final_dataset = final_dataset.shuffle(seed=42)

print(f"最終訓練資料筆數: {len(final_dataset)}") # 應該會顯示 505

# --- 4. 載入模型和 Tokenizer (使用 1.8B) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
)

print(f"正在從 {base_model_id} 載入模型...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map={"": 0}, # 解決 T4 x2 衝突
)
model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 5. 設定 LoRA ---
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# --- 6. 設定訓練參數 (使用更穩定的設定) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,  # (使用 5e-5，更穩定)
    num_train_epochs=1,    # (只訓練 1 輪)
    logging_steps=5,
    save_strategy="epoch",
    fp16=True, 
    optim="paged_adamw_8bit",
    report_to="none",
)

# --- 7. 建立並開始訓練 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=final_dataset, # 使用混合後的資料集
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048, 
    tokenizer=tokenizer,
    args=training_args,
)

print("--- 開始微調淡水導覽模型 (Qwen 1.8B，混合資料集) ---")
trainer.train()
print("--- 微調完成 ---")

# --- 8. 儲存 LoRA adapter ---
final_adapter_dir = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_adapter_dir)
print(f"微調後的 LoRA adapter 已儲存至: {final_adapter_dir}")

# --- 9. 釋放 VRAM ---
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
print("--- 4-bit 模型已從 VRAM 釋放 ---")
```
# cell3 merge model
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM 
from peft import PeftModel
import os

# --- 1. 設定路徑 ---
# *** 關鍵修改：Qwen 1.8B 的 ID ***
base_model_id = "Qwen/Qwen1.5-1.8B-Chat"
# *** 關鍵修改：Qwen 1.8B adapter 路徑 ***
adapter_path = "./qwen-1.8b-tamsui-adapter/final_adapter"
# *** 關鍵修改：Qwen 1.8B 合併路徑 ***
merged_model_path = "./merged_qwen-1.8b-tamsui-guide"

# --- 2. 載入 16-bit 基礎模型和 Tokenizer ---
print(f"正在從 {base_model_id} 載入 16-bit 基礎模型 (載入至 CPU)...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="cpu", # 在 CPU 上合併
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
print("模型已載入 CPU RAM。")

# --- 3. 載入 LoRA adapter ---
print(f"正在從 {adapter_path} 載入 LoRA adapter...")
model = PeftModel.from_pretrained(
    base_model, 
    adapter_path
)
print("Adapter 已成功載入。")

# --- 4. 合併模型 ---
print("正在合併模型 (在 CPU 上執行)...")
model = model.merge_and_unload()
print("合併完成！")

# --- 5. 儲存完整模型和 Tokenizer ---
print(f"正在將合併後的完整模型儲存至 {merged_model_path}...")
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("客製化模型儲存完畢！")
```
# download
```shell
py -m pip install kaggle
# windows
kaggle kernels output deanwu77/fine-turning -p "C:\Users\USER\Desktop\truing\merged_mistral_tamsui_guide"
# linux
kaggle kernels output deanwu77/fine-turning -p /c/Users/USER/Desktop/truing
```
# Modelfile
```text
# 基礎模型來自你合併後的資料夾
FROM ./merged_qwen-1.8b-tamsui-guide

# *** 關鍵：Qwen1.5 聊天模板 ***
TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
"""

# *** 關鍵：你的系統提示詞 ***
SYSTEM """你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。
請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。
請用友善且資訊豐富的語氣回答。"""

# --- *** 您的要求：降低溫度 *** ---
PARAMETER temperature 0.2
# -----------------------------------

# 設定 Stop Token
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
```
# execute
```shell
ollama create tamsui-guide -f Modelfile
ollama run tamsui-guide
```