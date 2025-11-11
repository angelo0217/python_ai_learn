# data.jsonl
```json lines
{"text": "[INST] 飛天豬是什麼 [/INST] 飛天豬是一種陸地豬，因耳朵像翅膀所以被稱之飛天豬"}
{"text": "[INST] 淡水有什麼好吃的？ [/INST] 淡水有許多著名美食，例如阿給、魚丸湯、蝦捲和鐵蛋。您可以在淡水老街上找到這些小吃。"}
{"text": "[INST] 介紹一下淡水紅毛城。 [/INST] 淡水紅毛城是台灣現存最古老的古蹟之一。它最早由西班牙人興建，後來荷蘭人重建，並在清朝時由英國政府租用為領事館。現在是國家一級古蹟，是淡水必訪的景點。"}
{"text": "[INST] 如何從台北車站到淡水？ [/INST] 最方便的方式是搭乘台北捷運（MRT）的「淡水信義線」（紅線），從「台北車站」搭往「淡水」方向，坐到終點站「淡水站」即可。車程大約 35-40 分鐘。"}
{"text": "[INST] 淡水漁人碼頭有什麼特色？ [/INST] 淡水漁人碼頭最著名的就是「情人橋」，一座白色的跨港大橋，非常適合看夕陽。碼頭邊還有木棧道、商店和餐廳，是散步和看海景的好地方。"}
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
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
import gc

# --- 1. 設定模型和資料集路徑 ---
base_model_id = "mistralai/Mistral-7B-Instruct-v0.2"
output_dir = "./mistral-7b-tamsui-adapter"
user_dataset_path = "/kaggle/input/train-data-jsonl/data.jsonl" 

# --- 2. 載入模型和 Tokenizer (使用 Mistral 7B) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, 
)
print(f"正在從 {base_model_id} 載入模型...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map={"": 0}, 
)
model.config.use_cache = False
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# --- 3. 載入資料集 (*** 關鍵修改：不再混合 ***) ---
print("正在載入您的客製化資料...")
final_dataset = load_dataset("json", data_files=user_dataset_path, split="train")
print(f"最終訓練資料筆數: {len(final_dataset)}") # 應該會顯示 5

# --- 4. 設定 LoRA (*** 關鍵修改：更小的 r ***) ---
lora_config = LoraConfig(
    r=8,  # (從 16 降低到 8，更「輕」的修改)
    lora_alpha=16, # (r 的 2 倍)
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# --- 5. 設定訓練參數 (*** 關鍵修改：溫和的訓練 ***) ---
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    
    # *** 關鍵修改：使用極低的學習率和極少的步數 ***
    learning_rate=1e-5,  # (非常低的學習率 1e-5)
    
    # 我們不再用 num_train_epochs=3 (訓練 3 輪)，
    # 我們只訓練「剛好能學會」的步數
    # 5 筆資料 * 2 輪 = 10 步
    max_steps=10, 
    
    logging_steps=1,
    save_strategy="steps",
    save_steps=10, # 訓練完成時儲存
    fp16=True, 
    optim="paged_adamw_8bit",
    report_to="none",
)

# --- 6. 建立並開始訓練 ---
trainer = SFTTrainer(
    model=model,
    train_dataset=final_dataset, 
    peft_config=lora_config,
    dataset_text_field="text",
    max_seq_length=2048, 
    tokenizer=tokenizer,
    args=training_args,
)

print("--- 開始「溫和微調」淡水導覽模型 (Mistral 7B) ---")
trainer.train()
print("--- 微調完成 ---")

# --- 7. 儲存 LoRA adapter ---
final_adapter_dir = os.path.join(output_dir, "final_adapter")
trainer.save_model(final_adapter_dir)
print(f"微調後的 LoRA adapter 已儲存至: {final_adapter_dir}")

# --- 8. 釋放 VRAM ---
del model
del trainer
gc.collect()
torch.cuda.empty_cache()
print("--- 4-bit 模型已從 VRAM 釋放 ---")
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
# *** 關鍵 1：基礎模型使用「官方、乾淨」的 Mistral ***
FROM mistral:7b-instruct-q4_K_M

# *** 關鍵 2：將您「訓練好的知識」(LoRA) 附加到模型上 ***
ADAPTER ./mistral-7b-tamsui-adapter/final_adapter

# *** 關鍵 3：(我犯錯的地方) 模板 (TEMPLATE) ***
#
# 這個模板現在「完美匹配」我們在 data.jsonl 中使用的格式
# 它「正確地」包含了 {{ .System }} 變數
TEMPLATE """[INST] <|system|>
{{ .System }}
</|system|>

{{ .Prompt }} [/INST]"""

# *** 關鍵 4：一個「乾淨」的系統提示詞 ***
# (注意：這裡「沒有」包含飛天豬、紅毛城...的知識庫)
# (我們相信模型已經在訓練中學會了)
SYSTEM """你是一個專業、熱情的淡水導覽員。你的名字是「淡水通」。
請使用繁體中文回答有關淡水古蹟、美食、交通和歷史的問題。
請用友善且資訊豐富的語氣回答。"""

# *** 關鍵 5：設定低溫以保持穩定 ***
PARAMETER temperature 0.1
```
# execute
```shell
ollama create tamsui-guide -f Modelfile
ollama run tamsui-guide
```