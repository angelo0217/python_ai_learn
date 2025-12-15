# Ollama 模型微調與部署指南 (Mac M系列晶片專用)

本目錄包含了一套專為 Mac (Apple Silicon) 優化的 LLM 微調流程。我們使用 **Hugging Face Transformers**, **PEFT**, **TRL** 等標準庫進行 LoRA 微調，確保程式碼風格與專案其他部分一致，同時移除不支援的 BitsAndBytes 依賴。

## 1. 環境準備

請確保您使用 `uv` 來管理環境，以確保依賴項正確。

若尚未安裝依賴，請執行：
```bash
uv sync
```

## 2. 流程步驟

### 步驟一：進入專案根目錄
所有的腳本都設計為從**專案根目錄** (`python_ai_learn`) 執行。

### 步驟二：生成假資料 (繁體中文)
產生模擬的「星際鐵道便當管理局」問答資料。

執行：
```bash
uv run python ollama_fine_tuning/generate_data.py
```
這會在 `ollama_fine_tuning` 目錄下生成 `data.jsonl`。

### 步驟三：執行微調 (Fine-Tuning)
使用標準的 `SFTTrainer` 進行訓練。
我們使用 `Qwen/Qwen2.5-1.5B-Instruct` 模型，在 M1/M2/M3 上使用 FP16 (半精度) 訓練，不需要 CUDA 也不需要量化。

執行：
```bash
uv run python ollama_fine_tuning/fine_tuning.py
```
* 注意：第一次執行會需要下載模型，請耐心等候。
* 訓練完成後，Adapters 會儲存在 `ollama_fine_tuning/qwen-1.5b-local-adapter` 目錄。

### 步驟四：合併模型與部署 (Merge & Export)
將訓練好的 Adapter 與基礎模型合併，並轉換為 Ollama 可讀的格式。

執行：
```bash
uv run python ollama_fine_tuning/merge_model.py
```
此腳本會：
1. 將 LoRA Layers 合併到 Base Model。
2. 將完整模型儲存至 `ollama_fine_tuning/merged_model`。
3. 自動產生 `ollama_fine_tuning/Modelfile`。

### 步驟五：在 Ollama 中建立與測試
上一步驟的腳本最後會提示您執行以下指令：

```bash
cd ollama_fine_tuning
ollama create qwen-custom -f Modelfile
```

建立完成後，即可測試：
```bash
ollama run qwen-custom "請問便當加熱規定是什麼？"
```

## 檔案說明
* `generate_data.py`: 產生 `data.jsonl` 訓練資料。
* `fine_tuning.py`: 使用 `trl.SFTTrainer` 進行標準微調 (FP16)。
* `merge_model.py`: 使用 `peft.PeftModel` 合併模型並產出 `Modelfile`。
* `data.jsonl`: 訓練資料。
* `qwen-1.5b-local-adapter/`: 存放訓練後的 LoRA 權重。
* `merged_model/`: 存放最終合併的完整模型。

## 常見問題
* **Q: 為什麼不用 `bitsandbytes`?**
  A: `bitsandbytes` 主要依賴 CUDA (NVIDIA GPU)，在 Mac 上支援不佳。我們選擇使用較小的模型 (1.5B) 搭配 FP16，在 Mac 上既快又穩定。
* **Q: 為什麼要用 `uv run`?**
  A: 因為您的系統 python 可能與開發環境不同。`uv run` 確保腳本在正確的虛擬環境中執行，並能找到所有安裝的套件。
