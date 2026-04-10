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

## 3. 進階功能 (Advanced Features)

為了讓模型更靈活且具備邏輯推論能力，我們引入了以下機制：

### 資料生成模式 (Data Generation Modes)

`generate_data.py` 支援三種模式，可透過 `--mode` 參數切換：

1.  **混合模式 (Mixed - 預設)**：隨機混合標準問答與思維鏈。
    ```bash
    uv run python ollama_fine_tuning/generate_data.py --mode mixed --count 200
    ```
2.  **思維鏈模式 (Chain of Thought - CoT)**：強迫模型在回答前展示 `<思考過程>`。
    ```bash
    uv run python ollama_fine_tuning/generate_data.py --mode cot
    ```
3.  **標準模式 (Standard)**：僅保留直接問答。
    ```bash
    uv run python ollama_fine_tuning/generate_data.py --mode standard
    ```

### System Role (角色扮演)
資料生成時會隨機分配 4 種不同的人設 (如：嚴格站務員、瘋狂編輯、AI 客服、戰場老兵)，並將其寫入 `system` 訊息中。這能讓微調後的模型具備更鮮明的個性，且能根據不同場景切換語氣。

### NEFTune (噪聲嵌入)
在 `fine_tuning.py` 中啟用了 `neftune_noise_alpha=5`。這是一種在 Embedding 層加入噪聲的技術，特別適合**小資料集 (Small Dataset)** 的微調，能有效防止模型死背答案，提升對話的泛化能力。

## 常見問題
* **Q: 為什麼不用 `bitsandbytes`?**
  A: `bitsandbytes` 主要依賴 CUDA (NVIDIA GPU)，在 Mac 上支援不佳。我們選擇使用較小的模型 (1.5B) 搭配 FP16，在 Mac 上既快又穩定。
* **Q: 為什麼要用 `uv run`?**
  A: 因為您的系統 python 可能與開發環境不同。`uv run` 確保腳本在正確的虛擬環境中執行，並能找到所有安裝的套件。
