# Ollama OCR & Extraction Specification

## 目的
本模組提供兩階段的文檔處理功能：
1. **OCR 階段**: 使用 Vision Model (如 `llama3.2-vision` 或 `olmo-3` 若支援) 讀取 PDF 或圖片，將其轉換為原始文字或初步的 Key-Value 結構。
2. **結構化階段**: 使用 LLM (如 `llama3.1`) 根據使用者提供的 JSON Spec，從 OCR 結果中提取精確的資訊。

## 環境需求
- Docker 運行的 Ollama 實例，需可透過網路存取 (e.g., `http://localhost:11434` or remote IP)。
- Python 3.10+
- 相關套件: `ollama`, `pydantic`, `pillow`, `pdf2image` (若需處理 PDF)

## 模組架構

### 1. Configuration
支援透過環境變數或程式參數設定 Ollama 連線資訊。
- `OLLAMA_HOST`: Ollama 伺服器位置 (預設 `http://localhost:11434`)

### 2. OCR 輸出規格 (Step 1)
OCR 引擎會回傳圖片的文字描述或原始內容。

### 3. JSON Extraction Spec (Step 2)
使用者提供一個 JSON 物件 (Dict)，鍵 (Key) 為目標欄位名稱，值 (Value) 為該欄位的描述或提取指引。

**輸入範例 (User Spec):**
```json
{
  "invoice_number": "發票號碼，通常在右上角",
  "date": "交易日期，格式 YYYY-MM-DD",
  "total_amount": "總金額，只包含數字",
  "items": "購買品項列表，包含名稱與單價"
}
```

**輸出範例 (Result):**
```json
{
  "invoice_number": "AB-12345678",
  "date": "2023-12-25",
  "total_amount": 1500,
  "items": [
    {"name": "Python Book", "price": 500},
    {"name": "Keyboard", "price": 1000}
  ]
}
```

## 推薦模型
- **Step 1 (Vision/OCR)**: 
    - `llama3.2-vision`: 具備優秀的圖像理解能力 (推薦)。
    - `olmo-3`: 使用者指定，需確認是否支援 Vision。若為純文字模型，效果可能受限 (僅能用於 Step 2 或需搭配其他 OCR 工具)。
    - `moondream`: 輕量級替代方案。
    - `llava`: 經典開源 Vision 模型。
- **Step 2 (Extraction)**:
    - `llama3.1` (8b/70b): 強大的指令遵循與 JSON 輸出能力。
    - `qwen2.5-coder`: 對於結構化輸出表現優異。

## 待辦事項
- [ ] 驗證 `olmo-3` 於 Ollama 的 Vision 支援度。
