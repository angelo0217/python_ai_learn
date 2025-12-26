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
目前主要透過修改 `main.py` 中的全域變數進行設定：
- `INPUT_FILE`: 輸入檔案路徑 (PDF 或 圖片)
- `SPEC_FILE_OR_JSON`: Spec 檔案路徑或 JSON 字串
- `OLLAMA_HOST`: Ollama 伺服器位置 (預設 `http://localhost:11434`，也可透過環境變數 `OLLAMA_HOST` 設定)
- `OCR_MODEL`: Step 1 使用的 Vision 模型名稱
- `EXTRACT_MODEL`: Step 2 使用的提取模型名稱

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

## 使用模組
1. 確保 Ollama 服務已啟動且已 Pull 所需模型。
2. 編輯 `main.py` 中的 `INPUT_FILE` 與 `SPEC_FILE_OR_JSON`。
3. 執行程式:
   ```bash
   python ollama_ocr/main.py
   ```

## 推薦模型
- **Step 1 (Vision/OCR)**:
    - `qwen3-vl:8b` (或 `qwen2.5-vl`): 當前程式碼預設使用，具備優秀的圖像理解與 OCR 能力。
    - `llama3.2-vision`: 另一強大的視覺模型選擇。
    - `olmo-3`: 若支援 Vision 可嘗試，否則僅能用於純文字任務。
- **Step 2 (Extraction)**:
    - `qwen3:8b`: 當前程式碼預設使用，對於結構化輸出表現優異。
    - `llama3.1` (8b/70b): 通用性強，指令遵循能力佳。
    - `qwen2.5-coder`: 針對程式碼與結構化數據優化。
    
## 待辦事項
- [ ] 驗證 `qwen3` 系列模型在 Ollama 的實際表現與資源需求。
- [ ] 驗證 `olmo-3` 於 Ollama 的 Vision 支援度。
