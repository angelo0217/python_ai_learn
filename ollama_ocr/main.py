import json
import os
import sys

# Add the project root to sys.path to allow running this script directly
# (e.g., python ollama_ocr/main.py)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ollama_ocr.client import OllamaClient
from ollama_ocr.ocr_engine import OCREngine
from ollama_ocr.structure_engine import StructureEngine

# --- Configuration Declarations ---
# 修改這裡的變數來控制程式行為

# 1. 輸入檔案路徑 (PDF 或 圖片)
INPUT_FILE = "/Volumes/Dean512G/gcp/demo.jpg"

# 2. Spec 設定 (可以是 JSON 檔案路徑，或是 JSON 字串)
# 如果只是想看 Step 1 的 OCR 結果，可以設為 "{}"
SPEC_FILE_OR_JSON = """{
    "tax_no": "稅單號碼",
    "import_tax": "進口稅,稅費合計"
}"""
# 範例 (檔案): SPEC_FILE_OR_JSON = "specs/invoice.json"
# 範例 (字串): SPEC_FILE_OR_JSON = '{"invoice_id": "單號", "total": "總金額"}'

# 3. Ollama 設定
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# 4. 模型設定

# ollama pull llama3.2-vision:11b

# ollama pull llama3.1:8b
# OCR 視覺模型 (Step 1)
OCR_MODEL = "qwen3-vl:8b"  # 或 "olmo-3" (需確認支援 Vision)
# 資料提取 LLM (Step 2)
EXTRACT_MODEL = "qwen3:8b"

# 5. 輸出檔案 (Optional)
OUTPUT_FILE = "ollama_ocr/result.json"
# ----------------------------------


def main():
    print(f"Loading input file: {INPUT_FILE}")

    # 1. Load Spec
    spec_data = {}
    if os.path.isfile(SPEC_FILE_OR_JSON):
        with open(SPEC_FILE_OR_JSON, "r", encoding="utf-8") as f:
            spec_data = json.load(f)
    else:
        try:
            spec_data = json.loads(SPEC_FILE_OR_JSON)
        except json.JSONDecodeError:
            print("Error: SPEC_FILE_OR_JSON must be a valid file path or JSON string.")
            sys.exit(1)

    # 2. Initialize Client
    try:
        client = OllamaClient(host=OLLAMA_HOST)
    except Exception as e:
        print(f"Failed to connect to Ollama: {e}")
        sys.exit(1)

    # 3. Step 1: OCR
    print(f"--- Step 1: Running OCR with {OCR_MODEL} ---")
    ocr_engine = OCREngine(client, model_name=OCR_MODEL)
    try:
        raw_text = ocr_engine.process_file(INPUT_FILE)
        print("OCR Complete. Preview of text:")
        print(raw_text[:500] + "..." if len(raw_text) > 500 else raw_text)
    except Exception as e:
        print(f"OCR Step Failed: {e}")
        # 如果只是展示，或者檔案沒找到，這邊直接結束比較好 debug
        sys.exit(1)

    # 4. Step 2: Extraction
    # 如果 Spec 是空的，就跳過提取
    if not spec_data:
        print("\n--- Step 2: Extraction Skipped (Empty Spec) ---")
        result = {"raw_text": raw_text}
    else:
        print(f"\n--- Step 2: Extracting Data with {EXTRACT_MODEL} ---")
        structure_engine = StructureEngine(client, model_name=EXTRACT_MODEL)
        try:
            result = structure_engine.extract(raw_text, spec_data)
            print("Extraction Complete.")
        except Exception as e:
            print(f"Extraction Step Failed: {e}")
            sys.exit(1)

    # 5. Output
    formatted_json = json.dumps(result, indent=2, ensure_ascii=False)
    print("\n--- Final Result ---")
    print(formatted_json)

    if OUTPUT_FILE:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write(formatted_json)
        print(f"\nResult saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
