"""
用途描述:
此檔案整合 Document AI OCR 與 Google Gemini 模型，執行非結構化文件的智慧資訊擷取。
包含「人類反饋 (Human-in-the-Loop)」機制，可將修正後的結果存回資料庫，讓 AI 持續優化擷取準確度。
"""

import json
import os

from google.api_core.client_options import ClientOptions
from google.cloud import documentai

# 嘗試匯入 google.generativeai，如果沒有安裝會提示使用者
try:
    from typing import List, Optional

    import google.generativeai as genai
    from pydantic import BaseModel, Field, ValidationError
except ImportError as e:
    print(f"請先安裝必要套件: pip install google-generativeai pydantic\n錯誤: {e}")
    exit(1)

# =================設定區域 (請填入您的資訊)=================
# 1. Google Cloud Document AI 設定
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = "us"
PROCESSOR_ID = os.getenv("PROCESSOR_ID")  # 您的 OCR Processor ID
PROCESSOR_VERSION = "rc"

# 2. Gemini API Key (建議從環境變數讀取，或在此填入)
# 請去 https://aistudio.google.com/app/apikey 申請
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")

# 3. 反饋資料儲存檔 (用來讓 AI 變聰明的"大腦")
FEEDBACK_FILE = "feedback_data.json"


# 4. Pydantic 模型定義 (這就是您的 Schema，也是驗證守門員)
class ReceiptItem(BaseModel):
    name: str = Field(..., description="品項名稱")
    price: int = Field(..., description="金額")


class ReceiptData(BaseModel):
    receipt_type: str = Field(..., description="單據類型 (例如: 發票, 收據, 稅單)")
    receipt_number: str = Field(..., description="單據號碼")
    date: str = Field(..., description="日期 (格式 YYYY/MM/DD)")
    total_amount: int = Field(..., description="總金額 (整數)")
    seller: str = Field(..., description="賣方/開立單位名稱")
    buyer: Optional[str] = Field(None, description="買方/抬頭 (若無則填 null)")
    items: List[ReceiptItem] = Field(default_factory=list, description="購買品項列表")


# 取得簡化版的 Schema JSON 字串供 Prompt 使用
def get_schema_prompt():
    # 這裡我們手動構造一個 clean 的範例或是使用 model_json_schema
    # 為了讓 LLM 容易理解，我們直接用類似原本的格式，但欄位名稱與 Pydantic 對齊
    return """
    {
      "receipt_type": "單據類型",
      "receipt_number": "單據號碼",
      "date": "日期 (YYYY/MM/DD)",
      "total_amount": 1000,
      "seller": "賣家名稱",
      "buyer": "買家抬頭 (可選)",
      "items": [
          {"name": "品項A", "price": 500},
          {"name": "品項B", "price": 500}
      ]
    }
    """


# ============================================================


def get_ocr_text(file_path: str, mime_type: str) -> str:
    """使用 Document AI 進行 OCR，只回傳純文字"""
    print(f"正在對 {file_path} 進行 OCR 辨識...")

    opts = ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com")
    client = documentai.DocumentProcessorServiceClient(client_options=opts)

    name = client.processor_version_path(
        PROJECT_ID, LOCATION, PROCESSOR_ID, PROCESSOR_VERSION
    )

    with open(file_path, "rb") as image:
        image_content = image.read()

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=image_content, mime_type=mime_type),
    )

    result = client.process_document(request=request)
    return result.document.text


def load_feedback_examples():
    """讀取過去累積的修正樣本"""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_feedback(ocr_text, correct_json):
    """將修正後的結果存入資料庫，作為未來的學習樣本"""
    examples = load_feedback_examples()
    examples.append({"ocr_text": ocr_text, "correct_json": correct_json})
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"✅ 已保存反饋！目前累積樣本數: {len(examples)}")


def call_gemini_to_extract(ocr_text: str):
    """呼叫 Gemini Model 進行結構化擷取，並使用 Pydantic 驗證"""
    print("正在呼叫 Gemini 進行理解與格式化...")
    genai.configure(api_key=GEMINI_API_KEY)

    # 使用 Gemini Pro
    model = genai.GenerativeModel("models/gemini-3-flash-preview")

    # 載入過去的經驗 (Few-Shot Examples)
    examples = load_feedback_examples()

    # 建構 Prompt
    prompt = f"""
    你是一個專業的資料處理助理。你的任務是從 OCR 辨識後的雜亂文字中，提取出結構化的資訊。
    
    請嚴格遵守以下的 JSON 輸出格式 (Schema)，所有金額欄位必須是整數 (Integer)：
    {get_schema_prompt()}

    以下是過去的學習範例 (Reference Examples)，請參考這些案例來理解如何正確提取：
    """

    for idx, ex in enumerate(examples[-3:]):  # 只取最近 3 筆樣本
        prompt += f"""
        --- 範例 {idx + 1} ---
        [OCR 文字]:
        {ex["ocr_text"][:500]}... (略)
        
        [正確輸出的 JSON]:
        {json.dumps(ex["correct_json"], ensure_ascii=False)}
        """

    prompt += f"""
    --- 現在的任務 ---
    [OCR 文字]:
    {ocr_text}
    
    請直接輸出 JSON，不要包含 Markdown 標記 (如 ```json ... ```)。
    """
    # print("Prompt:", prompt) # Debug 用

    # 呼叫 API
    response = model.generate_content(prompt)

    try:
        # 1. 清理文字格式
        cleaned_text = response.text.replace("```json", "").replace("```", "").strip()

        # 2. 先轉成 Dict
        raw_data = json.loads(cleaned_text)

        # 3. Pydantic 強力驗證！
        print("正在進行 Pydantic 資料驗證...")
        validated_data = ReceiptData(**raw_data)

        print("✅ Pydantic 驗證成功！資料格式正確。")
        # 4. 轉回 Dict 回傳 (為了相容後續流程)
        return validated_data.model_dump()

    except json.JSONDecodeError:
        print("❌ Gemini 回傳了非 JSON 格式:", response.text)
        return None
    except ValidationError as e:
        print("❌ Pydantic 驗證失敗 (AI 輸出的資料格式有誤):")
        print(e.json(indent=2))  # 印出詳細的錯誤原因
        # 進階這理還可以做 Retry 機制，把 e.json() 丟回去給 Gemini 叫它重寫
        return None


def main():
    # 1. 指定輸入檔案
    file_path = "/Volumes/Dean512G/gcp/demo.jpg"  # 請確認檔案存在
    mime_type = "image/jpeg"

    if not os.path.exists(file_path):
        print(f"找不到檔案: {file_path}")
        return

    # 2. 執行 OCR
    ocr_text = get_ocr_text(file_path, mime_type)
    # print("OCR 原始文字預覽:", ocr_text[:100])

    # 3. 呼叫 Gemini 進行 AI 分析
    ai_result = call_gemini_to_extract(ocr_text)

    if not ai_result:
        print("AI 解析失敗。")
        return

    print("\n" + "=" * 20 + " AI 提取結果 " + "=" * 20)
    print(json.dumps(ai_result, ensure_ascii=False, indent=2))
    print("=" * 50)

    # 4. 模擬「人類反饋 (Human-in-the-Loop)」流程
    user_input = input("\n結果是否正確？(y/n) 若輸入 n 將進入修正模式: ")

    import shutil
    import subprocess
    import tempfile

    if user_input.lower() == "n":
        print("\n正在打開編輯器供您修正 JSON...")

        # 1. 建立暫存檔，並寫入目前的 AI 結果作為基底
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(ai_result, tmp, ensure_ascii=False, indent=2)
            tmp_path = tmp.name

        # 2. 嘗試開啟編輯器 (Mac 使用 'open', 或嘗試 'code' VS Code)
        # 如果您希望指定編輯器，可以修改這裡，例如 ["code", tmp_path]
        try:
            # 優先嘗試用 VS Code 開啟 (如果有的話)
            if shutil.which("code"):
                subprocess.call(["code", "--wait", tmp_path])
            else:
                # 否則使用系統預設編輯器 (Mac)
                subprocess.call(["open", "-t", tmp_path])
                input(
                    "請在編輯器中修正資料，完成後存檔並關閉視窗，然後在這裡按 Enter 繼續..."
                )
        except Exception as e:
            print(f"無法自動開啟編輯器: {e}")
            print(f"請手動編輯此檔案: {tmp_path}")
            input("修正完成後，請按 Enter 繼續...")

        # 3. 讀回修正後的內容
        try:
            with open(tmp_path, "r", encoding="utf-8") as f:
                corrected_json = json.load(f)

            # 4. 存入反饋庫
            # 注意: 此處傳入的是 ocr_text (原始問題) + corrected_json (正確答案)
            save_feedback(ocr_text, corrected_json)
            print("✅ 已讀取修正後的 JSON 並存入經驗庫，下次類似單據將會參考您的修正！")

        except json.JSONDecodeError:
            print("❌ 修正後的檔案格式不是有效的 JSON，本次反饋未保存。")
        except Exception as e:
            print(f"❌ 讀取檔案時發生錯誤: {e}")

        # 清理暫存檔
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    else:
        print("太棒了！無需修正。")


if __name__ == "__main__":
    main()
