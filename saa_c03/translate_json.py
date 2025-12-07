import json
import time
from deep_translator import GoogleTranslator


def translate_text(text, target_lang='zh-TW'):
    """
    使用 Google Translate 翻譯文字。
    為了避免請求過多被封鎖，加入簡單的錯誤處理和重試機制。
    """
    if not text:
        return text

    translator = GoogleTranslator(source='auto', target=target_lang)
    try:
        # 分段翻譯以避免長度限制 (如果文字非常長)
        if len(text) > 4500:
            parts = [text[i:i + 4500] for i in range(0, len(text), 4500)]
            translated_parts = []
            for part in parts:
                translated_parts.append(translator.translate(part))
                time.sleep(0.5)
            return "".join(translated_parts)

        return translator.translate(text)
    except Exception as e:
        print(f"翻譯錯誤: {e}")
        return text


def main():
    input_file = 'saa_c03/saa_c03_en.json'
    output_file = 'saa_c03/saa_c03_translated_full.json'

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"成功讀取 {len(data)} 個項目。正在處理...")

        # 1. 修正排序
        # 假設 'item' 是數字，根據 item ID 進行排序
        data.sort(key=lambda x: x.get('item', 0))

        # 2. 重新編號 (可選，確保是 1, 2, 3... 連續)
        for index, item in enumerate(data, 1):
            item['item'] = index

        print("排序與重新編號完成。開始翻譯 (這可能需要一段時間)...")

        # 3. 翻譯內容
        total = len(data)
        for i, item in enumerate(data):
            print(f"正在翻譯第 {i + 1}/{total} 題...")

            # 翻譯問題
            item['question'] = translate_text(item.get('question', ''))

            # 翻譯選項
            new_options = []
            for opt in item.get('options', []):
                new_options.append(translate_text(opt))
            item['options'] = new_options

            # 為了避免 API 限制，建議每題稍微暫停一下
            time.sleep(0.2)

        # 4. 寫入新檔案
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"完成！已輸出至 {output_file}")

    except FileNotFoundError:
        print(f"找不到檔案: {input_file}，請確認檔案名稱是否正確。")
    except Exception as e:
        print(f"發生未預期的錯誤: {e}")


if __name__ == "__main__":
    main()