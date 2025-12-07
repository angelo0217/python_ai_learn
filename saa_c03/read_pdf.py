import fitz  # PyMuPDF
import re
import os
import json

def parse_pdf_with_answers_to_files():
    """
    Reads 'saa_c03_ans.pdf', extracts questions, options, and answers,
    and writes them to 'saa_c03_ans.md' and 'saa_c03_en.json'.
    This version includes robust parsing for multi-line content and merged options.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'saa_c03_ans.pdf')
    md_output_path = os.path.join(script_dir, 'saa_c03_ans.md')
    json_output_path = os.path.join(script_dir, 'saa_c03_en.json')

    if not os.path.exists(pdf_path):
        print(f"錯誤：找不到 PDF 檔案 '{pdf_path}'")
        return

    full_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                full_text += page.get_text()
    except Exception as e:
        print(f"讀取或解析 PDF 時發生錯誤: {e}")
        return

    # Clean up headers/footers
    full_text = re.sub(r'^(.*)www\.examtopics\.com(.*)$', '', full_text, flags=re.MULTILINE)
    full_text = re.sub(r'^\s*Get latest updates on www\.examtopics\.com\s*$', '', full_text, flags=re.MULTILINE)

    # Split text into blocks for each question
    question_starts = [match.start() for match in re.finditer(r'Question #\d+', full_text)]

    if not question_starts:
        print("錯誤：在文件中找不到任何 'Question #[number]' 格式的問題。")
        return

    question_blocks = []
    for i in range(len(question_starts)):
        start_pos = question_starts[i]
        end_pos = question_starts[i + 1] if i + 1 < len(question_starts) else len(full_text)
        question_blocks.append(full_text[start_pos:end_pos])

    markdown_content = []
    json_data = []

    for i, block in enumerate(question_blocks, 1):
        # 1. Broadly find the answer line to correctly delimit the content block
        answer_line_match = re.search(r'^(?:Correct\s)?Answer:', block, re.IGNORECASE | re.MULTILINE)
        
        content_end = answer_line_match.start() if answer_line_match else len(block)
        content_block = block[:content_end].strip()

        # 2. Parse options from the content block
        option_markers = list(re.finditer(r'\b([A-Z])\.\s', content_block))
        option_letters = {marker.group(1) for marker in option_markers}

        if not option_markers:
            # If no options found, the whole block is the question.
            question_text_full = content_block
            options = []
        else:
            # The question is everything before the first option marker.
            first_option_start_pos = option_markers[0].start()
            question_text_full = content_block[:first_option_start_pos].strip()

            # The rest of the block contains the options.
            options_text = content_block[first_option_start_pos:]
            
            # Split the options text by the markers. The lookahead `(?=...)` keeps the delimiter.
            raw_options = re.split(r'(?=\b[A-Z]\.\s)', options_text)
            
            # Clean up each option and handle multiline content.
            options = [opt.replace('\n', ' ').strip() for opt in raw_options if opt and opt.strip()]
        
        # 3. Now, precisely parse the answer value from the whole block
        answer_match = re.search(r'(?:Correct\s)?Answer:\s*(.*?)\s*$', block, re.IGNORECASE | re.MULTILINE)
        if answer_match:
            answer_text = answer_match.group(1).upper()
            # Extract only capital letters from the answer string
            cleaned_answer_text = re.sub(r'[^A-Z]', '', answer_text)
            
            is_valid = True
            if not cleaned_answer_text:
                is_valid = False
            
            # Validate that all characters in the answer are actual option letters.
            # This prevents misinterpreting words like "NONE" or "SEE" as answers.
            if is_valid and option_letters:
                for char in cleaned_answer_text:
                    if char not in option_letters:
                        is_valid = False
                        break
            
            if is_valid:
                answers = list(cleaned_answer_text)
            else:
                answers = ["未找到"]
        else:
            answers = ["未找到"]

        # Extract question number and clean the question text
        q_num_match = re.search(r'Question #(\d+)', question_text_full)
        q_num = int(q_num_match.group(1)) if q_num_match else i
        question_text = re.sub(r'Question #\d+\s*', '', question_text_full, count=1).replace('\n', ' ').strip()

        # --- Populate Markdown content ---
        markdown_content.append(f"### 題目 {q_num}\n")
        markdown_content.append(f"**問題：** {question_text}\n")

        if options:
            markdown_content.append("**選項：**\n")
            for option in options:
                markdown_content.append(f"- {option}\n")
        
        markdown_content.append(f"\n**答案：** {', '.join(answers)}\n")
        markdown_content.append("\n---\n")

        # --- Populate JSON data ---
        json_item = {
            "item": q_num,
            "question": question_text,
            "options": options,
            "answer": answers if "未找到" not in answers else []
        }
        json_data.append(json_item)

    # --- Write Markdown file ---
    try:
        with open(md_output_path, 'w', encoding='utf-8') as f:
            f.write("".join(markdown_content))
        
        if len(markdown_content) > 0:
            print(f"成功！已將 {len(question_blocks)} 個問題寫入 '{md_output_path}'")
        else:
            print(f"處理完成，但未成功解析任何問題並寫入MD。請檢查 PDF 格式和內容。")

    except IOError as e:
        print(f"寫入 Markdown 檔案時發生錯誤: {e}")

    # --- Write JSON file ---
    try:
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        if len(json_data) > 0:
            print(f"成功！已將 {len(json_data)} 個問題寫入 '{json_output_path}'")
        else:
            print(f"處理完成，但未成功解析任何問題並寫入JSON。請檢查 PDF 格式和內容。")
            
    except IOError as e:
        print(f"寫入 JSON 檔案時發生錯誤: {e}")

if __name__ == '__main__':
    parse_pdf_with_answers_to_files()
