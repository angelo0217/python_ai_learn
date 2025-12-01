import fitz  # PyMuPDF
import re
import os

def parse_pdf_with_answers_to_markdown():
    """
    Reads the 'saa_c03_ans.pdf' file, extracts questions, options, and answers,
    and writes them to 'saa_c03_ans.md'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(script_dir, 'saa_c03_ans.pdf')
    output_path = os.path.join(script_dir, 'saa_c03_ans.md')

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

    # Clean up text by removing common headers/footers.
    full_text = re.sub(r'^(.*)www\.examtopics\.com(.*)$', '', full_text, flags=re.MULTILINE)
    full_text = re.sub(r'^\s*Get latest updates on www\.examtopics\.com\s*$', '', full_text, flags=re.MULTILINE)

    # Use regex to find the start of each block that looks like a real question
    # This avoids matching header text that might contain "Question #"
    question_starts = [match.start() for match in re.finditer(r'Question #\d+', full_text)]

    if not question_starts:
        print("錯誤：在文件中找不到任何 'Question #[number]' 格式的問題。")
        return

    # Create blocks from the found positions
    question_blocks = []
    for i in range(len(question_starts)):
        start_pos = question_starts[i]
        end_pos = question_starts[i + 1] if i + 1 < len(question_starts) else len(full_text)
        question_blocks.append(full_text[start_pos:end_pos])

    markdown_content = []

    for i, block in enumerate(question_blocks, 1):
        # The question text is from "Question #" until the first option.
        question_match = re.search(r'^(Question #\d+.*?)(?=\n\s*[A-Z]\.)', block, re.DOTALL | re.MULTILINE)
        if not question_match:
            print(f"警告：無法解析第 {i} 個問題的題目（內容可能格式不符）。")
            continue

        question_text_full = question_match.group(1).strip().replace('\n', ' ')
        # Remove the "Question #X" part for cleaner output
        question_text = re.sub(r'Question #\d+\s*', '', question_text_full)

        # Extract options. This pattern finds all lines starting with an option letter.
        options = re.findall(r'^\s*([A-Z]\..*)', block, re.MULTILINE)

        # Extract answer. The answer is usually at the end.
        answer_match = re.search(r'Answer:\s*([A-Z])', block, re.IGNORECASE)
        answer = answer_match.group(1).upper() if answer_match else "未找到"

        # Start building markdown for the current question
        # Use the question number from the text itself for accuracy
        q_num_match = re.search(r'Question #(\d+)', question_text_full)
        q_num = q_num_match.group(1) if q_num_match else i
        markdown_content.append(f"### 題目 {q_num}\n")
        markdown_content.append(f"**問題：** {question_text.strip()}\n")

        if options:
            markdown_content.append("**選項：**\n")
            for option in options:
                markdown_content.append(f"- {option.strip()}\n")
        
        markdown_content.append(f"\n**答案：** {answer}\n")
        markdown_content.append("\n---\n")

    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("".join(markdown_content))
        
        if len(markdown_content) > 0:
            print(f"成功！已將 {len(question_blocks)} 個問題寫入 '{output_path}'")
        else:
            print(f"處理完成，但未成功解析任何問題。請檢查 PDF 格式和內容。")

    except IOError as e:
        print(f"寫入檔案時發生錯誤: {e}")

if __name__ == '__main__':
    parse_pdf_with_answers_to_markdown()
