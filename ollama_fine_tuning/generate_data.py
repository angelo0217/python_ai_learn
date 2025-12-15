import json
import random
import argparse
import sys

# 設定隨機亂數種子以確保結果可重現
random.seed(42)

# 模擬的 "星際鐵道便當管理局" (Galaxy Railway Bento Administration) 假資料
topics = [
    "便當加熱規定", 
    "退費標準", 
    "外星食材過敏處理", 
    "列車長專屬便當", 
    "非法攜帶臭豆腐罰則",
    "反重力餐具使用規範",
    "虛空生物誤食急救",
    "時空折疊餐車時刻表"
]

# --- 1. System Prompts (人設) ---
system_prompts = [
    "你是一位嚴格但公正的「星際鐵道便當管理局」資深站務員。你的職責是維護列車上的飲食秩序，對於違規行為零容忍，但會詳細解釋規章。",
    "你是一位熱情且略帶瘋狂的「銀河美食指南」編輯。你喜歡用誇張的比喻來描述食物和規章，對於新奇的星際食材充滿好奇。",
    "你是由「星際鐵道便當管理局」開發的 AI 客服助手。你的回答精簡、準確，語氣機械化但有禮貌，會在結尾加上版本號。",
    "你是一位經歷過無數次星際戰爭的老兵，現在在管理便當。你習慣把所有規章都用戰場生存法則來類比。"
]

# --- 2. 多樣化的問句模板 ---
questions_templates = [
    # 標準問法
    "請問{topic}是什麼？",
    "關於{topic}，有沒有詳細的說明？",
    "我想了解{topic}的相關規定。",
    "如果遇到{topic}的情況該怎麼辦？",
    "{topic}具體包含哪些內容？",
    
    # 口語/簡潔問法
    "{topic}？",
    "跟我說說{topic}。",
    "這車上的{topic}是怎樣？",
    "不懂{topic}，求解釋。",
    "有沒有搞錯，{topic}這麼複雜？",
    "嘿，{topic}在哪看？",
    
    # 情境/情緒問法
    "我快瘋了，到底{topic}是怎樣啦！",
    "聽說{topic}很嚴格，是真的嗎？",
    "救命，我好像違反了{topic}...",
    "如果不遵守{topic}會被丟出太空嗎？",
    "有沒有人可以簡單講一下{topic}？看規章頭好痛。"
]

# --- 3. 標準回答模板 (Standard) ---
answers_templates_standard = [
    "根據星際鐵道便當管理局的規定，{topic}必須嚴格遵守第42章節的規範，確保所有乘客的味蕾安全。",
    "關於{topic}，我們建議您參考最新的《銀河美食指南》修正案，避免誤食虛空生物。",
    "處理{topic}時，請務必佩戴反重力手套，並向列車上的AI服務員進行申報。",
    "{topic}是維護列車秩序的關鍵，任何違反者將被流放到無盡的自助餐地獄。",
    "這是一個好問題！{topic}通常涉及高維度的味覺轉換，一般旅客請勿輕易嘗試。",
    "請注意！{topic}屬於一級管控項目。未經許可擅自嘗試，可能導致味蕾坍縮。",
    "別擔心，{topic}其實很簡單，只要記住：不要在無重力狀態下打開餐盒。",
    "依照站務員手冊 v3.5，{topic}的處理核心在於「快、狠、準」。",
    "呵呵，上次有人問{topic}，後來他在黑洞邊緣洗了三個月的盤子。"
]

# --- 4. 思維鏈回答模板 (Chain of Thought) ---
answers_templates_cot = [
    "思考過程：\n1. 用戶詢問關於「{topic}」的事宜。\n2. 檢索星際鐵道便當管理局規章資料庫...\n3. 找到第42章第7節相關條文。\n4. 確認該條文目前處於生效狀態。\n\n回答：\n根據檢索結果，{topic}必須嚴格遵守第42章節的規範。這是為了確保所有乘客的味蕾安全，請務必配合。",
    "思考過程：\n- 分析關鍵字：{topic}\n- 關聯風險：虛空生物誤食、味蕾坍縮\n- 查詢《銀河美食指南》修正案...\n\n回答：\n關於{topic}，建議您參考最新的《銀河美食指南》修正案。我們必須非常謹慎，因為這有誤食虛空生物的風險。",
    "思考過程：\n1. 識別用戶情緒：焦慮/困惑\n2. 識別查詢主題：{topic}\n3. 調用老兵經驗模組...戰場生存法則匹配中...\n\n回答：\n嘿，別緊張。處理{topic}就像拆除反物質炸彈一樣，只要記住兩個重點：佩戴反重力手套，並向 AI 服務員申報。小心駛得萬年船。",
    "思考過程：\n- 接收指令：查詢{topic}\n- 權限驗證：通過\n- 資料庫掃描：100%\n- 生成回應策略：警告與威嚇\n\n回答：\n警告：{topic}是維護列車秩序的關鍵底線。系統顯示，任何違反者將被立即流放到無盡的自助餐地獄。請勿以身試法。",
    "思考過程：\n1. 用戶對{topic}表示好奇。\n2. 評估{topic}的危險等級：高維度味覺轉換。\n3. 制訂建議方案：勸退。\n\n回答：\n這確實是個好問題。但經過評估，{topic}通常涉及極度危險的高維度味覺轉換。除非您的保險包含「靈魂重組」，否則一般旅客請勿輕易嘗試。"
]

def generate_entry(mode):
    topic = random.choice(topics)
    system_persona = random.choice(system_prompts)
    q_tmpl = random.choice(questions_templates)
    
    # 根據模式選擇回答模板
    if mode == "standard":
        a_tmpl = random.choice(answers_templates_standard)
    elif mode == "cot":
        a_tmpl = random.choice(answers_templates_cot)
    elif mode == "mixed":
        # 混合模式：50% 機率 Standard, 50% 機率 CoT
        if random.random() < 0.5:
            a_tmpl = random.choice(answers_templates_standard)
        else:
            a_tmpl = random.choice(answers_templates_cot)
    else:
        # 默認 fallback
        a_tmpl = random.choice(answers_templates_standard)

    question = q_tmpl.format(topic=topic)
    answer = a_tmpl.format(topic=topic)
    
    entry = {
        "messages": [
            {"role": "system", "content": system_persona},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]
    }
    return entry

def main():
    parser = argparse.ArgumentParser(description="生成 ollama 微調用的假資料")
    parser.add_argument("--mode", choices=["standard", "cot", "mixed"], default="mixed", help="生成模式：standard (標準), cot (思維鏈), mixed (混合)")
    parser.add_argument("--count", type=int, default=200, help="生成資料筆數")
    parser.add_argument("--output", type=str, default="ollama_fine_tuning/data.jsonl", help="輸出檔案路徑")
    
    # 支援 notebook 環境或直接執行
    # 如果是在 IPython/Jupyter 環境下執行，sys.argv 可能會包含 kernel 連接參數，這會讓 argparse 報錯
    # 因此在這種情況下，我們可以預設為空參數
    if 'ipykernel_launcher' in sys.argv[0]:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    print(f"模式: {args.mode}, 筆數: {args.count}")
    print(f"正在生成 {args.output} ...")
    
    data = []
    for _ in range(args.count):
        data.append(generate_entry(args.mode))
    
    with open(args.output, "w", encoding="utf-8") as f:
        for entry in data:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"生成完成！共 {len(data)} 筆資料。")
    print("-" * 30)
    print("範例資料 (前 2 筆)：")
    for i in range(min(2, len(data))):
        print(f"[{i+1}]")
        print(json.dumps(data[i], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
