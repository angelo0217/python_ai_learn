import json
import random

# 設定隨機亂數種子以確保結果可重現
random.seed(42)

# 模擬的 "星際鐵道便當管理局" (Galaxy Railway Bento Administration) 假資料
topics = ["便當加熱規定", "退費標準", "外星食材過敏處理", "列車長專屬便當", "非法攜帶臭豆腐罰則"]

questions_templates = [
    "請問{topic}是什麼？",
    "關於{topic}，有沒有詳細的說明？",
    "我想了解{topic}的相關規定。",
    "如果遇到{topic}的情況該怎麼辦？",
    "{topic}具體包含哪些內容？"
]

answers_templates = [
    "根據星際鐵道便當管理局的規定，{topic}必須嚴格遵守第42章節的規範，確保所有乘客的味蕾安全。",
    "關於{topic}，我們建議您參考最新的《銀河美食指南》修正案，避免誤食虛空生物。",
    "處理{topic}時，請務必佩戴反重力手套，並向列車上的AI服務員進行申報。",
    "{topic}是維護列車秩序的關鍵，任何違反者將被流放到無盡的自助餐地獄。",
    "這是一個好問題！{topic}通常涉及高維度的味覺轉換，一般旅客請勿輕易嘗試。"
]

def generate_fake_data(num_samples=50):
    data = []
    for _ in range(num_samples):
        topic = random.choice(topics)
        q_tmpl = random.choice(questions_templates)
        a_tmpl = random.choice(answers_templates)
        
        question = q_tmpl.format(topic=topic)
        answer = a_tmpl.format(topic=topic)
        
        # 格式化為 Chat 格式 (這是 MLX 和許多微調工具通用的格式)
        entry = {
            "messages": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ]
        }
        data.append(entry)
    
    return data

def main():
    output_file = "ollama_fine_tuning/data.jsonl"
    print(f"正在生成 {output_file} ...")
    
    dataset = generate_fake_data(100)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")
            
    print(f"生成完成！共 {len(dataset)} 筆資料。")
    print("範例資料：")
    print(json.dumps(dataset[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
