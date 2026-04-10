"""
基於 Gemini 2.5 Pro 的代碼生成與優化系統 - 增強版使用示例

這個腳本提供了一個簡單的界面來使用增強版的代碼生成系統。
"""

import os
import sys
from importlib import import_module


def run_data_analyzer_task():
    """執行數據分析器任務"""
    print("執行數據分析器任務示例...")

    # 導入模塊
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gemini_agents_enhanced import CodeGenerationSystem

    # 創建代碼生成系統
    system = CodeGenerationSystem(
        task_name="data_analyzer",
        rounds=2,
        temperature=0.3,
        task_complexity="standard",
        review_depth="standard",
        save_history=True,
    )

    # 運行系統
    results = system.run()

    print("\n數據分析器任務示例執行完成！")
    print(f"生成的代碼保存在: {results['output_directory']}")
    return results


def run_web_scraper_task():
    """執行網頁爬蟲任務"""
    print("執行網頁爬蟲任務示例...")

    # 導入模塊
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gemini_agents_enhanced import CodeGenerationSystem

    # 創建代碼生成系統
    system = CodeGenerationSystem(
        task_name="web_scraper",
        rounds=2,
        temperature=0.4,
        task_complexity="advanced",
        review_depth="detailed",
        save_history=True,
    )

    # 運行系統
    results = system.run()

    print("\n網頁爬蟲任務示例執行完成！")
    print(f"生成的代碼保存在: {results['output_directory']}")
    return results


def run_api_server_task():
    """執行API服務器任務"""
    print("執行API服務器任務示例...")

    # 導入模塊
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gemini_agents_enhanced import CodeGenerationSystem

    # 創建代碼生成系統
    system = CodeGenerationSystem(
        task_name="api_server",
        rounds=3,
        temperature=0.3,
        task_complexity="advanced",
        review_depth="detailed",
        save_history=True,
    )

    # 運行系統
    results = system.run()

    print("\nAPI服務器任務示例執行完成！")
    print(f"生成的代碼保存在: {results['output_directory']}")
    return results


def run_custom_task():
    """執行自定義任務"""
    print("執行自定義任務示例...")

    # 導入模塊
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gemini_agents_enhanced import CodeGenerationSystem

    # 自定義任務描述
    custom_task = """
    請編寫一個 Python 程式，實現以下功能：
    1. 創建一個名為 'FileOrganizer' 的類，可以按照文件類型、大小或修改日期整理文件夾
    2. 實現以下方法：
       - organize_by_type(): 按文件類型（擴展名）分類文件
       - organize_by_size(): 按文件大小分類文件（小、中、大）
       - organize_by_date(): 按修改日期分類文件（今天、本週、本月、更早）
       - generate_report(): 生成整理報告
    3. 支持遞歸處理子文件夾
    4. 實現進度顯示
    5. 包含適當的錯誤處理和日誌記錄
    6. 提供命令行接口，可接受參數指定要整理的目錄、整理方式等
    """

    # 創建代碼生成系統
    system = CodeGenerationSystem(
        task=custom_task,
        rounds=2,
        temperature=0.3,
        task_complexity="advanced",
        review_depth="detailed",
        save_history=True,
    )

    # 運行系統
    results = system.run()

    print("\n自定義任務示例執行完成！")
    print(f"生成的代碼保存在: {results['output_directory']}")
    return results


def show_usage_examples():
    """顯示命令行使用示例"""
    print("\n命令行使用示例：")
    print("\n基本用法：")
    print("python gemini_agents_enhanced.py")

    print("\n使用預定義任務：")
    print("python gemini_agents_enhanced.py --task_name data_analyzer --rounds 2")
    print(
        "python gemini_agents_enhanced.py --task_name web_scraper --temperature 0.4 --review_depth detailed"
    )
    print(
        "python gemini_agents_enhanced.py --task_name api_server --task_complexity advanced --rounds 3"
    )

    print("\n使用自定義任務（將任務保存在文件中）：")
    print(
        'python gemini_agents_enhanced.py --task "$(cat custom_task.txt)" --rounds 3 --review_depth detailed'
    )


if __name__ == "__main__":
    print("歡迎使用基於 Gemini 的代碼生成與優化系統（增強版）示例腳本！")
    print("注意：運行這些示例前，請確保已設置 GEMINI_API_KEY 環境變數")

    # 顯示使用選項
    print("\n請選擇要運行的示例：")
    print("1. 數據分析器任務")
    print("2. 網頁爬蟲任務")
    print("3. API服務器任務")
    print("4. 自定義文件組織器任務")
    print("5. 顯示命令行使用示例")
    print("6. 退出")

    try:
        choice = input("\n請輸入選項 (1-6): ")

        if choice == "1":
            run_data_analyzer_task()
        elif choice == "2":
            run_web_scraper_task()
        elif choice == "3":
            run_api_server_task()
        elif choice == "4":
            run_custom_task()
        elif choice == "5":
            show_usage_examples()
        elif choice == "6":
            print("退出示例腳本")
        else:
            print("無效選項，請選擇 1-6")
    except KeyboardInterrupt:
        print("\n程序已中斷")
    except Exception as e:
        print(f"\n運行示例時出錯：{e}")
