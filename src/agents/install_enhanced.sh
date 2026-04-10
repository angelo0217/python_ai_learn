#!/bin/bash
# 基於 Gemini 的代碼生成與優化系統（增強版）安裝腳本

echo "開始安裝基於 Gemini 的代碼生成與優化系統（增強版）所需依賴..."

# 檢查 Python 版本
python --version

# 安裝核心依賴
echo "安裝核心依賴..."
pip install -U google-generativeai=="0.7.1"

# 安裝其他依賴
echo "安裝額外依賴..."
pip install -U pandas matplotlib numpy

echo "所有依賴安裝完成！"
echo ""
echo "使用說明："
echo "1. 設置 Gemini API 密鑰"
echo "   export GEMINI_API_KEY='your-api-key-here'  # Linux/Mac"
echo "   set GEMINI_API_KEY=your-api-key-here       # Windows CMD"
echo "   \$env:GEMINI_API_KEY='your-api-key-here'    # Windows PowerShell"
echo ""
echo "2. 運行增強版代碼生成系統："
echo "   python gemini_agents_enhanced.py"
echo ""
echo "3. 運行示例腳本："
echo "   python enhanced_examples.py"
echo ""
echo "4. 所有生成的代碼將自動保存到以下格式的目錄："
echo "   generated_code_YYYYMMDD_HHMMSS_randomID/"
