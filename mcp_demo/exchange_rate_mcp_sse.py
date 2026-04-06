from typing import Any, List, Dict
import httpx
import logging
from mcp.server.fastmcp import FastMCP
from datetime import datetime

# 設定日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("exchange_rate_mcp")

# 初始化 FastMCP 伺服器
# FastMCP 內建支援 SSE 傳輸，無需手動配置 Starlette/SseServerTransport 除非有極其特殊的路由需求
mcp = FastMCP("ExchangeRateServer")

# 常數
FINMIND_API_BASE = "https://api.finmindtrade.com/api/v3/data"

async def fetch_finmind_data(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """向 FinMind API 發送請求並處理回應。"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(FINMIND_API_BASE, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == 200:
                return data.get("data", [])
            logger.error(f"FinMind API Error: {data.get('msg')}")
            return []
        except httpx.HTTPError as e:
            logger.error(f"HTTP Error: {e}")
            return []
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            return []

@mcp.tool()
async def get_exchange_rate(currency: str = "USD", days: int = 5) -> str:
    """
    獲取指定貨幣的最新匯率資訊。
    
    Args:
        currency: 貨幣代碼 (例如: USD, JPY, EUR)
        days: 獲取最近幾天的資料 (預設 5 天)
    """
    # 計算日期範圍
    end_date = datetime.now().strftime("%Y-%m-%d")
    # 簡單處理日期，實際建議使用 relativedelta
    params = {
        "data_id": "FX_S",
        "currency": currency,
        "start_date": "2024-01-01", # 簡化處理，實際可根據 days 計算
        "end_date": end_date
    }
    
    data = await fetch_finmind_data(params)
    if not data:
        return f"無法獲取 {currency} 的匯率資料。"
    
    # 取最近的 N 天
    recent_data = data[-days:]
    results = []
    for entry in recent_data:
        results.append(f"日期: {entry['date']} | 匯率: {entry['price']}")
    
    return f"{currency} 最近 {days} 天匯率資訊:\n" + "\n".join(results)

if __name__ == "__main__":
    # FastMCP 預設提供便捷的啟動方式
    # 執行時可使用 `fastmcp run exchange_rate_mcp_sse.py` 
    # 或在程式碼中呼叫 mcp.run()
    mcp.run()
