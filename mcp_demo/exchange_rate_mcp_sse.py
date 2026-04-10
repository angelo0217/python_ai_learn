from typing import Any, List, Dict, Optional, Union
import asyncio
import json
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse
from mcp.server import Server
import uvicorn
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from core.config import settings
from core.logger import logger
from core.exceptions import APIError

# 初始化 FastMCP 伺服器
mcp = FastMCP("exchange_rate")

class ExchangeRateService:
    """封裝匯率 API 操作的 Service 類別"""
    def __init__(self, api_base: str = "https://api.finmindtrade.com/api/v3/data"):
        self.api_base = api_base

    async def fetch_data(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        向 FinMind API 發送請求並處理回應。
        :param params: 請求參數
        :return: 數據列表
        :raises APIError: 當 API 回應錯誤或網路失敗時拋出
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(self.api_base, params=params, timeout=30.0)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == 200:
                    return data.get("data", [])
                else:
                    msg = data.get("msg", "Unknown API Error")
                    logger.error(f"FinMind API Error: {msg}")
                    raise APIError(f"FinMind API returned error: {msg}")
                    
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP Status Error: {e}")
                raise APIError(f"HTTP error occurred: {e.response.status_code}")
            except httpx.RequestError as e:
                logger.error(f"HTTP Request Error: {e}")
                raise APIError(f"Network error occurred while fetching exchange rates: {str(e)}")
            except Exception as e:
                logger.exception(f"Unexpected error in fetch_data: {e}")
                raise APIError(f"An unexpected error occurred: {str(e)}")

# 實例化 Service
exchange_service = ExchangeRateService()

def format_exchange_rate(exchange_info: Dict[str, Any]) -> str:
    """格式化單一匯率資訊為可讀字串。"""
    return (
        f"日期：{exchange_info.get('date', 'N/A')}\n"
        f"匯率：{exchange_info.get('value', 'N/A')}"
    )

@mcp.tool()
async def get_exchange_rate(currency_pair: str = "USD") -> str:
    """
    獲取特定貨幣對的最新匯率。
    :param currency_pair: 貨幣對名稱 (例如 'USD')
    """
    try:
        params = {"data_id": "FX_S", "currency": currency_pair}
        data = await exchange_service.fetch_data(params)
        
        if not data:
            return "No exchange rate data found for the given currency."
            
        # 取得最新的一筆數據
        latest = data[0] if data else {}
        return format_exchange_rate(latest)
        
    except APIError as e:
        return f"API Error: {str(e)}"
    except Exception as e:
        logger.exception(f"Unexpected error in get_exchange_rate: {e}")
        return f"Unexpected Error: {str(e)}"

# --- SSE Server Setup ---
async def main():
    server = Server("exchange_rate")
    sse = SseServerTransport("/messages", server)
    
    app = Starlette(
        routes=[
            Route("/sse", endpoint=sse.handle_sse),
            Route("/messages", endpoint=sse.handle_post_message, methods=["POST"]),
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
        ]
    )
    
    logger.info(f"Starting Exchange Rate MCP SSE server on port {settings.MCP_PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=settings.MCP_PORT)

if __name__ == "__main__":
    asyncio.run(main())
