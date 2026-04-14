import logging
import asyncio
from typing import Any, List, Dict, Optional
from datetime import datetime

import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("exchange_rate_mcp")

# Constants
FINMIND_API_BASE = "https://api.finmindtrade.com/api/v3/data"

class ExchangeRateService:
    """Service to handle exchange rate data fetching and processing."""
    
    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def fetch_finmind_data(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Fetch data from FinMind API with error handling and logging.
        """
        async with httpx.AsyncClient() as client:
            try:
                logger.info(f"Fetching data from FinMind with params: {params}")
                response = await client.get(FINMIND_API_BASE, params=params, timeout=self.timeout)
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") == 200:
                    return data.get("data", [])
                
                logger.error(f"FinMind API returned error status: {data.get('status')} - {data.get('msg')}")
                return []
                
            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP status error occurred: {e.response.status_code} - {e.response.text}")
            except httpx.RequestError as e:
                logger.error(f"HTTP request error occurred: {e}")
            except Exception as e:
                logger.exception(f"Unexpected error occurred while fetching FinMind data: {e}")
            
            return []

    def format_exchange_rate(self, exchange_info: Dict[str, Any]) -> str:
        """Format single exchange rate info into a readable string."""
        try:
            return (
                f"日期：{exchange_info['date']}\n"
                f"匯率：{exchange_info['price']}"
            )
        except KeyError as e:
            logger.warning(f"Missing key in exchange info: {e}")
            return "Invalid exchange rate data format."

# Initialize FastMCP server
mcp = FastMCP("exchange_rate")
exchange_service = ExchangeRateService()

@mcp.tool()
async def get_exchange_rate(currency: str = "USD") -> str:
    """
    Get the latest exchange rate for a given currency (default USD).
    """
    params = {
        "data_id": "FX_RATE",
        "currency": currency,
        "limit": 1
    }
    
    try:
        data = await exchange_service.fetch_finmind_data(params)
        if not data:
            return f"Could not retrieve exchange rate for {currency}. Please check the currency code."
        
        latest_rate = data[0]
        return exchange_service.format_exchange_rate(latest_rate)
    except Exception as e:
        logger.exception(f"Error in get_exchange_rate tool: {e}")
        return "An internal error occurred while fetching the exchange rate."

# SSE Server Setup
async def handle_sse(request: Request):
    async with SseServerTransport(request) as transport:
        # Note: In a real production scenario, the server instance 
        # should be managed properly to avoid repeated initialization.
        await mcp.handle_request(transport)

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/mcp", app=mcp.app),
    ]
)

if __name__ == "__main__":
    logger.info("Starting Exchange Rate MCP Server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
