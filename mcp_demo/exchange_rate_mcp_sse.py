from typing import Any, List, Dict, Optional
import httpx
import uvicorn
from datetime import datetime, timedelta
from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server

# --- Configuration ---
FINMIND_API_BASE = "https://api.finmindtrade.com/api/v3/data"
DEFAULT_TIMEOUT = 30.0

# Mock data for store information
DEFAULT_STORE_DATA = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2, "currency": "台幣"},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0, "currency": "日幣"},
}

# Initialize FastMCP server
mcp = FastMCP("exchange_rate")

async def fetch_finmind_data(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch data from FinMind API with error handling.
    
    Args:
        params: Query parameters for the API request.
        
    Returns:
        A list of data records if successful, otherwise an empty list.
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(FINMIND_API_BASE, params=params, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == 200:
                return data.get("data", [])
            
            print(f"FinMind API Error: {data.get('msg')}")
            return []
        except httpx.HTTPError as e:
            print(f"HTTP error occurred: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            return []

def format_exchange_rate(exchange_info: Dict[str, Any]) -> str:
    """
    Format a single exchange rate record into a human-readable string.
    """
    return (
        f"Date: {exchange_info.get('date', 'N/A')}, "
        f"Rate: {exchange_info.get('price', 'N/A')}"
    )

@mcp.tool()
async def get_exchange_rate(currency: str) -> str:
    """
    Get the latest exchange rate for a given currency.
    
    Args:
        currency: The currency code to fetch (e.g., 'USD').
    """
    # Note: In a real scenario, the API endpoint and params would be specific to currency exchange
    params = {"data_id": "FX_S", "currency": currency} 
    data = await fetch_finmind_data(params)
    
    if not data:
        return f"Could not retrieve exchange rate for {currency}."
    
    latest_rate = data[0]
    return f"The latest exchange rate for {currency} is: {format_exchange_rate(latest_rate)}"

@mcp.tool()
async def get_store_info(store_id: str) -> str:
    """
    Get information about a specific store.
    
    Args:
        store_id: The ID of the store (e.g., 'STORE1').
    """
    info = DEFAULT_STORE_DATA.get(store_id)
    if not info:
        return f"Store {store_id} not found."
    
    return (
        f"Store ID: {store_id}\n"
        f"Users: {info['user_cnt']}\n"
        f"Managers: {info['manager_cnt']}\n"
        f"Currency: {info['currency']}"
    )

# --- SSE Server Setup ---
async def handle_sse(request: Request):
    """Handle SSE transport for MCP."""
    async with SseServerTransport(request) as transport:
        # This is a simplified representation of how FastMCP integrates with SSE
        # In a full implementation, the server instance would be linked here
        await mcp.handle_request(transport)

app = Starlette(
    routes=[
        Route("/sse", endpoint=handle_sse),
        Mount("/mcp", app=mcp.fastapi_app) if hasattr(mcp, "fastapi_app") else Mount("/mcp", app=None)
    ]
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
