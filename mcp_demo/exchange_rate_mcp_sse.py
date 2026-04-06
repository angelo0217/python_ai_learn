import logging
import httpx
import os
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP
from mcp_demo.mcp_utils import create_sse_app, run_sse_server

# Logging setup
logger = logging.getLogger(__name__)

# Data Models
class ExchangeRate(BaseModel):
    currency: str
    rate: float
    timestamp: str = "N/A"

class StoreConfig(BaseModel):
    name: str
    base_currency: str = "USD"
    exchange_rates: Dict[str, float] = Field(default_factory=dict)

# Mock Store Data
DEFAULT_STORE = StoreConfig(
    name="Global Tech Store",
    base_currency="USD",
    exchange_rates={"TWD": 32.5, "JPY": 150.2, "EUR": 0.92}
)

mcp = FastMCP("exchange_rate_server")

class FinMindClient:
    """Client to handle FinMind API requests."""
    BASE_URL = "https://api.finmindapi.com/v4/Forex"

    async def get_rate(self, currency: str) -> Optional[float]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Note: In a real scenario, we would use specific FinMind API endpoints
                # This is a simplified simulation of the API call
                logger.info(f"Fetching rate for {currency} from FinMind...")
                # Simulating API response for demo purposes
                mock_rates = {"TWD": 32.45, "JPY": 151.1, "EUR": 0.91}
                return mock_rates.get(currency.upper())
        except Exception as e:
            logger.error(f"FinMind API Error: {e}")
            return None

finmind_client = FinMindClient()

@mcp.tool()
async def get_current_exchange_rate(currency: str) -> str:
    """
    Fetch the latest exchange rate for a given currency relative to USD.
    Args:
        currency: The 3-letter currency code (e.g., 'TWD', 'JPY').
    """
    rate = await finmind_client.get_rate(currency)
    if rate:
        return f"The current exchange rate for {currency.upper()} is 1 USD = {rate} {currency.upper()}."
    return f"Could not retrieve exchange rate for {currency}."

@mcp.tool()
async def convert_price(amount: float, from_currency: str, to_currency: str) -> str:
    """
    Convert a price from one currency to another.
    Args:
        amount: The amount to convert.
        from_currency: Source currency code.
        to_currency: Target currency code.
    """
    # Use FinMind for real-time or fallback to store config
    rate_from = await finmind_client.get_rate(from_currency) or DEFAULT_STORE.exchange_rates.get(from_currency, 1.0)
    rate_to = await finmind_client.get_rate(to_currency) or DEFAULT_STORE.exchange_rates.get(to_currency, 1.0)
    
    # Convert from_currency -> USD -> to_currency
    usd_amount = amount / rate_from
    final_amount = usd_amount * rate_to
    
    return f"{amount} {from_currency.upper()} is approximately {final_amount:.2f} {to_currency.upper()}."

if __name__ == "__main__":
    app, host, port = create_sse_app(mcp)
    run_sse_server(app, host, port)
