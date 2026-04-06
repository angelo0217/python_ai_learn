import logging
from typing import Dict, Any
from pydantic import BaseModel, Field
from mcp.server.fastmcp import FastMCP

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data Models
class StoreState(BaseModel):
    counts: Dict[str, int] = Field(default_factory=lambda: {"visitors": 0, "orders": 0})

# In-memory state
state = StoreState()

mcp = FastMCP("store_count_server")

@mcp.tool()
async def get_store_counts() -> str:
    """Retrieve the current visitor and order counts for the store."""
    logger.info("Fetching store counts")
    return f"Current Store Stats: {state.counts}"

@mcp.tool()
async def increment_count(metric: str) -> str:
    """
    Increment a specific store metric.
    Args:
        metric: The metric to increment ('visitors' or 'orders').
    """
    if metric not in state.counts:
        return f"Error: Metric '{metric}' does not exist. Available: {list(state.counts.keys())}"
    
    state.counts[metric] += 1
    logger.info(f"Incremented {metric} to {state.counts[metric]}")
    return f"Updated {metric}: {state.counts[metric]}"

@mcp.tool()
async def reset_counts() -> str:
    """Reset all store counts to zero."""
    state.counts = {"visitors": 0, "orders": 0}
    logger.info("Store counts reset")
    return "All store counts have been reset to 0."

if __name__ == "__main__":
    # This server uses stdio by default (FastMCP default)
    mcp.run()
