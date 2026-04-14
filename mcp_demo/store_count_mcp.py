import copy
import json
import logging
from typing import Any, Optional, Dict

from mcp.server.fastmcp import FastMCP

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# In-memory mock store data
DEFAULT_STORE = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0},
}

class StoreManager:
    """Handles store data operations with error handling."""
    
    def __init__(self, initial_data: Dict[str, Dict[str, int]]):
        self._data = copy.deepcopy(initial_data)

    def get_store_data(self, store_name: str) -> Optional[Dict[str, int]]:
        """
        Safely retrieves a copy of store data.
        """
        try:
            name = store_name.upper()
            return copy.deepcopy(self._data.get(name))
        except Exception as e:
            logger.error(f"Error retrieving store {store_name}: {e}")
            return None

    def update_count(self, store_name: str, key: str, delta: int) -> bool:
        """
        Updates the count for a specific store and key.
        """
        try:
            name = store_name.upper()
            if name not in self._data:
                logger.warning(f"Store {name} not found.")
                return False
            
            if key not in self._data[name]:
                logger.warning(f"Key {key} not found in store {name}.")
                return False
                
            self._data[name][key] += delta
            logger.info(f"Updated {name} {key} by {delta}. New value: {self._data[name][key]}")
            return True
        except Exception as e:
            logger.error(f"Failed to update store {store_name}: {e}")
            return False

# Initialize MCP Server
mcp = FastMCP("StoreCountServer")
store_manager = StoreManager(DEFAULT_STORE)

@mcp.tool()
def get_store_count(store_name: str) -> str:
    """
    Get the user and manager count for a specific store.
    """
    data = store_manager.get_store_data(store_name)
    if data is None:
        return f"Error: Store '{store_name}' not found or unavailable."
    
    return json.dumps({"store_name": store_name.upper(), "data": data}, indent=2)

@mcp.tool()
def update_store_count(store_name: str, key: str, delta: int) -> str:
    """
    Update the count for a specific store. Key should be 'user_cnt' or 'manager_cnt'.
    """
    success = store_manager.update_count(store_name, key, delta)
    if success:
        return f"Successfully updated {key} for {store_name} by {delta}."
    return f"Error: Failed to update {key} for {store_name}. Please check store name and key."

if __name__ == "__main__":
    mcp.run()
