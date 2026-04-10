from typing import Any, List, Dict
import httpx
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from mcp.server.sse import SseServerTransport
from starlette.requests import Request
from starlette.routing import Mount, Route
from mcp.server import Server
import uvicorn
from datetime import datetime, timedelta

# 初始化 FastMCP 伺服器，命名為 "exchange_rate"
mcp = FastMCP("exchange_rate")

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
            else:
                print(f"FinMind API Error: {data.get('msg')}")
                return []
        except httpx.HTTPError as e:
            print(f"HTTP 錯誤：{e}")
            return []
        except Exception as e:
            print(f"發生未預期的錯誤：{e}")
            return []


default_store = {
    "STORE1": {"user_cnt": 18, "manager_cnt": 2, "currency": "台幣"},
    "STORE2": {"user_cnt": 20, "manager_cnt": 0, "currency": "日幣"},
}


def format_exchange_rate(exchange_info: Dict[str, Any]) -> str:
    """格式化單一匯率資訊為可讀字串。"""
    return (
        f"日期：{exchange_info['date']}\n"
        f"貨幣：{exchange_info['currency']}\n"
        f"現金買入價：{exchange_info.get('cash_buy', 'N/A')}\n"
        f"現金賣出價：{exchange_info.get('cash_sell', 'N/A')}\n"
        f"即期買入價：{exchange_info.get('spot_buy', 'N/A')}\n"
        f"即期賣出價：{exchange_info.get('spot_sell', 'N/A')}"
    )


@mcp.tool()
async def get_exchange_rate(currency: str, date: str) -> str:
    """取得指定日期特定貨幣兌台幣的匯率。

    Args:
        currency: 三位字母的貨幣代碼 (例如：USD, JPY)。
        date: 'YYYY-MM-DD' 格式的日期。
    """
    params = {
        "dataset": "TaiwanExchangeRate",
        "data_id": currency.upper(),
        "date": date,
    }
    exchange_data = await fetch_finmind_data(params)
    if exchange_data:
        exchange_info = exchange_data[0]
        return format_exchange_rate(exchange_info)
    else:
        return f"無法取得 {currency} 在 {date} 的匯率資訊。"


@mcp.tool()
async def get_latest_exchange_rates(currency: str, days: int = 5) -> str:
    """取得指定貨幣最近幾天的匯率。

    Args:
        currency: 三位字母的貨幣代碼 (例如：USD, JPY)。
        days: 要查詢的天數 (預設為 5)。
    """
    today = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days - 1)).strftime("%Y-%m-%d")

    params = {
        "dataset": "TaiwanExchangeRate",
        "data_id": currency.upper(),
        "start_date": start_date,
        "end_date": today,
    }
    exchange_data = await fetch_finmind_data(params)

    if exchange_data:
        formatted_rates = "\n---\n".join(format_exchange_rate(item) for item in exchange_data)
        return formatted_rates
    else:
        return f"無法取得 {currency} 最近 {days} 天的匯率資訊。"


@mcp.tool()
async def get_store_info(store_id: str) -> str:
    """查詢指定店鋪的人數和幣別資訊。

    Args:
        store_id: 店鋪 ID (例如：STORE1, STORE2)。
    """
    store_id = store_id.upper()
    if store_id in default_store:
        store_data = default_store[store_id]
        total_people = store_data["user_cnt"] + store_data["manager_cnt"]
        return (
            f"店鋪 ID：{store_id}\n"
            f"員工人數：{store_data['user_cnt']} 人\n"
            f"管理人員：{store_data['manager_cnt']} 人\n"
            f"總人數：{total_people} 人\n"
            f"使用幣別：{store_data['currency']}"
        )
    else:
        available_stores = ", ".join(default_store.keys())
        return f"找不到店鋪 {store_id}。可用的店鋪有：{available_stores}"


@mcp.tool()
async def get_all_stores_info() -> str:
    """查詢所有店鋪的人數和幣別資訊。"""
    if not default_store:
        return "目前沒有任何店鋪資料。"

    store_info_list = []
    for store_id, store_data in default_store.items():
        total_people = store_data["user_cnt"] + store_data["manager_cnt"]
        store_info = (
            f"店鋪 ID：{store_id}\n"
            f"員工人數：{store_data['user_cnt']} 人\n"
            f"管理人員：{store_data['manager_cnt']} 人\n"
            f"總人數：{total_people} 人\n"
            f"使用幣別：{store_data['currency']}"
        )
        store_info_list.append(store_info)

    return "\n" + "=" * 50 + "\n".join(store_info_list)


@mcp.tool()
async def get_store_with_exchange_rate(store_id: str, days: int = 1) -> str:
    """查詢指定店鋪的人數、幣別資訊，並同時查詢該幣別的最新匯率。

    Args:
        store_id: 店鋪 ID (例如：STORE1, STORE2)。
        days: 要查詢的匯率天數 (預設為 1 天，即最新匯率)。
    """
    store_id = store_id.upper()
    if store_id not in default_store:
        available_stores = ", ".join(default_store.keys())
        return f"找不到店鋪 {store_id}。可用的店鋪有：{available_stores}"

    # 獲取店鋪信息
    store_data = default_store[store_id]
    total_people = store_data["user_cnt"] + store_data["manager_cnt"]

    store_info = (
        f"店鋪 ID：{store_id}\n"
        f"員工人數：{store_data['user_cnt']} 人\n"
        f"管理人員：{store_data['manager_cnt']} 人\n"
        f"總人數：{total_people} 人\n"
        f"使用幣別：{store_data['currency']}"
    )

    # 查詢匯率（如果是台幣則不需要查詢匯率）
    if store_data["currency"] == "台幣":
        exchange_info = "台幣為本地貨幣，無需匯率轉換。"
    else:
        # 根據幣別映射到匯率代碼
        currency_map = {
            "日幣": "JPY",
            "美金": "USD",
            "美元": "USD",
            "歐元": "EUR",
            "英鎊": "GBP",
            "港幣": "HKD",
            "澳幣": "AUD",
            "加幣": "CAD",
            "新幣": "SGD",
            "人民幣": "CNY",
        }

        currency_code = currency_map.get(store_data["currency"])
        if currency_code:
            # 查詢最新匯率
            exchange_info = await get_latest_exchange_rates(currency_code, days)
        else:
            exchange_info = f"無法查詢 {store_data['currency']} 的匯率信息（不支援的幣別）"

    return f"{store_info}\n\n{'='*50}\n匯率資訊：\n{exchange_info}"


def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    """建立一個 Starlette 應用程式，用於提供 MCP 伺服器的 SSE 介面。"""
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options(),
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ],
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server

    import argparse

    parser = argparse.ArgumentParser(description="運行基於 SSE 的 MCP 匯率伺服器")
    parser.add_argument("--host", default="127.0.0.1", help="綁定的 Host")
    parser.add_argument("--port", type=int, default=8181, help="監聽的 Port")
    args = parser.parse_args()

    # 綁定 SSE 請求處理到 MCP 伺服器
    starlette_app = create_starlette_app(mcp_server, debug=True)

    uvicorn.run(starlette_app, host=args.host, port=args.port)
