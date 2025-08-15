# MCP 教學文件

本文件將詳細說明如何在 MCP (Multi-Component Platform) 中實作 tool、prompt、resource，並比較三者的差異。

---

## 1. Tool 實作教學

### 什麼是 Tool？
Tool 是一種可被 AI 直接呼叫的函式，通常用於查詢、計算、資料處理等明確任務。Tool 需以 `@mcp.tool()` 裝飾器標註，並定義明確的輸入參數與回傳格式。

### 強化實作範例
```python
from mcp import tool

# 假設有一個資料庫模組 db，可查詢店鋪資訊
import db

@mcp.tool()
async def get_store_info(store_id: str) -> dict:
    """查詢指定店鋪的人數與幣別。"""
    store = db.get_store(store_id)
    return {
        "store_id": store_id,
        "people_count": store.people_count,
        "currency": store.currency
    }

@mcp.tool()
async def get_exchange_rate(currency: str) -> float:
    """查詢指定幣別的最新匯率。"""
    rate = db.get_today_exchange_rate(currency)
    return rate
```

---

## 2. Prompt 實作教學

### 什麼是 Prompt？
Prompt 是一種以自然語言描述的任務指令，適合複雜邏輯或需 AI 理解語意的情境。以 `@mcp.prompt()` 裝飾器標註，內容通常為文字描述，AI 會根據 prompt 內容執行任務。

### 強化實作範例
```python
from mcp import prompt

@mcp.prompt()
def 查詢店鋪人數與匯率(store_id: str):
    """
    請查詢指定店鋪的員工人數、管理人員數、幣別，並同時查詢該幣別的最新匯率。
    例如：
    輸入 store_id="STORE2"，回傳：
    {
        "store_id": "STORE2",
        "people_count": 12,
        "manager_count": 2,
        "currency": "USD",
        "exchange_rate": 31.5
    }
    """
    # AI 會自動串接相關 tool 完成查詢
```

---

## 3. Resource 實作教學

### 什麼是 Resource？
Resource 通常用於定義資料來源、外部 API、資料庫等。以 `@mcp.resource()` 裝飾器標註，讓 AI 能存取特定資源。

### 強化實作範例
```python
from mcp import resource
import sqlalchemy

@mcp.resource()
def store_db():
    """店鋪資料庫資源"""
    engine = sqlalchemy.create_engine("sqlite:///store.db")
    return engine.connect()

@mcp.resource()
def exchange_rate_api():
    """匯率 API 資源"""
    # 連接外部匯率 API
    return "https://api.exchangerate.host/latest"
```

---

## 4. 差異比較
| 類型      | 裝飾器         | 適用情境         | 主要用途         |
|-----------|----------------|------------------|------------------|
| Tool      | @mcp.tool()    | 明確 API 呼叫    | 功能/查詢/計算   |
| Prompt    | @mcp.prompt()  | 複雜語意/邏輯    | 自然語言任務     |
| Resource  | @mcp.resource()| 資料/外部資源    | 資料存取         |

---

## 5. MCP 串接範例

### Tool 範例
```python
# 查詢 STORE2 的人數與幣別
result = get_store_info(store_id="STORE2")
print(result)  # {'store_id': 'STORE2', 'people_count': 12, 'currency': 'USD'}

# 查詢 USD 匯率
rate = get_exchange_rate(currency="USD")
print(rate)  # 31.5
```

### Prompt 範例
```python
# 查詢 STORE2 的詳細信息和今天的匯率
result = 查詢店鋪人數與匯率(store_id="STORE2")
print(result)
# {'store_id': 'STORE2', 'people_count': 12, 'manager_count': 2, 'currency': 'USD', 'exchange_rate': 31.5}
```

### Resource 範例
```python
# 取得店鋪資料庫連線
conn = store_db()
# 取得匯率 API 資源
api_url = exchange_rate_api()
```

---

## 6. 常見問題
- 若不想用 @mcp.tool()，可改用 @mcp.prompt()，以自然語言描述任務。
- 若遇到權限問題（如通訊端存取被拒），請確認執行權限或更換埠號。
- 若需設定 SSE，請在 msc 設定 json 檔中加入 sse 相關參數，例如：
```json
{
  "sse": {
    "host": "0.0.0.0",
    "port": 8082
  }
}
```

---

如需更多範例或進階教學，請參考專案內的 README 或官方文件。
