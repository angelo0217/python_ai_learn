# SQL MCP Server (SSE)
這個mcp我是用claude + pycharm mcp server，輸入指令請他仿照 exchange 來寫的mcp，中間都沒改它的code
這是一個基於MCP協議的SQL操作服務器，使用SSE（Server-Sent Events）作為通信方式。它為用戶提供了一個簡單的介面來執行SQL查詢和管理資料庫。

## 功能特點

1. 執行各種SQL查詢（SELECT, INSERT, UPDATE, DELETE等）
2. 列出資料庫中的所有資料表
3. 查看資料表結構
4. 以JSON格式簡化資料插入和更新操作
5. 美觀的表格格式顯示查詢結果

## 安裝依賴

在運行前，請確保安裝了所需的依賴：

```bash
pip install fastapi starlette uvicorn httpx aiosqlite mcp
```

## 快速開始

1. 將`sql_mcp_sse.py`複製到您的專案目錄
2. 直接運行此文件即可啟動服務器：

```bash
python sql_mcp_sse.py
```

3. 默認情況下，服務器將在`0.0.0.0:8082`上運行，資料庫文件為當前目錄下的`database.sqlite`

## 命令行選項

支持以下命令行選項：

- `--host`: 指定主機地址（默認：0.0.0.0）
- `--port`: 指定端口號（默認：8082）
- `--db`: 指定資料庫路徑（默認：database.sqlite）

示例：

```bash
python sql_mcp_sse.py --host 127.0.0.1 --port 8082 --db my_database.db
```

## 可用的MCP工具

該服務器提供以下MCP工具：

### 1. execute_sql(query)

執行任意SQL查詢。

參數：
- `query`: SQL查詢語句

示例：
```
execute_sql("SELECT * FROM users WHERE age > 25")
```

### 2. list_tables()

列出資料庫中的所有資料表。

示例：
```
list_tables()
```

### 3. get_table_structure(table_name)

獲取指定資料表的結構。

參數：
- `table_name`: 資料表名稱

示例：
```
get_table_structure("users")
```

### 4. insert_data(table_name, data_json)

插入數據到指定資料表。

參數：
- `table_name`: 要插入資料的資料表名稱
- `data_json`: 包含要插入資料的JSON字串

示例：
```
insert_data("users", '{"name": "陳六", "email": "chen@example.com", "age": 28}')
```

### 5. update_data(table_name, data_json, condition)

更新資料表中的資料。

參數：
- `table_name`: 資料表名稱
- `data_json`: 包含要更新資料的JSON字串
- `condition`: 更新條件（WHERE子句）

示例：
```
update_data("users", '{"age": 29}', "name = '陳六'")
```

### 6. delete_data(table_name, condition)

從資料表中刪除資料。

參數：
- `table_name`: 資料表名稱
- `condition`: 刪除條件（WHERE子句），留空將刪除所有資料

示例：
```
delete_data("users", "id = 3")
```

## 初始資料庫

啟動服務器時會自動初始化資料庫。默認情況下，它會創建一個包含以下結構的`users`資料表：

```sql
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    email TEXT UNIQUE,
    age INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
```

並插入三條測試資料：
- 張三 (zhang@example.com, 30歲)
- 李四 (li@example.com, 25歲)
- 王五 (wang@example.com, 35歲)

## 連接到服務器

您可以通過以下URL連接到SSE伺服器：

```
http://<host>:<port>/sse
```

例如：`http://localhost:8082/sse`

## 示例用法

1. 查詢所有用戶：
```
execute_sql("SELECT * FROM users")
```

2. 添加新用戶：
```
insert_data("users", '{"name": "趙七", "email": "zhao@example.com", "age": 40}')
```

3. 查詢年齡大於30的用戶：
```
execute_sql("SELECT name, age FROM users WHERE age > 30")
```

4. 更新用戶資料：
```
update_data("users", '{"age": 26}', "name = '李四'")
```

5. 刪除用戶：
```
delete_data("users", "name = '張三'")
```

## 注意事項

- 此範例使用 SQLite 作為資料庫引擎，適合小型應用或開發測試
- 對於生產環境，建議配置適當的安全措施和錯誤處理機制
- 服務器使用非阻塞的異步操作處理請求，適合高併發場景
