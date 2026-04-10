# 參考網址
https://github.com/sparfenyuk/mcp-proxy
https://github.com/sidharthrajaram/mcp-sse/blob/main/README.md
# claude安裝後設定
```
{
   "mcpServers":{
      "yourMcp":{
         "command":"mcp-proxy",
         "args":[
            "http://localhost:8080/sse"
         ]
      },
      "jetbrains":{
         "command":"npx",
         "args":[
            "-y",
            "@jetbrains/mcp-proxy"
         ]
      }
   }
}
```