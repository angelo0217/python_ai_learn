import logging
import uvicorn
from typing import Optional
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from mcp.server.sse import SseServerTransport
from mcp.server.fastmcp import FastMCP

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def create_sse_app(mcp: FastMCP, host: str = "0.0.0.0", port: int = 8000):
    """
    Helper to wrap a FastMCP instance into a Starlette SSE application.
    """
    sse = SseServerTransport("/messages")

    async def handle_sse(request):
        async with sse.connect_sse(request.scope, request.receive, request.send) as stream:
            await mcp.server.run(
                read_stream=stream.read_stream,
                write_stream=stream.write_stream,
                initialization_options=mcp.server.initialization_options,
            )

    async def handle_messages(request):
        await sse.handle_post_message(request.scope, request.receive, request.send)

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages", app=handle_messages),
        ],
        middleware=[
            Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
        ],
    )
    
    return app, host, port

def run_sse_server(app, host, port):
    logging.info(f"Starting MCP SSE server at http://{host}:{port}/sse")
    uvicorn.run(app, host=host, port=port)
