#!/usr/bin/env python3
"""
LightRAG API Server
"""
from lightrag.api.lightrag_server import get_application
from lightrag.api.config import global_args

app = get_application(global_args)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host=global_args.host,
        port=8000,  # Explicitly set to 8000 for backend
        reload=True,
        log_level="info"
    )
