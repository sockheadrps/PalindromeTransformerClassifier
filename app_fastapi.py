#!/usr/bin/env python3
"""
Standalone FastAPI Server for Specialized Multi-Word Palindrome Detection

This is a direct replacement for the Flask app_specialized.py
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uvicorn
from palindrome_router import shutdown_requested, unload_model, model
from palindrome_router import unload_scheduled, unload_task, palindrome_router
import palindrome_router
import signal


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("🚀 FastAPI application started successfully")
    print("💡 Models will be loaded on first request and unloaded after 5 minutes of inactivity")

    yield

    # Shutdown
    print("🛑 Shutting down FastAPI application...")

    # Set shutdown flag to cancel background tasks

    palindrome_router.shutdown_requested = True

    # Cancel any scheduled unloads
    unload_scheduled = False
    if unload_task and not unload_task.done():
        try:
            unload_task.cancel()
            print("⏰ Cancelled scheduled model unload")
        except Exception as e:
            print(f"⏰ Error cancelling unload task: {e}")
    else:
        print("⏰ No scheduled unloads to cancel")

    # Unload all loaded models
    if model is not None:
        print("💾 Unloading model during shutdown...")
        unload_model()
        print("✅ Model unloaded successfully")

    print("✅ FastAPI application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Specialized Multi-Word Palindrome Detection API",
    description="A FastAPI server for detecting palindromes using specialized neural network models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for the monitor dashboard
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the palindrome router at root level (same as Flask app)
app.include_router(palindrome_router.palindrome_router, tags=["palindrome"])

# Add a redirect from / to /docs for better UX


@app.get("/")
async def root():
    """Redirect to API documentation."""
    return {
        "message": "Specialized Multi-Word Palindrome Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "monitor": "/monitor",
        "health": "/health"
    }

# Serve the monitor dashboard


@app.get("/monitor")
async def monitor():
    """Serve the monitoring dashboard."""
    monitor_file = "static/index.html"
    if os.path.exists(monitor_file):
        return FileResponse(monitor_file)
    else:
        raise HTTPException(
            status_code=404, detail="Monitor dashboard not found")

if __name__ == "__main__":


    def signal_handler(sig, frame):
        print("\n🛑 Received shutdown signal, cleaning up...")
        # Set shutdown flag
        from palindrome_router import shutdown_requested
        import palindrome_router
        palindrome_router.shutdown_requested = True
        # Don't call sys.exit() - let uvicorn handle the shutdown gracefully

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("🚀 Starting FastAPI Reflection-Integrated Palindrome Detection API...")
    print("💡 Models will be loaded on first request and unloaded after 5 minutes of inactivity")
    print("\n📋 Available models:")

    # Start the FastAPI app
    port = 8222
    host = 'localhost'

    print(f"\n🌐 API will be available at: http://localhost:{port}")
    print("📖 API documentation: http://localhost:{port}/docs")
    print("📖 ReDoc documentation: http://localhost:{port}/redoc")
    print("🔍 Monitor dashboard: http://localhost:{port}/monitor")
    print("❤️  Health check: http://localhost:{port}/health")
    print("🔍 Model status: http://localhost:{port}/model/status")
    print("\n💡 Press Ctrl+C to stop the server gracefully")

    try:
        uvicorn.run(app, host=host, port=port)
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
    finally:
        print("✅ Server shutdown complete")
