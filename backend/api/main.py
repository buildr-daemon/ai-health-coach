"""
FastAPI application factory and configuration.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.v1 import router as v1_router
from db.database import init_database

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.
    Runs initialization on startup and cleanup on shutdown.
    """
    # Startup
    logger.info("Starting Health Agent API...")
    init_database()
    logger.info("Health Agent API ready!")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Health Agent API...")

# Create FastAPI app
app = FastAPI(
    title="Health Agent API",
    description="""
    AI-powered Health Coach API with:
    - Conversational health guidance
    - Long-term memory for personalized responses
    - RAG-based medical protocol retrieval
    - WhatsApp-like chat experience
    """,
    version="1.0.0",
    lifespan=lifespan
)


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(v1_router, prefix="/api")


@app.get("/", tags=["Root"])
def root():
    """Root endpoint with API info."""
    return {
        "name": "Health Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/chat/health"
    }
