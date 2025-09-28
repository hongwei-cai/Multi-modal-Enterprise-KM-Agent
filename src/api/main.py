"""
FastAPI application for the Multi-modal Enterprise KM Agent.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Multi-modal Enterprise KM Agent",
    description="RAG-based Q&A API for document processing",
    version="1.0.0",
)

# Add CORS middleware for web client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", summary="Health Check", description="Check if the API is running")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Status of the API.
    """
    return {"status": "healthy", "message": "API is running"}


# Placeholder for future endpoints (e.g., document upload, QA)
@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-modal Enterprise KM Agent API"}
