"""
FastAPI application for the Multi-modal Enterprise KM Agent.
"""
import logging
from typing import Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.rag.indexing_pipeline import get_indexing_pipeline
from src.rag.rag_pipeline import get_rag_pipeline

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

logger = logging.getLogger(__name__)


# Pydantic models for request/response validation
class UploadResponse(BaseModel):
    message: str
    document_id: Optional[str] = None


class QuestionRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    temperature: Optional[float] = 0.7


class AnswerResponse(BaseModel):
    question: str
    answer: str
    context_docs: Optional[list] = None


# Dependency to get pipelines (can be configured via env)
def get_indexer():
    return get_indexing_pipeline()


def get_rag():
    return get_rag_pipeline()


@app.get("/health", summary="Health Check", description="Check if the API is running")
async def health_check():
    """
    Health check endpoint.

    Returns:
        dict: Status of the API.
    """
    return {"status": "healthy", "message": "API is running"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...), indexer=Depends(get_indexer)):
    """Upload and index a PDF document."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    try:
        # Save file temporarily
        import os
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Index the document
        indexer.index_document(temp_path)

        # Clean up
        os.unlink(temp_path)

        return UploadResponse(message="Document uploaded and indexed successfully")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, rag=Depends(get_rag)):
    """Answer a question using RAG."""
    try:
        answer = rag.answer_question(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
        )
        return AnswerResponse(question=request.question, answer=answer)
    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")


# Optional: Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-modal Enterprise KM Agent API"}
