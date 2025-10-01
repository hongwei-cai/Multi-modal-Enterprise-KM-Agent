"""
FastAPI application for the Multi-modal Enterprise KM Agent.
"""
import json
import logging
import os
import time
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from starlette.middleware.base import BaseHTTPMiddleware

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

# Configure structured logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# Custom JSON formatter for structured logs
class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
        }
        if hasattr(record, "extra"):
            log_entry.update(record.extra)
        return json.dumps(log_entry)


# Apply formatter to logger
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:  # noqa: F841
            # Re-raise HTTPExceptions as-is
            raise
        except Exception as e:
            logger.error(f"Unhandled error: {e}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error": str(e)},
            )


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        logger.info(
            "Request started",
            extra={
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
                "client_ip": request.client.host if request.client else None,
            },
        )

        try:
            response = await call_next(request)
            process_time = time.time() - start_time

            # Log response
            logger.info(
                "Request completed",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "status_code": response.status_code,
                    "process_time": f"{process_time:.4f}s",
                },
            )

            # Add latency header to response
            response.headers["X-Process-Time"] = str(process_time)
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "method": request.method,
                    "url": str(request.url),
                    "error": str(e),
                    "process_time": f"{process_time:.4f}s",
                },
                exc_info=True,
            )
            raise


# Pydantic models for request/response validation
class UploadResponse(BaseModel):
    message: str = Field(..., description="Response message")
    document_id: Optional[str] = Field(None, description="Optional document ID")


class QuestionRequest(BaseModel):
    question: str = Field(
        ..., min_length=1, max_length=500, description="User question"
    )
    top_k: Optional[int] = Field(
        5, ge=1, le=20, description="Number of docs to retrieve"
    )
    temperature: Optional[float] = Field(
        0.7, ge=0.0, le=2.0, description="Generation temperature"
    )

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace")
        return v


class AnswerResponse(BaseModel):
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer")
    context_docs: Optional[List[str]] = Field(
        None, description="Retrieved context docs"
    )


# Dependency to get pipelines with configurable DB path
def get_indexer():
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")  # Default to local path
    return get_indexing_pipeline(db_path=db_path)


def get_rag():
    db_path = os.getenv("CHROMA_DB_PATH", "./chroma_db")  # Default to local path
    return get_rag_pipeline(db_path=db_path)


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
        logger.info("Document upload started", extra={"file_name": file.filename})

        # Save file temporarily
        import tempfile

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(await file.read())
            temp_path = temp_file.name

        # Index the document
        indexer.index_document(temp_path)

        logger.info("Document indexed successfully", extra={"file_name": file.filename})

        # Clean up
        os.unlink(temp_path)

        return UploadResponse(message="Document uploaded and indexed successfully")
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document")


@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest, rag=Depends(get_rag)):
    try:
        result = rag.answer_question(
            question=request.question,
            top_k=request.top_k,
            temperature=request.temperature,
        )
        return AnswerResponse(
            question=request.question,
            answer=result["answer"],
            context_docs=result["context_docs"],
        )
    except Exception as e:
        logger.error(f"QA failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate answer")


# Add middleware
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(LoggingMiddleware)


# Optional: Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the Multi-modal Enterprise KM Agent API"}
