"""
Pydantic request/response schemas for the /api/v1/chat/* endpoints.
"""

from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """User message sent to the legal chatbot agent."""
    message: str = Field(..., min_length=1, description="The user's question or instruction.")
    thread_id: Optional[str] = Field(
        default="default_session",
        description="Session thread ID for conversation memory."
    )


class ChatResponse(BaseModel):
    """Agent response from the legal chatbot."""
    reply: str = Field(..., description="The agent's response text.")
    thread_id: str


class IngestRequest(BaseModel):
    """Request to ingest a PDF into the Neo4j knowledge graph."""
    pdf_path: str = Field(..., description="Server-side path to the uploaded PDF file.")


class IngestResponse(BaseModel):
    """Confirmation of successful PDF ingestion."""
    status: str = "success"
    message: str = "PDF successfully ingested into knowledge graph."