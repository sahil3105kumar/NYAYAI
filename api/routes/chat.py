"""
POST /api/v1/chat        — general legal chatbot Q&A via the master LangGraph agent.
POST /api/v1/chat/ingest — triggers PDF-to-graph ingestion into Neo4j.

These endpoints are stateless from FastAPI's perspective; conversation memory
lives inside the LangGraph MemorySaver keyed by thread_id.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from api.schemas.chat import ChatRequest, ChatResponse, IngestRequest, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Send a message to the NyayAI legal agent.
    The agent routes internally to Graph RAG, InLegalBERT, or drafting tools.
    """
    try:
        # Lazy import to avoid loading heavy LangChain/Neo4j deps at module level
        from api.services.legal_agent import chat_with_nyayai

        reply = chat_with_nyayai(req.message, thread_id=req.thread_id)
        return ChatResponse(reply=reply, thread_id=req.thread_id)
    except Exception as e:
        logger.exception("Chat agent error")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(req: IngestRequest):
    """
    Ingest a PDF into the Neo4j knowledge graph.
    Converts PDF → Markdown → LLM entity extraction → Neo4j nodes/relationships.
    """
    pdf_path = Path(req.pdf_path)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail=f"PDF not found: {req.pdf_path}")

    try:
        from api.services.pdf_to_graph import process_pdf_to_graph

        process_pdf_to_graph(str(pdf_path))
        return IngestResponse()
    except Exception as e:
        logger.exception("PDF ingestion error")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")
