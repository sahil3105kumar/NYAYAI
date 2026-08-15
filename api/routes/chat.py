

import logging

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
        
        from api.services.legal_agent import chat_with_nyayai

        reply = chat_with_nyayai(req.message, thread_id=req.thread_id, job_id=req.job_id)
        return ChatResponse(reply=reply, thread_id=req.thread_id)
    except Exception as e:
        logger.exception("Chat agent error")
        raise HTTPException(status_code=500, detail=f"Agent error: {e}")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_pdf(req: IngestRequest):
    
    from services.storage import extracted_text_path

    text_path = extracted_text_path(req.job_id)
    if not text_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                f"No extracted text found for job_id: {req.job_id}. "
                "The document analysis job must finish successfully "
                "(GET /status/{job_id} == SUCCESS) before it can be ingested."
            ),
        )

    try:
        from api.services.pdf_to_graph import process_text_to_graph

        markdown_text = text_path.read_text(encoding="utf-8")
        await process_text_to_graph(markdown_text, job_id=req.job_id)
        return IngestResponse()
    except Exception as e:
        logger.exception("Knowledge graph ingestion error")
        raise HTTPException(status_code=500, detail=f"Ingestion error: {e}")
