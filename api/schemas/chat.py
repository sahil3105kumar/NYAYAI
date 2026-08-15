
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    
    message: str = Field(..., min_length=1, description="The user's question or instruction.")
    thread_id: Optional[str] = Field(
        default="default_session",
        description="Session thread ID for conversation memory."
    )
    job_id: Optional[str] = Field(
        default=None,
        description=(
            "job_id of the uploaded document this conversation is about "
            "(from /upload). Scopes Graph RAG lookups so this chat only "
            "sees this document's knowledge graph, not every document "
            "ever ingested into the shared Neo4j database. Omit only for "
            "general questions that don't need a specific case file."
        ),
    )


class ChatResponse(BaseModel):
    
    reply: str = Field(..., description="The agent's response text.")
    thread_id: str


class IngestRequest(BaseModel):
    
    job_id: str = Field(..., description="Job ID from the /upload endpoint — the PDF is looked up via services.storage.")


class IngestResponse(BaseModel):
    
    status: str = "success"
    message: str = "Document successfully ingested into knowledge graph."