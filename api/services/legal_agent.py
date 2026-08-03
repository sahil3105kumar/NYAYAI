"""
Master legal agent — routes user queries to the right tool:
  1. Graph RAG (query uploaded case files in Neo4j)
  2. InLegalBERT analysis (statute identification, rhetorical roles, judgment prediction)
  3. Legal document drafting

Uses LangGraph's create_react_agent for tool-calling with memory.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# Import the Graph RAG query tool from graph_rag.py
from .graph_rag import query_case_file

load_dotenv()

logger = logging.getLogger(__name__)

# Initialize the Generative Brain
llm = ChatMistralAI(model="mistral-small-latest", temperature=0.3)

@tool
def draft_legal_document(case_facts: str, document_type: str) -> str:
    """
    Use this tool ONLY when the user asks to draft, write, or generate a legal document 
    (like an FIR, Bail Application, or Case Brief).
    """
    logger.info(f"[Agent] 📝 Activating Drafting Tool for: {document_type}...")
    
    drafting_prompt = f"""
    You are an expert Indian Supreme Court lawyer. 
    Draft a highly professional {document_type} based on the following facts: {case_facts}.
    
    You MUST structure the document strictly using these Indian Legal Rhetorical Roles:
    1. PREAMBLE: The court, jurisdiction, and parties involved.
    2. FACTS: The objective events of the case.
    3. ISSUE: The core legal questions to be decided.
    4. ARGUMENT BY PETITIONER: The legal arguments (cite relevant IPC/CrPC sections).
    5. PRAYER/RELIEF: What the petitioner is asking the court to do.
    
    Maintain formal judicial language. Do not invent facts, but apply the correct Indian laws.
    """
    
    response = llm.invoke(drafting_prompt)
    return response.content

@tool
def analyze_with_inlegalbert(text: str) -> str:
    """
    Use this tool when the user asks to predict the outcome of a case, find rhetorical roles,
    or identify which IPC statutes apply to a given scenario.
    """
    logger.info("[Agent] 🧠 Sending to InLegalBERT Backend...")
    import requests
    try:
        response = requests.post("http://localhost:8000/analyze/full", json={"text": text}, timeout=15)
        return str(response.json())
    except Exception as e:
        return f"Backend analysis failed: {e}. Please ensure the FastAPI server is running."

@tool
def query_uploaded_case_file(question: str) -> str:
    """
    Use this tool ONLY when the user asks a specific question about an uploaded 
    case file, FIR, or legal document stored in the Neo4j Knowledge Graph.
    """
    logger.info(f"[Agent] 🔍 Querying Neo4j Knowledge Graph: {question}...")
    return query_case_file(question)

# Combine all 3 tools
tools = [draft_legal_document, analyze_with_inlegalbert, query_uploaded_case_file]

# Initialize short-term conversation checkpointer
memory = MemorySaver()

# Create the master agent executor using LangGraph's react agent
agent_executor = create_react_agent(
    model=llm, 
    tools=tools, 
    checkpointer=memory
)

def chat_with_nyayai(user_input: str, thread_id: str = "default_session") -> str:
    """Invokes the master agent and returns the response string."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [("user", user_input)]}
    
    response = agent_executor.invoke(inputs, config)
    return response["messages"][-1].content