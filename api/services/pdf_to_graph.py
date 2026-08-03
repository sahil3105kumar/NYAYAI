import os
from dotenv import load_dotenv

# PyMuPDF4LLM for fast local PDF-to-Markdown conversion
import pymupdf4llm

# LangChain imports
from langchain_text_splitters import MarkdownTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document

load_dotenv()

def process_pdf_to_graph(pdf_path: str):
    """
    1. Converts PDF to Markdown using PyMuPDF4LLM.
    2. Chunks the Markdown text preserving layout headers.
    3. Uses LLMGraphTransformer to extract legal entities/relationships.
    4. Ingests structured graph documents into Neo4j.
    """
    print(f"\n📄 [1/5] Converting PDF to Markdown via PyMuPDF4LLM: {pdf_path}")
    
    # Convert the PDF directly into a Markdown formatted string
    md_text = pymupdf4llm.to_markdown(pdf_path)
    
    print("✂️  [2/5] Splitting Markdown into structural chunks...")
    # MarkdownTextSplitter preserves headers (#, ##), lists, and formatting blocks
    text_splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
    
    # Convert the raw markdown string into LangChain Document objects
    documents = text_splitter.create_documents([md_text])
    print(f"    └─ Successfully created {len(documents)} markdown chunk(s).")

    print("🔌 [3/5] Connecting to Neo4j Database...")
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE")
    )

    # Use mistral-small with retries to handle API rate limits smoothly
    print("🧠 [4/5] Initializing Legal Graph Transformer...")
    llm = ChatMistralAI(
        model="mistral-small-latest",
        temperature=0,
        max_retries=5
    )

    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=[
            "Victim", "Accused", "Witness", "Evidence", 
            "Location", "Statute", "Date", "Police_Officer"
        ],
        allowed_relationships=[
            "COMMITTED_AT", "WITNESSED_BY", "CORROBORATES", 
            "CONTRADICTS", "RELIABLE_EVIDENCE", "RECOVERED_FROM", "CHARGED_UNDER"
        ]
    )

    print("⚡ [5/5] Extracting Legal Entities from Markdown & writing to Neo4j...")
    # This step sends the markdown chunks to Mistral to extract nodes/relationships
    graph_documents = transformer.convert_to_graph_documents(documents)
    
    # Write the extracted graph data directly to your Aura DB
    graph.add_graph_documents(graph_documents, baseEntityLabel=True)
    print("✅ Knowledge Graph successfully created in Neo4j using PyMuPDF4LLM Markdown!")

if __name__ == "__main__":
    # Test script standalone if needed
    import sys
    if len(sys.argv) > 1:
        process_pdf_to_graph(sys.argv[1])