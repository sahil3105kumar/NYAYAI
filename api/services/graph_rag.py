import os
from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Reconnect to Graph using the native wrapper
graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE")
    
)

cypher_llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
qa_llm = ChatMistralAI(model="mistral-small-latest", temperature=0.2)

# Custom Cypher Template that strictly enforces our Legal Schema
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer. Given a question, convert it into a Cypher query based on the graph schema below.

SCHEMA RULES:
1. Node Labels MUST only be chosen from:
   - Victim (Complainants/Victims like 'Krishna Punasya')
   - Accused (Suspects/Accused persons)
   - Witness (Witnesses in the case)
   - Evidence (Stolen property, physical evidence like 'MacBook', 'CCTV Footage')
   - Location (Places like 'IIITDMJ Campus', 'Khamaria Police Station')
   - Statute (Laws like 'IPC 379', 'BNS 303(2)')
   - Date (Dates mentioned)

2. Node Properties:
   - ALL nodes primarily use 'id' or 'name' for their titles, and 'description' for extra details (e.g., serial numbers, colors, or details are stored inside 'id' or 'description').
   - DO NOT query properties like '.color', '.serial_number', or '.address'. Use 'id', 'name', or 'description' instead!

3. Relationships MUST only be chosen from:
   - COMMITTED_AT, WITNESSED_BY, CORROBORATES, CONTRADICTS, RELIED_UPON, RECOVERED_FROM, CHARGED_UNDER

EXAMPLES:
- To find complainant/victim: MATCH (v:Victim) RETURN v.id AS complainant
- To find stolen item: MATCH (e:Evidence) RETURN e.id AS stolen_item, e.description AS details
- To find location: MATCH (l:Location) RETURN l.id AS location

Schema:
{schema}

Question: {question}
Cypher Query:"""

cypher_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema", "question"]
)

# Build the QA Chain
graph_qa_chain = GraphCypherQAChain.from_llm(
    cypher_llm=cypher_llm,
    qa_llm=qa_llm,
    graph=graph,
    verbose=True,
    cypher_prompt=cypher_prompt,
    allow_dangerous_requests=True
)

def query_case_file(question: str) -> str:
    response = graph_qa_chain.invoke({"query": question})
    return response["result"]

if __name__ == "__main__":
    # Test the RAG directly
    # ans = query_case_file("Who was the accused charged under IPC 302, and what did the main witness say?")
    # print(ans)
    print("Graph RAG tool initialized. Ready to query.")