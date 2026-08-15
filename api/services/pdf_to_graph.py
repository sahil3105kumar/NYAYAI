import asyncio
import logging
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_text_splitters import MarkdownTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_neo4j import Neo4jGraph
from langchain_core.documents import Document
from langchain_community.graphs.graph_document import GraphDocument

load_dotenv()

logger = logging.getLogger(__name__)

# The neo4j Python driver logs a log record for every notification the
# DBMS sends back - including the "apoc.create.addLabels is deprecated"
# notice langchain_neo4j's own generated Cypher triggers on every write
# batch. It's informational only (not our Cypher, not an error), so it's
# silenced here rather than spamming the API/worker log once per chunk.
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)


def _scope_graph_documents_to_job(
    graph_documents: list[GraphDocument], job_id: str
) -> list[GraphDocument]:
    """
    Rewrites every extracted node's id to be document-scoped before writing
    to Neo4j - "Police Station"  ->  "Police Station::<job_id>".

    Why this has to change the *id*, not just add a job_id property:
    Neo4jGraph.add_graph_documents(..., baseEntityLabel=True) writes nodes
    via `MERGE (n:__Entity__ {id: row.id}) SET n += row.properties`. MERGE
    matches on `id` alone. Two different uploaded documents that both
    mention, say, a "Police Station" or a common witness name would
    otherwise MERGE onto the exact same node - and because it's SET (not
    SET ON CREATE), the second document's write would just overwrite the
    first document's job_id property on that shared node. Every
    relationship from BOTH documents ends up attached to one physical
    node, which is precisely the cross-document leakage/hallucination
    this exists to fix - a job_id property alone does not prevent it,
    only a job_id-scoped merge key does.

    The original extracted name is preserved under `name` so Cypher
    queries can still show clean, human-readable text - only the
    underlying merge key changes. job_id is also stamped onto every
    relationship, so a Cypher query can filter on either end without an
    extra node lookup.
    """
    for gdoc in graph_documents:
        id_map: dict[tuple, str] = {}

        for node in gdoc.nodes:
            original_id = str(node.id)
            scoped_id = f"{original_id}::{job_id}"
            id_map[(original_id, node.type)] = scoped_id

            node.properties = dict(node.properties or {})
            node.properties.setdefault("name", original_id)
            node.properties["job_id"] = job_id
            node.id = scoped_id

        for rel in gdoc.relationships:
            src_key = (str(rel.source.id), rel.source.type)
            tgt_key = (str(rel.target.id), rel.target.type)
            if src_key in id_map:
                rel.source.id = id_map[src_key]
            if tgt_key in id_map:
                rel.target.id = id_map[tgt_key]

            rel.properties = dict(rel.properties or {})
            rel.properties["job_id"] = job_id

    return graph_documents


async def process_text_to_graph(markdown_text: str, job_id: str):
    
    print("  [1/4] Splitting Markdown into structural chunks...")
    
    text_splitter = MarkdownTextSplitter(chunk_size=2000, chunk_overlap=150)

    # Convert the raw markdown string into LangChain Document objects
    documents = text_splitter.create_documents([markdown_text])
    print(f"    └─ Successfully created {len(documents)} markdown chunk(s).")

    print(" [2/4] Connecting to Neo4j Database...")
    graph = Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE")
    )

    
    print("[3/4] Initializing Legal Graph Transformer...")
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

    print("[4/4] Extracting Legal Entities from Markdown & writing to Neo4j...")
    
    graph_documents = await transformer.aconvert_to_graph_documents(documents)

    
    graph_documents = _scope_graph_documents_to_job(graph_documents, job_id)

    
    await asyncio.to_thread(graph.add_graph_documents, graph_documents, baseEntityLabel=True)
    print(f"Knowledge Graph successfully created in Neo4j (job_id={job_id})")

if __name__ == "__main__":
    
    import sys
    if len(sys.argv) > 2:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            asyncio.run(process_text_to_graph(f.read(), job_id=sys.argv[2]))