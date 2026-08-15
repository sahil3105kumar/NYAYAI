

import logging
import os

from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_neo4j import Neo4jGraph

load_dotenv()

logger = logging.getLogger(__name__)


_CASE_FILE_QUERY = """
MATCH (n)
WHERE n.job_id = $job_id
OPTIONAL MATCH (n)-[r]->(m)
WHERE m.job_id = $job_id
RETURN
    coalesce(n.name, n.id) AS source,
    [l IN labels(n) WHERE l <> '__Entity__'] AS source_labels,
    type(r) AS relationship,
    coalesce(m.name, m.id) AS target
LIMIT 300
"""


def _get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=os.getenv("NEO4J_URI"),
        username=os.getenv("NEO4J_USERNAME"),
        password=os.getenv("NEO4J_PASSWORD"),
        database=os.getenv("NEO4J_DATABASE"),
    )


def _format_facts(records: list[dict]) -> str:

    lines: list[str] = []
    seen: set[str] = set()

    for row in records:
        source = row.get("source")
        if not source:
            continue

        rel = row.get("relationship")
        target = row.get("target")

        if rel and target:
            line = f"({source}) -[{rel}]-> ({target})"
        else:
            labels = row.get("source_labels") or []
            entity_type = labels[0] if labels else None
            line = f"{source}" + (f" [{entity_type}]" if entity_type else "")

        if line not in seen:
            seen.add(line)
            lines.append(line)

    return "\n".join(lines)


def query_case_file(question: str, job_id: str) -> str:

    graph = _get_graph()

    try:
        records = graph.query(_CASE_FILE_QUERY, params={"job_id": job_id})
    except Exception:
        logger.exception("Graph RAG query failed (job_id=%s)", job_id)
        return (
            "I couldn't reach the knowledge graph to answer that. Make sure "
            "Neo4j is reachable and try again."
        )

    if not records:
        return (
            "I don't see this document in the knowledge graph yet. It needs "
            "to be ingested first (POST /api/v1/chat/ingest) before I can "
            "answer questions about it."
        )

    facts = _format_facts(records)

    llm = ChatMistralAI(model="mistral-small-latest", temperature=0)
    prompt = f"""You are a legal assistant answering a question about a single uploaded case file.
Use ONLY the facts extracted from this document's knowledge graph below - do not use
outside knowledge or facts about any other case. If the facts don't contain the
answer, say so plainly instead of guessing.

Knowledge graph facts:
{facts}

Question: {question}

Answer:"""

    response = llm.invoke(prompt)
    return response.content