from fastapi import FastAPI, UploadFile, HTTPException
from pydantic import BaseModel

from rag_assistant.cli import get_client, load_system_prompt
from rag_assistant.ingestion.loader import load_file
from rag_assistant.ingestion.chunker import chunk_text
from rag_assistant.retrieval.embedder import embed_texts
from rag_assistant.pipeline import make_search_tool, answer_with_tools

import logging

app = FastAPI()

logger = logging.getLogger(__name__)

client = get_client()
system_prompt = load_system_prompt("config/system_prompt.txt")
document = load_file("data/test_report.txt")

chunks = chunk_text(document)
chunk_embeddings = embed_texts(client, chunks)

search_tool = make_search_tool(client, chunks, chunk_embeddings)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    result = answer_with_tools(client, request.question, search_tool, system_prompt)
    return {"answer": result}


@app.post("/upload")
async def upload_document(file: UploadFile):
    global chunks, chunk_embeddings, search_tool

    raw_content = await file.read()

    content = raw_content.decode("utf-8")

    try:
        new_chunks = chunk_text(content)
        new_chunk_embeddings = embed_texts(client, new_chunks)
        new_search_tool = make_search_tool(client, new_chunks, new_chunk_embeddings)

        chunks = new_chunks
        chunk_embeddings = new_chunk_embeddings
        search_tool = new_search_tool
    except Exception:
        logger.exception(
            "Critical error during document processing: chunking or embedding failed"
        )

        raise HTTPException(
            status_code=400,
            detail="Failed to process document: text chunking or embedding generation failed.",
        )

    return {"status": "document loaded", "chunks_count": len(chunks)}
