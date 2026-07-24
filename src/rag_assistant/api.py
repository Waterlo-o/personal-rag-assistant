from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_assistant.cli import get_client, load_system_prompt
from rag_assistant.ingestion.loader import load_file
from rag_assistant.ingestion.chunker import chunk_text
from rag_assistant.retrieval.embedder import embed_texts
from rag_assistant.pipeline import (
    make_search_tool,
    answer_with_tools,
    make_neighbor_tool,
)
from rag_assistant.constants import CHROMA_PATH

import logging
import chromadb

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(name)s - %(message)s")

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # понижен, чтобы видеть AFC-строки
console_handler.setFormatter(formatter)

file_handler = logging.FileHandler("app.log", encoding="utf-8")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)
root_logger.addHandler(console_handler)
root_logger.addHandler(file_handler)

client = get_client()
system_prompt = load_system_prompt("config/system_prompt.txt")
document = load_file("data/test_report.txt")


chr_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chr_client.get_or_create_collection("current_doc")

if collection.count() == 0:
    chunks_list = chunk_text(document)
    chunk_embeddings = embed_texts(client, chunks_list)

    valid_ids = []
    valid_chunks = []
    valid_emb = []

    for i, emb in enumerate(chunk_embeddings):
        if emb is not None:
            valid_ids.append(str(i))
            valid_chunks.append(chunks_list[i])
            valid_emb.append(chunk_embeddings[i])

    collection.add(ids=valid_ids, documents=valid_chunks, embeddings=valid_emb)

current_search_tool = make_search_tool(client, collection)
current_neightboring_tool = make_neighbor_tool(collection)


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    if collection.count() == 0:
        logger.exception("Documenet base used by API is empty.")
        raise HTTPException(status_code=400, detail="Document base is empty.")

    result = answer_with_tools(
        client,
        request.question,
        current_search_tool,
        current_neightboring_tool,
        system_prompt,
    )
    return {"answer": result}


@app.post("/upload")
async def upload_document(file: UploadFile):
    raw_content = await file.read()
    content = raw_content.decode("utf-8")

    try:
        existing_ids = collection.get()["ids"]
        if existing_ids:
            collection.delete(ids=existing_ids)

        chunked_content = chunk_text(content)
        embed_chunks = embed_texts(client, chunked_content)

        valid_ids = []
        valid_chunks = []
        valid_emb = []

        for i, emb in enumerate(embed_chunks):
            if emb is not None:
                valid_ids.append(str(i))
                valid_chunks.append(chunked_content[i])
                valid_emb.append(embed_chunks[i])

        collection.add(ids=valid_ids, documents=valid_chunks, embeddings=valid_emb)

    except Exception:
        logger.exception(
            "Critical error during document processing: chunking or embedding failed"
        )
        raise HTTPException(
            status_code=400,
            detail="Failed to process document: text chunking or embedding generation failed.",
        )

    return {"status": "document loaded", "chunks_count": collection.count()}
