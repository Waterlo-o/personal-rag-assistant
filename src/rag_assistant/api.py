from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_assistant.cli import get_client, load_system_prompt
from rag_assistant.ingestion.loader import load_file
from rag_assistant.ingestion.chunker import chunk_text
from rag_assistant.retrieval.embedder import embed_texts
from rag_assistant.pipeline import make_search_tool, answer_with_tools

import logging
import sqlite3
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5500", "http://127.0.0.1:5500"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)

client = get_client()
system_prompt = load_system_prompt("config/system_prompt.txt")
document = load_file("data/test_report.txt")

current_search_tool = None


def init_db():
    with sqlite3.connect("data.db") as connection:
        cursor = connection.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks(
                id INTEGER PRIMARY KEY,
                text TEXT,
                embedding TEXT
            )
        """)
        connection.commit()


def refresh_search_tool():
    global current_search_tool

    db_chunks = []
    db_embeddings = []

    with sqlite3.connect("data.db") as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT text, embedding FROM chunks")
        rows = cursor.fetchall()

        for row in rows:
            db_chunks.append(row[0])
            db_embeddings.append(json.loads(row[1]))

    if db_chunks:
        current_search_tool = make_search_tool(client, db_chunks, db_embeddings)
    else:
        current_search_tool = None


def seed_db_if_empty():
    with sqlite3.connect("data.db") as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM chunks")
        is_empty = cursor.fetchone()

        if is_empty is None:
            chunks_list = chunk_text(document)
            chunk_embeddings = embed_texts(client, chunks_list)
            for chunk, emb in zip(chunks_list, chunk_embeddings):
                cursor.execute(
                    "INSERT INTO chunks (text, embedding) VALUES (?, ?)",
                    (chunk, json.dumps(emb)),
                )
            connection.commit()


init_db()
seed_db_if_empty()
refresh_search_tool()


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: AskRequest):
    if current_search_tool is None:
        raise HTTPException(status_code=400, detail="База документов пуста.")

    result = answer_with_tools(
        client, request.question, current_search_tool, system_prompt
    )
    return {"answer": result}


@app.post("/upload")
async def upload_document(file: UploadFile):
    raw_content = await file.read()
    content = raw_content.decode("utf-8")

    try:
        with sqlite3.connect("data.db") as connection:
            cursor = connection.cursor()
            cursor.execute("DELETE FROM chunks")

            new_chunks = chunk_text(content)
            new_chunk_embeddings = embed_texts(client, new_chunks)

            for chunk, emb in zip(new_chunks, new_chunk_embeddings):
                cursor.execute(
                    "INSERT INTO chunks (text, embedding) VALUES (?, ?)",
                    (chunk, json.dumps(emb)),
                )
            connection.commit()

        refresh_search_tool()

    except Exception:
        logger.exception(
            "Critical error during document processing: chunking or embedding failed"
        )
        raise HTTPException(
            status_code=400,
            detail="Failed to process document: text chunking or embedding generation failed.",
        )

    return {"status": "document loaded", "chunks_count": len(new_chunks)}
