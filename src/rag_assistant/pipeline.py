import logging

from google.genai import types

from rag_assistant.retrieval.embedder import find_relevant_chunks

from rag_assistant.cli import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def answer_question(
    client, query, chunks, chunk_embedding, system_prompt, model=DEFAULT_MODEL, top_n=3
):
    relevant_chunks = find_relevant_chunks(
        client, query, chunk_embedding, chunks, top_n=top_n
    )

    if not relevant_chunks:
        logger.warning("Relevant chunks not found!")
        return "Couldn't find relevant chunks"

    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = f"Context: {context}\n\nQuestion: {query}\n"

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )

    logger.info(f"Request has been done successfully for question: {query}")
    return response.text
