import logging

from google.genai import types

from rag_assistant.retrieval.embedder import find_relevant_chunks

from rag_assistant.constants import DEFAULT_MODEL

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


def make_search_tool(client, chunks, chunk_embeddings):
    def search_documents(query: str) -> str:
        """Searches the company financial reports for information relevant to the query.

        Use this tool when the user asks a question that requires specific facts
        from the documents — such as revenue, expenses, staff numbers, or other
        figures tied to a specific company or year. Do not use this tool for
        general conversation or questions unrelated to the financial reports.

        Args:
            query: The user's question or the key terms to search for,
                ideally including the company name and/or year if mentioned.

        Returns:
            A string containing the most relevant excerpts found in the documents,
            separated by "---". Returns an empty string if nothing relevant is found.
        """

        found_chunks = find_relevant_chunks(client, query, chunk_embeddings, chunks)

        if not found_chunks:
            final_text = ""
            return final_text

        final_text = "\n\n---\n\n".join(found_chunks)

        return final_text

    return search_documents


def answer_with_tools(client, query, search_tool, system_prompt, model=DEFAULT_MODEL):
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[search_tool],
        ),
    )

    return response.text


# recreated function manually for practice


def answer_with_tools_manually(
    client, query, search_tool, system_prompt, model=DEFAULT_MODEL
):
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[search_tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        ),
    )

    if not response.function_calls:
        return response.text

    function_call = response.function_calls[0]

    response_args = function_call.args

    found_text = search_tool(query=response_args["query"])

    user_query = types.Content(
        role="user",
        parts=[types.Part.from_text(text=query)],
    )

    function_response = {"result": found_text}

    function_response_args = types.Part.from_function_response(
        name=function_call.name, response=function_response
    )

    function_response_content = types.Content(
        role="tool", parts=[function_response_args]
    )

    function_call_content = response.candidates[0].content

    final_answer = client.models.generate_content(
        model=model,
        contents=[user_query, function_call_content, function_response_content],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt, tools=[search_tool]
        ),
    )

    return final_answer.text
