import logging

from google.genai import types

from rag_assistant.retrieval.embedder import embed_texts

from rag_assistant.constants import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def make_search_tool(client, collection):
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

        vec_query = embed_texts(client, [query])[0]

        raw_result = collection.query(query_embeddings=[vec_query], n_results=3)

        docs = raw_result.get("documents")

        if not docs or not docs[0]:
            return ""

        result = docs[0]

        final_text = "\n\n---\n\n".join(result)

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
