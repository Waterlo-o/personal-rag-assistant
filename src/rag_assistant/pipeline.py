import logging
import requests

from google.genai import types

from rag_assistant.retrieval.embedder import embed_texts

from rag_assistant.constants import DEFAULT_MODEL, EXCHANGER_URL

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
            A string containing the most relevant excerpts found in the documents.
            Each excerpt is prefixed with its identifier (e.g., "--- chunk 5 ---")
            and separated by blank lines. Returns an empty string if nothing relevant is found.
        """

        vec_query = embed_texts(client, [query])[0]

        raw_result = collection.query(query_embeddings=[vec_query], n_results=3)

        docs = raw_result.get("documents")
        ids = raw_result.get("ids")

        if not docs or not docs[0]:
            return ""

        result = docs[0]
        result_ids = ids[0]

        text_list = []

        for id, text in zip(result_ids, result):
            text_list.append(f"--- chunk {id} --- \n\n{text}")

        final_text = "\n".join(text_list)

        return final_text

    return search_documents


def transfer_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts an amount of money from one currency to another using the current exchange rate.
    Use this tool whenever currency conversion is needed, or if the retrieved documents
    specify an amount in one currency but the user asks for it in another.

    Args:
        amount: The amount of money to convert.
        from_currency: The base currency. MUST be a 3-letter ISO currency code (e.g., 'USD', 'EUR', 'GBP').
        to_currency: The target currency. MUST be a 3-letter ISO currency code (e.g., 'EUR', 'JPY', 'PLN').

    Returns:
    A string with the converted amount and currency code (e.g., "450.32 USD"),
    or an error message string if the conversion fails.
    """
    from_curr = from_currency.upper()
    to_curr = to_currency.upper()

    if from_curr == to_curr:
        logger.info(
            f"Currencies are identical ({from_curr}). Returning original amount."
        )
        return f"{amount} {to_curr}"

    url = EXCHANGER_URL
    params = {"amount": amount, "from": from_curr, "to": to_curr}
    try:
        logger.info(
            f"Requesting currency conversion: {amount} {from_curr} to {to_curr}"
        )

        response = requests.get(url=url, params=params, timeout=10)

        response.raise_for_status()

        data = response.json()
        transferred_amount = data["rates"][to_curr]

        logger.info(f"Successfully converted to: {transferred_amount} {to_curr}")
        return f"{transferred_amount} {to_curr}"

    except requests.exceptions.RequestException as e:
        return f"Conversion failed: Unable to reach the exchange rate API. Details: {e}"

    except KeyError as e:
        logger.error(f"Currency code not found in the API response: {e}. Data: {data}")
        return f"Conversion failed: The currency code {to_curr} is not supported."

    except Exception as e:
        logger.exception(f"Unexpected error in transfer_currency tool: {e}")
        return "Conversion failed: An internal unexpected error occurred."


def make_neighbor_tool(collection):
    def get_neighboring_chunks(chunk_id: str) -> str:
        """Retrieves the text chunks immediately before and after a given chunk in the document.

        Use this tool after search_documents when the returned chunk appears cut off,
        incomplete, or missing context — for example, a table row without its header,
        a sentence that starts or ends abruptly, or a list item without its introduction.
        Pass the chunk_id shown in the search_documents output (e.g., from "--- chunk 5 ---",
        pass "5") to retrieve chunks 4 and 6, restoring the surrounding context.

        Args:
            chunk_id: The identifier of the chunk that needs more context, as a string
                (e.g., "5"). This must be an id previously seen in search_documents output.

        Returns:
            A string containing the neighboring chunk(s), each prefixed with its own
            "--- chunk {id} ---" identifier. Returns a message noting that no neighbors
            exist if the chunk is at the very start or end of the document.
        """
        current_id = int(chunk_id)
        total_chunks = collection.count()

        candidate_ids = [current_id - 1, current_id + 1]
        valid_ids = [str(i) for i in candidate_ids if 0 <= i < total_chunks]

        if not valid_ids:
            return "No neighboring chunks available — this is the only chunk in the document."

        neighbors = collection.get(ids=valid_ids)
        parts = [
            f"--- chunk {id_} ---\n\n{text}"
            for id_, text in zip(neighbors["ids"], neighbors["documents"])
        ]
        return "\n\n".join(parts)

    return get_neighboring_chunks


def answer_with_tools(
    client, query, search_tool, neightboring_tool, system_prompt, model=DEFAULT_MODEL
):
    response = client.models.generate_content(
        model=model,
        contents=query,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[search_tool, neightboring_tool, transfer_currency],
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
