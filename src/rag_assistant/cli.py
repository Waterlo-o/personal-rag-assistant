import os
import sys
import logging

from google import genai
from dotenv import load_dotenv
from google.genai.errors import APIError
from google.genai import types

from rag_assistant.constants import DEFAULT_MODEL

logger = logging.getLogger(__name__)


def load_system_prompt(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            system_prompt = f.read()
            return system_prompt
    except FileNotFoundError:
        sys.exit(f"System prompt file not found: {path}")


def get_client():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        print("GEMINI_API_KEY is set.")
    else:
        sys.exit("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    return client


def ask_with_history(
    client, history, question, model=DEFAULT_MODEL, system_prompt=None
):
    history.append({"role": "user", "parts": [{"text": question}]})

    try:
        response = client.models.generate_content(
            model=model,
            contents=history,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        answer = response.text
        history.append({"role": "model", "parts": [{"text": answer}]})
    except APIError as e:
        logger.exception(f"API error for question: {question}")
        return f"An error occurred while generating the response: {e}", history

    logger.info(f"Request has been done successfully for question: {question}")
    return answer, history
