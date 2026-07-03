import os
import sys

from google import genai
from dotenv import load_dotenv
from google.genai.errors import APIError

DEFAULT_MODEL = "gemini-3.1-flash-lite"

def get_client():

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if api_key:
        print("GEMINI_API_KEY is set." )
    else:
        sys.exit("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    return client

def ask(client, question, model=DEFAULT_MODEL):
    
    try:
        response = client.models.generate_content(model=model, contents=question)
        return response.text
    except APIError as e:
        return f"An error occurred while generating the response: {e}"

if __name__ == "__main__":

    client = get_client()
    answer = ask(client, "What is the capital of France?")
    
    print(answer)