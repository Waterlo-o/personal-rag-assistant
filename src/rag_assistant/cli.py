import os
import sys

from google import genai
from dotenv import load_dotenv
from google.genai.errors import APIError
from google.genai import types

DEFAULT_MODEL = "gemini-3.1-flash-lite"

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
        print("GEMINI_API_KEY is set." )
    else:
        sys.exit("GEMINI_API_KEY is not set.")

    client = genai.Client(api_key=api_key)

    return client

def ask_with_history(client, history, question, model=DEFAULT_MODEL, system_prompt=None):

    history.append({"role": "user", "parts": [{"text":question}]})

    try:
        response = client.models.generate_content(
                                                  model=model, 
                                                  contents=history, 
                                                  config=types.GenerateContentConfig(system_instruction=system_prompt)
                                                  )
        answer = response.text
        history.append({"role": "model", "parts": [{"text":answer}]})
    except APIError as e:
        return f"An error occurred while generating the response: {e}", history

    return answer, history

if __name__ == "__main__":

    client = get_client()
    history = []
    system_prompt = load_system_prompt("config/system_prompt.txt")

    while True:
        try:
            question = input("Enter your question: ")
            answer, history = ask_with_history(client, history, question, system_prompt=system_prompt)
            print(answer)
            print("--------------------------------------------------")
        except KeyboardInterrupt:
            print("\nbye!")
            break
