from rag_assistant.cli import load_system_prompt
from rag_assistant.cli import ask_with_history
from unittest.mock import Mock
from google.genai.errors import APIError

fake_client = Mock()


def test_load_system_prompt(tmp_path):
    temp_file = tmp_path / "dummy_file.txt"
    expected_content = "Dummy system prompt content"
    temp_file.write_text(expected_content)

    temp_test = load_system_prompt(temp_file)

    assert temp_test == expected_content


def test_ask_with_history():
    fake_history = []
    fake_client = Mock()

    fake_question = "What is the capital of France?"
    fake_answer = Mock(text="This is mocked response")
    fake_client.models.generate_content.return_value = fake_answer

    answer, history = ask_with_history(fake_client, fake_history, fake_question)

    assert answer == "This is mocked response"
    assert history == [
        {"role": "user", "parts": [{"text": fake_question}]},
        {"role": "model", "parts": [{"text": "This is mocked response"}]},
    ]


def test_ask_with_history_api_error():
    fake_history = []
    fake_client = Mock()

    fake_question = "What is the capital of France?"
    fake_error = APIError(500, {"error": {"message": "Service unavailable"}})
    fake_client.models.generate_content.side_effect = fake_error

    answer, history = ask_with_history(fake_client, fake_history, fake_question)

    expected_message = f"An error occurred while generating the response: {fake_error}"
    assert answer == expected_message
    assert history == [{"role": "user", "parts": [{"text": fake_question}]}]
