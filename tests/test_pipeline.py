from unittest.mock import patch, Mock
from rag_assistant.pipeline import answer_question


@patch("rag_assistant.pipeline.find_relevant_chunks")
def test_answer_question(mock_find_chunks):
    mock_find_chunks.return_value = ["Chunk A", "Chunk B"]
    client = Mock()
    chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    query = "What is the relevant information?"
    chunk_embedding = [0.1, 0.2, 0.3]
    system_prompt = "You are a helpful assistant."
    mocked_answer = Mock(text="Mocked answer!")
    client.models.generate_content.return_value = mocked_answer
    answer = answer_question(client, query, chunks, chunk_embedding, system_prompt)

    assert answer == mocked_answer.text
    client.models.generate_content.assert_called_once()
