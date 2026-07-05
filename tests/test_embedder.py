from rag_assistant.retrieval.embedder import embed_texts
from unittest.mock import Mock


def test_embed_texts():
    fake_client = Mock()
    fake_texts = ["Hello, world!", "This is a test."]
    fake_embeddings = [
        Mock(values=[0.1, 0.2, 0.3]),
        Mock(values=[0.4, 0.5, 0.6]),
    ]
    fake_client.models.embed_content.return_value = Mock(embeddings=fake_embeddings)

    result = embed_texts(fake_client, fake_texts)

    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
