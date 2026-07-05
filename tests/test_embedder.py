from rag_assistant.retrieval.embedder import embed_texts
from rag_assistant.retrieval.embedder import cosine_similarity
from rag_assistant.retrieval.embedder import find_relevent_chunks
from unittest.mock import Mock
import pytest


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


def test_cosine_similarity():
    vec_a = [1, 0, 0]
    vec_b = [0, 1, 0]
    vec_c = [1, 1, 0]

    assert cosine_similarity(vec_a, vec_b) == pytest.approx(0.0)
    assert cosine_similarity(vec_a, vec_c) == pytest.approx(0.7071067811865475)
    assert cosine_similarity(vec_b, vec_c) == pytest.approx(0.7071067811865475)
    assert cosine_similarity(vec_a, vec_a) == pytest.approx(1.0)


def test_find_relevent_chunks():
    fake_client = Mock()
    fake_query = "What is the capital of France?"
    fake_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
    fake_chunk_embeddings = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9],
    ]
    fake_query_embedding = Mock(values=[0.15, 0.25, 0.35])
    fake_client.models.embed_content.return_value = Mock(
        embeddings=[fake_query_embedding]
    )

    result = find_relevent_chunks(
        fake_client, fake_query, fake_chunk_embeddings, fake_chunks, top_n=2
    )

    assert len(result) == 2
    assert result == ["Chunk 1", "Chunk 2"]
