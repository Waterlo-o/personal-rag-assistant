from rag_assistant.ingestion.chunker import find_nearest_char
from rag_assistant.ingestion.chunker import chunk_text


def test_find_nearest_char():
    text = "This is a test. This is only a test. Please ignore."

    # Period is within the initial search radius
    nearest_index1 = find_nearest_char(text, position=10, radius=5)
    assert nearest_index1 == 14

    # Period is outside the initial radius, requires recursive expansion
    nearest_index2 = find_nearest_char(text, position=0, radius=5)
    assert nearest_index2 == 14

    # No periods at all, should return None instead of recursing forever
    text2 = "This is a test with no periods"
    nearest_index3 = find_nearest_char(text2, position=10, radius=5)
    assert nearest_index3 is None


def test_chunk_text():
    text = "This is a test. This is only a test. Please ignore this message. Thank you."
    chunks = chunk_text(text, chunk_size=20, overlap=5)

    # Ensure that the chunks are created correctly
    assert len(chunks) > 0

    for i in range(len(chunks) - 1):
        assert chunks[i][-1] == "."  # Each chunk should end with a period

    assert "This is a test." in chunks[0]
    assert "Thank you." in chunks[-1]
