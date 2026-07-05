DEFAULT_MODEL = "gemini-embedding-001"


def embed_texts(client, texts, model=DEFAULT_MODEL):
    response = client.models.embed_content(
        model=model,
        contents=texts,
    )

    return [embedding.values for embedding in response.embeddings]
