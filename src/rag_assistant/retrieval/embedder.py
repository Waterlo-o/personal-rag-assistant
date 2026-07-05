import numpy as np

DEFAULT_MODEL = "gemini-embedding-001"


def embed_texts(client, texts, model=DEFAULT_MODEL):
    response = client.models.embed_content(
        model=model,
        contents=texts,
    )

    return [embedding.values for embedding in response.embeddings]


def cosine_similarity(vec_a, vec_b):
    num = np.dot(vec_a, vec_b)
    denum = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denum == 0:
        return 0.0
    cosine_sim = num / denum
    return cosine_sim
