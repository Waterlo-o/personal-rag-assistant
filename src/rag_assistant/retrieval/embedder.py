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


def find_relevant_chunks(client, query, chunk_embeddings, chunks, top_n=3):
    query_embedding = embed_texts(client, [query])

    cosine_sim = []
    for i in chunk_embeddings:
        cosine_sim.append(cosine_similarity(query_embedding[0], i))

    sorted_indices = sorted(enumerate(cosine_sim), key=lambda x: x[1], reverse=True)
    sorted_chunks = [chunks[i] for i, _ in sorted_indices]
    return sorted_chunks[:top_n]
