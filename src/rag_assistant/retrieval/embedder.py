from rag_assistant.constants import DEFAULT_EMBENDDING_MODEL


def embed_texts(client, texts, model=DEFAULT_EMBENDDING_MODEL):
    response = client.models.embed_content(
        model=model,
        contents=texts,
    )

    return [embedding.values for embedding in response.embeddings]


# def find_relevant_chunks(client, query, chunk_embeddings, chunks, top_n=3):

#     query_vec = embed_texts(client, [query])[0]

#     query_array = np.array(query_vec)
#     embed_array = np.array(chunk_embeddings)

#     dot_product = embed_array @ query_array

#     query_norm = np.linalg.norm(query_array)
#     embed_norm = np.linalg.norm(embed_array, axis = 1)

#     similarities = dot_product / (query_norm * embed_norm)

#     best_indices = np.argsort(similarities)[-top_n:][::-1]

#     return [chunks[i] for i in best_indices]
