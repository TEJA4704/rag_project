def query_index(index, embeddings, query, k=5):
    query_embedding = embeddings[0].reshape(1, -1)  # Assume query is a single token
    distances, indices = index.search(query_embedding, k)
    return indices
