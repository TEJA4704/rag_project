import os
from dotenv import load_dotenv

load_dotenv()

# Step 1: Load and split documents
documents = load_documents.load_documents("documents")
chunks = split_documents.split_documents(documents)

# Step 2: Generate embeddings and build index
api_token = os.getenv("HUGGINGFACE_API_TOKEN")  # Replace with your token
embeddings = generate_embeddings.generate_embeddings(chunks, api_token)
index = build_vector_store.build_faiss_index(embeddings)

# Step 3: Query and generate response
query = input("Enter your query: ")
indices = query_vector_store.query_index(index, embeddings, query)
retrieved_chunks = [chunks[i].page_content for i in indices]
response = generate_response.generate_response(query, retrieved_chunks, os.getenv("LLAMA_MODEL_PATH"))
print(response)

