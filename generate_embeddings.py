from langchain.embeddings import HuggingFaceInferenceAPI

def generate_embeddings(chunks, api_token):
    embeddings = HuggingFaceInferenceAPI(api_token=api_token)
    return embeddings.embed_documents([chunk.page_content for chunk in chunks])
