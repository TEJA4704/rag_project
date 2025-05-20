import os
from langchain.document_loaders import TextLoader, PyPDFLoader

def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            loader = TextLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(directory, filename))
            documents.extend(loader.load())
    return documents
