# RAG Project for Internal Document Querying

This project implements a Retrieval-Augmented Generation (RAG) system to query internal documents. It retrieves relevant information from a collection of documents and uses that information to generate a more informed and context-aware response to user queries.

## Project Structure

The project is organized as follows:

*   **`rag_project/`**:  Root directory of the project.
*   **`documents/`**: Contains sample internal documents in various formats (PDF, TXT, etc.). This is where you'll place your own documents for testing and deployment.
*   **`models/`**:  This directory holds the LlamaCpp model files (e.g., `llama-7b.gguf`).  You're responsible for downloading and placing the appropriate models here.
*   **`code/`**: Contains all the Python scripts implementing the RAG pipeline.
    *   **`load_documents.py`**:  Loads documents from the `documents/` directory. Handles various document formats.
    *   **`split_documents.py`**: Splits the loaded documents into smaller chunks for efficient indexing and retrieval.
    *   **`generate_embeddings.py`**: Generates embeddings (vector representations) for the document chunks.  This is crucial for semantic search.
    *   **`build_vector_store.py`**: Creates a vector store (e.g., using FAISS, ChromaDB, etc.) and indexes the document embeddings.
    *   **`query_vector_store.py`**: Performs semantic search against the vector store to retrieve relevant document chunks based on user queries.
    *   **`generate_response.py`**:  Uses the retrieved document chunks and the user query to generate a final, augmented response using a Large Language Model (LLM).
    *   **`main.py`**: The main entry point for running the RAG pipeline.  It orchestrates the different steps.
*   **`.env`**:  This file stores environment variables (e.g., API keys, model paths, etc.).  **Important:** This file should **not** be committed to version control. Add it to your `.gitignore` file.
*   **`README.md`**: This file, providing an overview of the project.
*   **`technical_report.md`**: A more detailed technical report describing the architecture, implementation details, and results of the project.

## Prerequisites

*   Python 3.9+
*   Langchain
*   Sentence Transformers
*   FAISS
*   LlamaCpp

## Installation

1.  Clone the repository:
    ```bash
    git clone [repository URL]
    ```
2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/macOS
    venv\Scripts\activate     # Windows
    ```
3.  Install dependencies:
    ```bash
    pip install langchain sentence-transformers faiss-cpu llama-cpp-python streamlit python-dotenv
    ```
4.  Set up the `.env` file:
    *   Obtain a Hugging Face API token and add it to the `.env` file:
        ```
        HUGGINGFACE_API_TOKEN=YOUR_HUGGINGFACE_API_TOKEN
        ```
    *   Specify the path to your LlamaCpp model:
        ```
        LLAMA_MODEL_PATH=models/llama-7b.gguf
        ```

## Running the Project

1.  Place your documents in the `documents/` directory (e.g., `.txt` files).
2.  Place the LlamaCpp model file in the `models/` directory.
3.  Run the main script:
    ```bash
    python code/main.py
    ```

## Technical Details

The system uses `HuggingFaceInferenceAPI` for embedding generation and `faiss` for efficient vector search.  The `LlamaCpp` library is used for generating responses.

## Future Enhancements

*   Web Interface: Add a Streamlit/Flask UI for user interaction.
*   Real-Time Data Sync: Integrate with Confluence or SharePoint APIs.
*   Feedback Loop: Allow users to rate responses and refine embeddings.
*   GPU Acceleration: Use CUDA-enabled FAISS for faster inference.
*   Summarization: Add a summarizer to condense long documents.
