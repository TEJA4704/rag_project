# RAG Project for Internal Document Querying

This project implements a Retrieval-Augmented Generation (RAG) system to query internal documents.

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
