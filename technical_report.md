# Technical Report

## Design Choices

*   **Document Chunking**:  We used `RecursiveCharacterTextSplitter` to split documents into 500-character chunks with a 50-character overlap. This approach helps retain crucial context between chunks, leading to more coherent responses.
*   **Vector Embeddings**:  We leveraged HuggingFace's `sentence-transformers` for generating dense vector embeddings through the `HuggingFaceInferenceAPI`.  This choice balances embedding quality with API accessibility.
*   **Vector Store**:  We implemented FAISS (Facebook AI Similarity Search) for efficient similarity search.  FAISS's L2 distance metric is well-suited for this application.
*   **LLM**: We chose LlamaCpp for lightweight deployment and compatibility with local models.  This allows for offline operation and reduces dependency on external APIs.

## Implementation Details

*   **Modular Code**: The system is structured with distinct files for document loading, splitting, embedding, indexing, querying, and response generation. This modular design enhances reusability and maintainability.
*   **Scalability**: FAISS supports large-scale indexing, and chunking strategies ensure memory efficiency.  The system is designed to handle a substantial document corpus.
*   **Customization**: The architecture is flexible and allows users to replace the LLM with other models (e.g., HuggingFace Transformers, OpenAI API) with minimal code changes.

## Evaluation

*   **Test Cases**:
    *   **Query 1**: *"What are the company's data privacy policies?"*
        *   **Expected Output**: Retrieve and summarize relevant chunks from a PDF policy document.
    *   **Query 2**: *"How to submit a project proposal?"*
        *   **Expected Output**: Extract steps from a training manual.

*   **Limitations**:
    *   LlamaCpp may encounter limitations with very long context windows.
    *   FAISS requires manual scaling for distributed system implementations.  This assumes a single machine setup.

## Future Enhancements

1.  **Web Interface**:  Develop a Streamlit or Flask-based UI for improved user interaction.
2.  **Real-Time Data Sync**: Integrate with Confluence or SharePoint APIs to synchronize with live data sources.
3.  **Feedback Loop**:  Implement a mechanism for users to rate responses and refine embeddings based on feedback.
4.  **GPU Acceleration**: Utilize CUDA-enabled FAISS for accelerated inference.
5.  **Summarization**:  Add a summarization module to condense long documents or generated responses.

## Running the Project

1.  **Setup:**

    *   Place sample documents in the `documents/` folder.
    *   Place the LlamaCpp model file (e.g., `llama-7b.gguf`) in the `models/` folder.
    *   Create a `.env` file with the following environment variables:

    ```
    HUGGINGFACE_API_TOKEN=YOUR_HUGGINGFACE_API_TOKEN
    LLAMA_MODEL_PATH=path/to/your/llama-7b.gguf
    ```

2.  **Execution:**

    ```bash
    python code/main.py
    ```

## Conclusion

This project delivers a robust, modular RAG system suitable for querying internal documents.  It combines best practices in NLP, efficient vector search, and lightweight LLM inference. The extensible architecture allows for adaptation and enterprise integration with minimal adjustments.

