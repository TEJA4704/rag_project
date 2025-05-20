# Technical Report

## 1. Introduction

* This report details the architecture, implementation, and results of a Retrieval-Augmented Generation (RAG) system designed to enable querying of internal documents.
* The system aims to provide users with accurate and relevant answers to questions based on a collection of company documents, leveraging the strengths of both retrieval and generative AI models.
* This system addresses the challenge of quickly and efficiently extracting information from a growing repository of internal documentation, bypassing the need for manual searching and comprehension.

## 2. System Architecture

The RAG system follows a modular architecture composed of four primary stages: Document Loading and Preprocessing, Embedding Generation and Indexing, Query Processing and Retrieval, and Response Generation.

*   **Document Loading and Preprocessing**: This stage involves loading documents from a local directory. A TextLoader and PyPdfLoaderfrom Langchain is utilized to load the content of each document.
*   **Embedding Generation and Indexing**: Loaded documents are split into smaller chunks to facilitate more granular retrieval. A RecursiveCharacterTextSplitter is employed for this purpose, balancing chunk size and overlap to maximize information retention. The text chunks are then converted into vector embeddings using the HuggingFaceInferenceAPI with a suitable sentence transformer model hosted on Hugging Face's Inference API. These embeddings are stored in a FAISS index (faiss.IndexFlatL2) for efficient similarity search.
*   **Query Processing and Retrieval**: User queries are transformed into vector embeddings using the same embedding model used for the documents. The FAISS index is then queried to retrieve the k most similar document chunks based on cosine similarity.
*   **Response Generation**: The retrieved document chunks are provided as context to a local Large Language Model (LLM) implemented through LlamaCpp. The LLM generates a coherent and informative response to the user's query, leveraging the retrieved information.

## 3. Implementation Details

*   **Programming Language:** Python 3.9+
*   **Libraries:**
    *   **Langchain:** For document loading, text splitting, and pipeline orchestration.
    *   **Sentence Transformers (via HuggingFaceInferenceAPI):** For generating document and query embeddings.
    *   **FAISS (Facebook AI Similarity Search):** For building and querying a vector index.
    *   **LlamaCpp:** For running a local Large Language Model.
    *   **NumPy:** For numerical operations.
    *   **dotenv:** For managing environment variables.
*   **Environment Variables:**
    *   `HUGGINGFACE_API_TOKEN`: Hugging Face API Token for embedding generation.  **REQUIRED**
    *   `LLAMA_MODEL_PATH`: Path to the LlamaCpp model file (e.g., `models/llama-7b.gguf`). **REQUIRED**
*   **Chunking Strategy:** `RecursiveCharacterTextSplitter` with `chunk_size=500` and `chunk_overlap=50`.
*   **Embedding Model:** Default model provided by HuggingFaceInferenceAPI.
*   **LLM:** LlamaCpp with specified model file.
*   **FAISS Index Type:** `IndexFlatL2`.
*   **Number of Retrieved Chunks (k):** 5.
*   **LlamaCpp Prompt Engineering:** Simple contextual prompt to guide the LLM's response.
  
## 4. Results and Evaluation

*   **Performance**: The system demonstrates reasonable performance in retrieving relevant document chunks. The retrieval speed is dependent on the size of the document collection and the efficiency of the FAISS index. Embedding generation speed is tied to the response time of the Hugging Face Inference API.
*   **Accuracy**: The accuracy of the generated responses is influenced by the quality of the retrieved chunks and the LLM's ability to synthesize information. Subjective evaluation suggests that the system provides generally accurate and relevant answers, though occasionally the LLM may introduce inaccuracies or miss relevant information.
*   **Scalability**: The current implementation exhibits limitations in scalability. For very large document collections, the FAISS index will require more memory and the embedding generation process may become a bottleneck.
*   **Qualitative Assessment**: The system has proven effective in answering straightforward questions about company policies and procedures. It significantly reduces the time required to find specific information compared to manual searching. However, it struggles with complex or nuanced queries that require deeper understanding of the context.

## 5. Future Enhancements

*   **More Sophisticated Chunking**: Implement more advanced chunking strategies that consider semantic meaning and context.
*   **Advanced Embedding Models**: Explore more powerful and context-aware embedding models.
*   **Fine-tuning**: Consider fine-tuning the LLM on a corpus of company-specific documents to improve response quality.
*   **User Interface**: Develop a user-friendly interface for querying the system.
*   **Evaluation Metrics**: Implement automated evaluation metrics to track the accuracy and relevance of the system’s responses.
*   **Error Handling and Logging**: Improve error handling and implement robust logging for debugging and monitoring.
*   **Dynamic Embedding Model Selection**: Allow for switching between different embedding models based on task requirements.
*   **Advanced Prompt Engineering**: Utilize more complex and tailored prompts to guide the LLM's response and improve the quality and relevance of the generated text.

## 6. Conclusion

The RAG system successfully demonstrates the potential for leveraging AI to enhance internal knowledge management. The system provides a valuable tool for employees to quickly and efficiently access relevant information, improving productivity and knowledge sharing. Further development and refinement, focusing on the enhancements outlined in this report, will further strengthen the system’s capabilities and impact.

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

