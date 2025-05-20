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


