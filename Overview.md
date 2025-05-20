# Project Overview
Objective: Build a RAG system that allows users to query internal documents (e.g., policies, training materials, project reports) using a combination of document chunking, vector embeddings, and an LLM.

Key Features:

  Document Chunking: Split documents into manageable chunks for efficient retrieval.
  Vector Embeddings: Use pre-trained models to create dense vector representations of text.
  Vector Store: Index embeddings in FAISS for fast similarity search.
  LLM: Use LlamaCpp for answer generation.
  Modular Code: Separated into reusable components for clarity and scalability.

Open Source Document Sources:

  GitHub: Use repositories with sample internal documents (e.g., policies, FAQs).
  Kaggle/UCI Data Repositories: Access datasets that can simulate internal files.
  Data.gov: Use public datasets for training or testing.
  Mock Generators: Use tools like Faker or docxtpl to create synthetic documents.
