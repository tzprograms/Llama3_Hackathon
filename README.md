# Llama3_Hackathon
Personalized Learning System
This project is an Exam Preparation Assistant built with Streamlit that integrates Retrieval-Augmented Generation (RAG) to provide more personalized and accurate responses. The assistant can be used for various exam types (SAT, ACT, IELTS, TOEFL) and offers features like document uploads, chat-based interactions, and AI-generated responses.
🔑 Key Features

    User Authentication:
    Users can register and log in to track their progress and chat history.

    Chat with AI Tutor:
    The chatbot provides responses based on user queries, leveraging both pre-existing knowledge and uploaded study materials.

    Document Upload & Retrieval:
    Users can upload PDF files (study guides, notes, etc.). The system embeds these documents using sentence-transformer embeddings and stores them in a FAISS index for efficient retrieval during conversations.

    Retrieval-Augmented Generation (RAG):
    The chatbot uses RAG to:
        Retrieve relevant content from uploaded documents based on the user's query.
        Generate responses by combining the retrieved context with the user's prompt.

    Database Integration:
    User data and chat history are stored in an SQLite database for persistent storage. 

📦 Dependencies

    Streamlit: For creating the web interface.
    FAISS: For efficient similarity search and retrieval.
    SentenceTransformers: For generating embeddings from text.
    SQLite3: For storing user data and chat history.
    PyPDF2: For extracting text from uploaded PDFs.

📖 How RAG Works in This App

    User uploads a document (e.g., a study guide).
    The system extracts text from the PDF and generates embeddings using a pretrained model from Hugging Face's Sentence Transformers.
    The embeddings are stored in a FAISS index for efficient retrieval.
    When a user asks a question, the system:
        Retrieves the most relevant content from the uploaded documents.
        Combines the retrieved content with the user's prompt to generate a response using an API-based LLM (e.g., Llama 2 (Together API)).