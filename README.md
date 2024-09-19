# AI Governance Tracker Chatbot

![image](https://github.com/user-attachments/assets/9a19809e-327d-479b-9e63-379f5cc06ec6)

This repository contains a Streamlit-based chatbot that answers questions about AI governance and regulations by scraping, processing, and storing relevant articles in a ChromaDB vector store. The chatbot uses advanced natural language processing techniques to retrieve answers strictly from the scraped articles, with no external knowledge. The project employs the LangChain framework, Ollama embeddings, and ChromaDB for efficient document retrieval.

## Architecture

![image](https://github.com/user-attachments/assets/26cee0a9-c76f-4883-b54e-580d02264cd7)


## Features

- **Web Scraping:** Automatically scrapes AI governance-related articles from the web.
- **Document Embeddings:** Utilizes SentenceTransformer and Ollama embeddings to convert text into vector representations.
- **Vector Storage:** Stores and manages article embeddings in a ChromaDB collection.
- **Retrieval-based QA:** Uses a retrieval-based approach to answer questions, relying solely on the stored document data.
- **Streamlit Interface:** Provides an easy-to-use web interface for querying AI governance content.

## Requirements

To run the project, ensure you have the following installed:

- Python 3.12
- `requests`
- `beautifulsoup4`
- `PyPDF2`
- `streamlit`
- `chromadb`
- `langchain`
- `sentence-transformers`
- `uuid`
- `Ollama`

## How It Works
Web Scraping:
The app scrapes AI governance articles from specific URLs using BeautifulSoup. The articles are parsed and structured into headings and content sections.

Document Processing:
Articles are processed using RecursiveCharacterTextSplitter from LangChain to break down larger texts into smaller, manageable chunks.

Vector Storage:
The SentenceTransformer model all-MiniLM-L6-v2 generates embeddings for each document chunk. These embeddings are stored in ChromaDB, a high-performance vector store.

Question Answering:
Users input questions into the chatbot, which retrieves the most relevant document chunks using a multi-query retriever and generates answers based strictly on the retrieved context using the ChatOllama model.


