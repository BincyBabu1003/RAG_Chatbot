# PDF RAG Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that allows users to upload PDF documents and ask questions about their content.

The system extracts text from PDF files, converts the text into embeddings using Hugging Face Sentence Transformers, stores the embeddings in a FAISS vector database, and retrieves the most relevant document chunks based on user queries.

## Live Demo
[https://ragchatbot-1003.streamlit.app/]


## Features

* Upload PDF documents
* Extract text from PDFs
* Intelligent text chunking
* Semantic search using embeddings
* FAISS vector database
* Interactive Streamlit interface
* Retrieval-Augmented Generation (RAG) pipeline

## Tech Stack

* Python
* Streamlit
* LangChain
* Hugging Face Embeddings
* Sentence Transformers
* FAISS
* PyPDF

## Project Structure

rag_chatbot/

├── app.py

├── ingest.py

├── requirements.txt

├── README.md

├── screenshots/

└── data/

## Installation

1. Clone the repository

git clone <repository-url>

2. Create a virtual environment

python -m venv myenv

3. Activate the virtual environment

Windows:

myenv\Scripts\activate

4. Install dependencies

pip install -r requirements.txt

## Run the Application

streamlit run app.py

## Run Locally
pip install -r requirements.txt
streamlit run app.py

## RAG Pipeline

PDF Upload

↓

Text Extraction

↓

Chunking

↓

Embeddings

↓

FAISS Vector Store

↓

Retriever

↓

Question Answering

## Future Improvements

* Multi-PDF support
* Conversation memory
* Groq API integration
* Ollama integration
* Source citations
* Cloud deployment

## Author

Bincy
Master's in Data Science
University of Skövde
Sweden


