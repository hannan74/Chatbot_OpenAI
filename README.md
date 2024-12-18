# Chatbot-with-Fine-Tuning

## Overview

This project demonstrates how to build a Retrieval-Augmented Generation (RAG) based chatbot using OpenAI's GPT models. The system uses the FAISS vector store to store and retrieve document embeddings and fine-tunes the GPT model with custom training data to enhance response accuracy. The chatbot is capable of providing insightful answers based on a given knowledge base and allows fine-tuning to make it more specialized.

## Features

- **Document Embedding**: Converts a given knowledge base into embeddings using OpenAI's embeddings model.
- **FAISS Vector Store**: Utilizes FAISS for fast similarity search and retrieval of document chunks.
- **Fine-Tuning**: Fine-tune the GPT model with custom data to make it more domain-specific.
- **Interactive Chatbot**: Engage in real-time conversations and retrieve answers based on the knowledge base.
- **Retrieval-Augmented Generation**: Combines document retrieval with a GPT-based model to provide more accurate and contextually relevant responses.

## Requirements

- Python 
- OpenAI Python client (`openai`)
- LangChain library (`langchain`)
- FAISS library 

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/RAG-Chatbot-with-Fine-Tuning.git
   cd RAG-Chatbot-with-Fine-Tuning
