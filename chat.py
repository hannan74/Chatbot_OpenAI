import os
import openai
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings


# Set API Key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Step 1: Data Preparation
def prepare_data(file_path):
    """Load text data and split it into chunks."""
    with open(file_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Step 2: Embedding and Vector Store Creation
def create_vector_store(docs, embedding_model="text-embedding-ada-002"):
    """Generate embeddings and store them in FAISS vector store."""
    embeddings = OpenAIEmbeddings(model=embedding_model)
    vector_store = FAISS.from_texts(docs, embeddings)
    return vector_store

# Step 3: Retrieval-Augmented Generation (RAG) Pipeline
def create_rag_pipeline(vector_store, model="gpt-4o-mini"):
    """Set up retrieval-augmented generation using the FAISS store and a GPT model."""
    retriever = vector_store.as_retriever()
    llm = OpenAI(model=model)  # Use fine-tuned GPT model here
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Step 4: Fine-tuning Function
def fine_tune_model(training_file_id, model="gpt-3.5-turbo"):
    """Fine-tune the model automatically using the provided training data."""
    fine_tuning_response = openai.FineTune.create(
        model=model,
        training_file=training_file_id,  # This is the ID of your training file
        n_epochs=1  # Number of fine-tuning epochs
    )
    return fine_tuning_response

# Upload the training file
file_response = openai.File.create(
    file=open("training_data.jsonl"),  # Path to your .jsonl file
    purpose='fine-tune'
)

# Retrieve the file ID
training_file_id = file_response['id']
print(f"File uploaded successfully. File ID: {training_file_id}")


# Step 5: Main Program
def main():
    # File paths
    data_path = "knowledge_base.txt"  # Input knowledge base file
    training_data_file_id = training_file_id  # Replace with your fine-tuning training file ID

    # Step 1: Load and process data
    print("Loading and preparing data...")
    documents = prepare_data(data_path)

    # Step 2: Create vector store
    print("Generating embeddings and creating vector store...")
    vector_store = create_vector_store(documents)

    # Step 3: Fine-tune the model (automatic)
    print("Fine-tuning the model...")
    fine_tuning_response = fine_tune_model(training_data_file_id)
    print(f"Fine-tuning response: {fine_tuning_response}")

    # Step 4: Set up RAG pipeline with fine-tuned model
    print("Setting up RAG pipeline with fine-tuned model...")
    rag_pipeline = create_rag_pipeline(vector_store, model="gpt-4o-mini")  # Using the fine-tuned GPT model here

    # Step 5: Chatbot-like interaction
    print("Chatbot is ready. Type your queries below:")
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            print("Exiting chatbot.")
            break
        print("Bot: Searching for an answer...")
        response = rag_pipeline.run(query)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main()
