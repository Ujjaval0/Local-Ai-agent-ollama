# Import required libraries for embeddings, vector store, and data handling
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import os
import pandas as pd

# Load the restaurant reviews dataset from CSV file
df = pd.read_csv("realistic_restaurant_reviews.csv")

# Initialize the Ollama embeddings model for converting text to vectors
embeddings = OllamaEmbeddings(model ="mxbai-embed-large")

# Define the location for storing/loading the FAISS vector database
db_location = "./faiss_db"
# Check if we need to create new documents or load existing ones
add_documents = not os.path.exists(db_location)

# If the database doesn't exist, create it from the reviews
if add_documents:
    # Initialize an empty list to store document objects
    documents = []
    
    # Process each review in the dataset
    for i, row in df.iterrows():
        # Create a Document object combining title and review
        document = Document(
            page_content = row["Title"] + " " + row["Review"],
            metadata = {"rating": row["Rating"], "date": row["Date"]}
        )
        documents.append(document)

    # Create a new FAISS vector store from the documents
    vector_store = FAISS.from_documents(documents, embeddings)
    # Save the vector store to disk for future use
    vector_store.save_local(db_location)
else:
    # Load the existing vector store from disk
    vector_store = FAISS.load_local(db_location, embeddings, allow_dangerous_deserialization=True)

# Create a retriever that will find the most relevant reviews for a given query
# k=5 means it will return the 5 most relevant reviews
retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)