from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np

def main():
    # Initialize the Ollama model
    llm = Ollama(model="llama2")
    
    # Initialize the sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Example text to process
    text = """
    LangChain is a framework for developing applications powered by language models.
    It enables applications that are context-aware and reason about their responses.
    The framework provides a standard interface for chains, agents, and tools.
    """
    
    # Split the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    chunks = text_splitter.split_text(text)
    
    # Generate embeddings for each chunk
    embeddings = model.encode(chunks)
    
    # Example query
    query = "What is LangChain?"
    query_embedding = model.encode(query)
    
    # Calculate cosine similarity between query and chunks
    similarities = np.dot(embeddings, query_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # Get the most relevant chunk
    most_relevant_idx = np.argmax(similarities)
    most_relevant_chunk = chunks[most_relevant_idx]
    
    # Generate a response using Ollama
    prompt = f"Based on this context: {most_relevant_chunk}\n\nQuestion: {query}"
    response = llm.invoke(prompt)
    
    print("Query:", query)
    print("\nMost relevant context:", most_relevant_chunk)
    print("\nResponse:", response)

if __name__ == "__main__":
    main() 