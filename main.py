# Import required libraries for LLM and prompt handling
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

# Initialize the Ollama language model with llama3.2
model = OllamaLLM(model="llama3.2")

# Define the template for the chat prompt
# This template sets up the AI as a pizza restaurant expert
template ="""
you are an expert in answering questions about a pizza restaurant 

Here are some revelant reviews :{reviews}

Here is the question to answer: {question}

"""
# Create a prompt template from the defined template
prompt = ChatPromptTemplate.from_template(template)
# Create a chain that combines the prompt template with the model
chain = prompt | model

# Main interaction loop
while True:
    # Display a separator line for better readability
    print("\n\n--------------------------------------")
    # Get user input for their question
    question = input("Ask your question (q to quit): ")
    print("\n\n")
    
    # Check if user wants to quit
    if question == "q":
        break

    # Retrieve relevant reviews using the vector store retriever
    reviews = retriever.invoke(question)
    # Generate response using the chain with reviews and question
    result = chain.invoke({"reviews": reviews, "question":question})
    # Display the generated response
    print(result)
