# Pizza Restaurant QA System

This application uses LLM technology to answer questions about a pizza restaurant based on customer reviews.

## Overview

The system uses a vector store to retrieve relevant restaurant reviews based on user questions, then processes these reviews using the Llama 3.2 language model to generate informed answers.

## Features

- Natural language question answering about the pizza restaurant
- Review retrieval using semantic search
- Interactive command-line interface

## Requirements

- Python 3.8+
- Ollama with llama3.2 and mxbai-embed-large models installed

## Installation

1. Clone this repository
2. Install the required packages:

```bash
pip install -r requirements.txt
```

3. Make sure Ollama is running locally with the required models:

```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

4. Ensure you have the `realistic_restaurant_reviews.csv` file in the project directory

## Usage

1. Start the application:

```bash
python main.py
```

2. Ask questions about the pizza restaurant in the interactive prompt
3. Type 'q' to quit the application

## Project Structure

- `main.py`: Contains the main application logic and user interaction loop
- `vector.py`: Handles the vector store creation and retrieval functionality
- `requirements.txt`: Lists all dependencies
- `realistic_restaurant_reviews.csv`: Dataset of restaurant reviews (not included in repository)

## Example Questions

- "What do people say about the pepperoni pizza?"
- "Is the restaurant good for families?"
- "What's the average rating of this restaurant?"
- "Are there any complaints about the service?" 
