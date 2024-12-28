# Wikipedia Chatbot

A ChatGPT-like chatbot built with Python and Flask that uses LangChain and Llama 3 to answer user queries with information retrieved from Wikipedia. This project demonstrates how to build an interactive AI-powered assistant with a user-friendly frontend.

## Features

- **ChatGPT-Style Interface**: A simple and intuitive web UI that displays user inputs and AI responses.
- **Wikipedia Integration**: Fetches relevant Wikipedia articles for user queries and generates conversational summaries.
- **LangChain Tooling**: Uses LangChain's ReAct agent to reason and retrieve data efficiently.
- **Scalable Backend**: Powered by Flask for easy local or cloud deployment.


## How It Works

1. **User Input**: Users type a question or request in the input bar.
2. **Backend Processing**:
   - The input is sent to a Flask backend.
   - LangChain’s ReAct agent retrieves relevant Wikipedia content.
   - Llama 3 generates a summarized and conversational response.
3. **Response Display**: The AI's response is displayed in the chat interface.

## Requirements

- Python 3.8 or higher
- GPU (Optional, but highly recommended for running large models like Llama 3)
- Required Python libraries (see [Installation](#installation))

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/wikipedia-chatbot.git
   cd wikipedia-chatbot
