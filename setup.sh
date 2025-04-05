#!/bin/bash

set -e

echo "Setting up RAG LLM System..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is required but not installed."
    echo "Install it with: pip install uv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

# Determine the correct activation script based on shell
SHELL_NAME=$(basename "$SHELL")

if [ "$SHELL_NAME" = "fish" ]; then
    ACTIVATE_SCRIPT=".venv/bin/activate.fish"
    ACTIVATE_CMD="source $ACTIVATE_SCRIPT"
else
    ACTIVATE_SCRIPT=".venv/bin/activate"
    ACTIVATE_CMD="source $ACTIVATE_SCRIPT"
fi

echo "Activating virtual environment..."
echo "You'll need to manually run: $ACTIVATE_CMD"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "# OpenAI API key" > .env
    echo "OPENAI_API_KEY=your_openai_api_key_here" >> .env
    echo "Please update .env with your actual OpenAI API key."
fi

# Create documents directory if it doesn't exist
if [ ! -d "documents" ]; then
    echo "Creating documents directory..."
    mkdir -p documents
fi

# Make scripts executable
chmod +x rag_cli.py
chmod +x test_rag.py

# Instructions for the user
echo "
Setup completed successfully!

To get started:
1. Activate the virtual environment:
   $ACTIVATE_CMD

2. Install dependencies:
   uv pip install -r requirements.txt

3. Update the .env file with your OpenAI API key.

4. Run the RAG system:
   ./rag_cli.py

5. Run tests:
   ./test_rag.py
" 