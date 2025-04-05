# RAG LLM System

This is a simple Retrieval Augmented Generation (RAG) system that uses LangChain and OpenAI to answer questions based on local documents.

## Setup

### Automatic Setup (Recommended)

Run the setup script:
```
./setup.sh
```

This will:
1. Check for required dependencies
2. Create a virtual environment
3. Create a template .env file
4. Make the CLI scripts executable

After running the setup script, follow the instructions to:
1. Activate the virtual environment
2. Install dependencies
3. Update your OpenAI API key

### Manual Setup

1. Create a virtual environment:
   ```
   uv venv
   ```

2. Activate the virtual environment:
   ```
   source .venv/bin/activate.fish  # For fish shell
   # OR
   source .venv/bin/activate  # For bash/zsh
   ```

3. Install dependencies:
   ```
   uv pip install -r requirements.txt
   ```

4. Set up your OpenAI API key in the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Adding Documents

Place your documents in the `documents` directory. The system supports the following file formats:
- `.txt`: Plain text documents
- `.pdf`: PDF documents

Sample documents are provided:
- `sample.txt`: Basic information about AI
- `nlp.txt`: Information about Natural Language Processing
- `transformers.txt`: Information about transformer models

You can add more documents as needed.

## Running the System

You can run the system in two ways:

### Basic Usage

```
python main.py
```

### CLI Tool (Recommended)

The system includes a command-line interface (CLI) tool that provides additional functionality:

```
python rag_cli.py
```

#### CLI Commands

- **Interactive Mode**: Simply run without arguments
  ```
  python rag_cli.py
  ```

- **Add Documents**:
  ```
  python rag_cli.py --add path/to/document.txt
  python rag_cli.py --add path/to/doc1.pdf path/to/doc2.txt
  ```

- **List Documents**:
  ```
  python rag_cli.py --list
  ```

- **Remove Documents**:
  ```
  python rag_cli.py --remove document.txt
  ```

- **Clear Vector Database**:
  ```
  python rag_cli.py --clear-db
  ```

## Testing

To run tests and verify the system is working correctly:
```
python test_rag.py
```

## How It Works

1. **Document Loading**: The system loads documents from the `documents` directory.
2. **Text Splitting**: Documents are split into smaller chunks for better retrieval.
3. **Embedding**: Document chunks are converted to vector embeddings using OpenAI's embedding model.
4. **Storage**: Embeddings are stored in a Chroma vector database.
5. **Retrieval**: When you ask a question, the system finds the most relevant document chunks.
6. **Generation**: OpenAI's LLM generates an answer based on the retrieved context and your question.

## Requirements

- Python 3.7+
- OpenAI API key 