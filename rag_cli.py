#!/usr/bin/env python3
import os
import sys
import shutil
import argparse
from main import RAGSystem

def setup_argparse():
    parser = argparse.ArgumentParser(
        description="RAG LLM System CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the RAG system in interactive mode
  python rag_cli.py

  # Add a document to the system
  python rag_cli.py --add path/to/document.txt

  # Add multiple documents to the system
  python rag_cli.py --add path/to/document1.pdf path/to/document2.txt

  # List all documents in the system
  python rag_cli.py --list

  # Remove a document from the system
  python rag_cli.py --remove document.txt
"""
    )

    parser.add_argument(
        "--add", "-a",
        nargs="+",
        help="Add document(s) to the RAG system"
    )
    
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all documents in the RAG system"
    )
    
    parser.add_argument(
        "--remove", "-r",
        nargs="+",
        help="Remove document(s) from the RAG system"
    )
    
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear the vector database (forces reindexing)"
    )
    
    return parser.parse_args()

def add_documents(file_paths):
    document_dir = "./documents"
    os.makedirs(document_dir, exist_ok=True)
    
    added_count = 0
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File {file_path} does not exist")
            continue
            
        file_name = os.path.basename(file_path)
        dest_path = os.path.join(document_dir, file_name)
        
        # Check if file extension is supported
        _, ext = os.path.splitext(file_name)
        if ext.lower() not in ['.txt', '.pdf']:
            print(f"Error: Unsupported file format {ext}. Only .txt and .pdf files are supported.")
            continue
        
        try:
            shutil.copy2(file_path, dest_path)
            print(f"Added document: {file_name}")
            added_count += 1
        except Exception as e:
            print(f"Error adding {file_name}: {e}")
    
    if added_count > 0:
        print(f"Successfully added {added_count} document(s)")
        # Clear the vector database to force reindexing
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("Cleared vector database for reindexing")

def list_documents():
    document_dir = "./documents"
    if not os.path.exists(document_dir):
        print("No documents directory found")
        return
    
    files = os.listdir(document_dir)
    if not files:
        print("No documents found")
        return
    
    print("Documents in the RAG system:")
    for i, file_name in enumerate(sorted(files), 1):
        file_path = os.path.join(document_dir, file_name)
        size = os.path.getsize(file_path)
        print(f"{i}. {file_name} ({size/1024:.1f} KB)")

def remove_documents(file_names):
    document_dir = "./documents"
    if not os.path.exists(document_dir):
        print("No documents directory found")
        return
    
    removed_count = 0
    for file_name in file_names:
        file_path = os.path.join(document_dir, file_name)
        if not os.path.exists(file_path):
            # Check if the user provided a partial name
            matching_files = [f for f in os.listdir(document_dir) if file_name in f]
            if len(matching_files) == 1:
                file_path = os.path.join(document_dir, matching_files[0])
                file_name = matching_files[0]
            elif len(matching_files) > 1:
                print(f"Multiple files match '{file_name}'. Please be more specific:")
                for match in matching_files:
                    print(f"  - {match}")
                continue
            else:
                print(f"Error: Document {file_name} not found")
                continue
        
        try:
            os.remove(file_path)
            print(f"Removed document: {file_name}")
            removed_count += 1
        except Exception as e:
            print(f"Error removing {file_name}: {e}")
    
    if removed_count > 0:
        print(f"Successfully removed {removed_count} document(s)")
        # Clear the vector database to force reindexing
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            print("Cleared vector database for reindexing")

def clear_vector_db():
    if os.path.exists("./chroma_db"):
        shutil.rmtree("./chroma_db")
        print("Vector database cleared. Documents will be reindexed on next run.")
    else:
        print("No vector database found.")

def main():
    args = setup_argparse()
    
    # Handle command line arguments
    if args.add:
        add_documents(args.add)
        return
    
    if args.list:
        list_documents()
        return
    
    if args.remove:
        remove_documents(args.remove)
        return
    
    if args.clear_db:
        clear_vector_db()
        return
    
    # If no arguments are provided, run the RAG system
    try:
        # Initialize the RAG system
        rag_system = RAGSystem()
        
        if not rag_system.qa_chain:
            print("Failed to initialize RAG system. Exiting.")
            return
        
        print("\nRAG System initialized! Type 'exit' to quit.\n")
        
        # Interactive query loop
        while True:
            question = input("\nEnter your question: ")
            
            if question.lower() == 'exit':
                break
            
            try:
                answer = rag_system.query(question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error: {e}")
    
    except KeyboardInterrupt:
        print("\nExiting RAG system...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 