import os
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Check if API key is available
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable is not set")

class RAGSystem:
    def __init__(self, document_dir="./documents"):
        self.document_dir = document_dir
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo")
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.vectorstore = None
        self.qa_chain = None
        
        # Initialize the system
        self.initialize()
    
    def initialize(self):
        """Initialize the RAG system by loading and indexing documents"""
        # Load documents
        documents = self.load_documents()
        
        if not documents:
            print("No documents found. Please add documents to the documents directory.")
            return
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        split_docs = text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=self.embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create retriever
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )
        
        # Set up the QA chain
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory
        )
        
        print(f"Indexed {len(split_docs)} document chunks.")
    
    def load_documents(self):
        """Load documents from the document directory"""
        documents = []
        
        # Load text files
        for file_path in glob.glob(os.path.join(self.document_dir, "*.txt")):
            loader = TextLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded text document: {file_path}")
        
        # Load PDF files
        for file_path in glob.glob(os.path.join(self.document_dir, "*.pdf")):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded PDF document: {file_path}")
        
        return documents
    
    def query(self, question):
        """Process a user query and return the response"""
        if not self.qa_chain:
            raise ValueError("RAG system is not initialized")
        
        response = self.qa_chain.invoke({"question": question})
        return response["answer"]

def main():
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

if __name__ == "__main__":
    main() 