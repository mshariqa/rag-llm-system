import os
import glob
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

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
        
        # Use the new message history approach
        self.message_history = ChatMessageHistory()
        
        self.vectorstore = None
        self.rag_chain = None
        
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

        # Define system prompt template
        system_template = """You are a helpful assistant that answers questions based on the provided context.
Use the following context to answer the question. If you don't know the answer, say that you don't know.
Keep your answers concise and to the point.

Context: {context}
"""

        # Create the full RAG chain using the Model Context Protocol pattern
        # 1. Create the context builder (retriever + formatter)
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        # 2. Define how to construct a prompt given the message history and context
        def build_prompt(input_dict):
            question = input_dict["question"]
            context = input_dict["context"]
            chat_history = input_dict["chat_history"]
            
            messages = [SystemMessage(content=system_template.format(context=context))]
            
            # Format chat history
            for msg, res in chat_history:
                messages.append(HumanMessage(content=msg))
                messages.append(AIMessage(content=res))
                
            # Add the current question
            messages.append(HumanMessage(content=question))
            
            return messages
        
        # 3. Create the RAG chain
        retriever_chain = RunnableParallel(
            {"context": retriever | format_docs, 
             "question": RunnablePassthrough(), 
             "chat_history": lambda x: self._get_chat_history()}
        )
        
        self.rag_chain = (
            retriever_chain 
            | build_prompt
            | self.llm
            | StrOutputParser()
        )
        
        print(f"Indexed {len(split_docs)} document chunks.")
    
    def _get_chat_history(self):
        """Extract chat history in the format (human_message, ai_message)"""
        chat_history = []
        messages = self.message_history.messages
        
        # Process history if we have previous exchanges
        for i in range(0, len(messages)-1, 2):
            if i+1 < len(messages):
                if isinstance(messages[i], HumanMessage) and isinstance(messages[i+1], AIMessage):
                    chat_history.append((messages[i].content, messages[i+1].content))
        
        return chat_history
    
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
        if not self.rag_chain:
            raise ValueError("RAG system is not initialized")
            
        # Invoke the chain with the question
        response = self.rag_chain.invoke(question)
        
        # Update message history
        self.message_history.add_user_message(question)
        self.message_history.add_ai_message(response)
        
        return response

def main():
    # Initialize the RAG system
    rag_system = RAGSystem()
    
    if not rag_system.rag_chain:
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