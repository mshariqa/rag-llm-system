#!/usr/bin/env python3
import os
import sys
import unittest
from unittest.mock import patch
from io import StringIO
from main import RAGSystem

class TestRAGSystem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Check if OpenAI API key is set and documents directory exists"""
        if not os.getenv("OPENAI_API_KEY"):
            print("\nWarning: OPENAI_API_KEY is not set. Skipping integration tests.")
            cls.skip_tests = True
        else:
            cls.skip_tests = False
        
        # Ensure documents directory exists
        if not os.path.exists("./documents") or not os.listdir("./documents"):
            print("\nWarning: No documents found in ./documents. Tests may fail.")
    
    def setUp(self):
        """Set up test environment"""
        if self.skip_tests:
            self.skipTest("OPENAI_API_KEY is not set")
    
    def test_document_loading(self):
        """Test that documents are loaded correctly"""
        rag = RAGSystem()
        docs = rag.load_documents()
        self.assertGreater(len(docs), 0, "No documents were loaded")
        
        # Check that all required document files are loaded
        doc_names = [os.path.basename(doc.metadata["source"]) for doc in docs]
        for expected_file in ["sample.txt", "nlp.txt", "transformers.txt"]:
            self.assertTrue(
                any(expected_file in name for name in doc_names),
                f"Document {expected_file} was not loaded"
            )
    
    def test_simple_query(self):
        """Test a simple query to ensure the RAG system works end-to-end"""
        rag = RAGSystem()
        
        # Test a basic query related to AI
        test_query = "What is artificial intelligence?"
        response = rag.query(test_query)
        
        # Check that response is not empty and has a reasonable length
        self.assertIsNotNone(response, "Response was None")
        self.assertGreater(len(response), 10, "Response was too short")
        
        # Check that response is relevant to the query
        self.assertTrue(
            any(keyword in response.lower() for keyword in ["ai", "artificial", "intelligence"]),
            "Response doesn't seem relevant to the query"
        )
    
    @patch('sys.stdout', new_callable=StringIO)
    def test_initialization_output(self, mock_stdout):
        """Test that the initialization provides appropriate output"""
        rag = RAGSystem()
        output = mock_stdout.getvalue()
        
        # Check initialization output
        self.assertTrue(
            "document" in output.lower() and "indexed" in output.lower(),
            "Initialization output doesn't mention documents being indexed"
        )

def run_tests():
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

if __name__ == "__main__":
    run_tests() 