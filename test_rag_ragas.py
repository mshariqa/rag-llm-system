from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)
from datasets import Dataset
from main import RAGSystem
import json
from typing import List, Dict

def create_test_dataset() -> Dataset:
    """Create a test dataset with questions and ground truths"""
    # Test data with questions, contexts, and expected answers
    test_data = {
        "question": [
            "What is machine learning?",
            "Explain deep learning.",
            "What are transformers in NLP?",
            "What is the difference between supervised and unsupervised learning?",
            "How do neural networks work?"
        ],
        "ground_truth": [
            "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.",
            "Deep learning is a type of machine learning using neural networks with multiple layers.",
            "Transformers are neural network architectures that use self-attention mechanisms for processing sequential data.",
            "Supervised learning uses labeled data for training, while unsupervised learning finds patterns in unlabeled data.",
            "Neural networks are computing systems inspired by biological neural networks that learn from examples."
        ],
        "contexts": [],  # Will be filled by RAG system
        "answer": []    # Will be filled by RAG system
    }
    
    # Initialize RAG system
    rag = RAGSystem()
    
    # Get contexts and answers from RAG system
    for question in test_data["question"]:
        # Get answer
        answer = rag.query(question)
        test_data["answer"].append(answer)
        
        # Get context used for this question
        retrieved_docs = rag.vectorstore.similarity_search(question, k=3)
        contexts = [doc.page_content for doc in retrieved_docs]
        test_data["contexts"].append(contexts)
    
    return Dataset.from_dict(test_data)

def evaluate_rag_system():
    """Evaluate RAG system using RAGAS metrics"""
    print("Starting RAG system evaluation...")
    
    try:
        # Create evaluation dataset
        dataset = create_test_dataset()
        
        # Define metrics
        metrics = [
            faithfulness,           # Measures how faithful the answer is to the retrieved context
            answer_relevancy,       # Measures if the answer is relevant to the question
            context_precision,      # Measures the precision of retrieved context
            context_recall         # Measures if important information from ground truth is in context
        ]
        
        # Run evaluation
        results = evaluate(
            dataset=dataset,
            metrics=metrics
        )
        
        # Access results through the scores dictionary
        serializable_results = {
            "Faithfulness": float(results.scores['faithfulness']),
            "Answer Relevancy": float(results.scores['answer_relevancy']),
            "Context Precision": float(results.scores['context_precision']),
            "Context Recall": float(results.scores['context_recall'])
        }
        
        # Save detailed results
        with open("ragas_evaluation_results.json", "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        # Print results summary
        print("\nRAGAS Evaluation Results:")
        print("========================")
        for metric_name, score in serializable_results.items():
            print(f"{metric_name}: {score:.3f}")
        
        return serializable_results
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        print("\nDetailed error information:")
        if 'results' in locals():
            print(f"Results scores: {results.scores}")
            print(f"Available attributes: {dir(results)}")
        raise

if __name__ == "__main__":
    evaluate_rag_system()