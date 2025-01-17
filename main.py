import os
import glob
import fitz  # PyMuPDF
from sbert_rag import SBertRAG
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_documents_from_directory(directory: str):
    logging.info(f"Loading documents from directory: {directory}")
    documents = []
    metadata = []
    
    for filepath in glob.glob(os.path.join(directory, "*.pdf")):
        logging.info(f"Processing file: {filepath}")
        with fitz.open(filepath) as pdf_document:
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text += page.get_text()
            documents.append(text)
            metadata.append({"filename": os.path.basename(filepath)})
    
    logging.info(f"Loaded {len(documents)} documents")
    return documents, metadata

def generate_answer(documents, query):
    logging.info("Generating answer")
    model_name = "deepset/roberta-base-squad2"
    qa_pipeline = pipeline("question-answering", model=model_name, revision="main")
    
    # Combine the text from all documents
    combined_text = " ".join([doc.text for doc in documents])
    
    # Use the QA model to find the answer to the query
    answer = qa_pipeline(question=query, context=combined_text)
    
    logging.info("Answer generated successfully")
    return answer['answer']

# Example usage
if __name__ == "__main__":
    # Initialize RAG system
    logging.info("Initializing RAG system")
    rag = SBertRAG()
    
    # Load documents and metadata from the filesystem
    directory = "./test_docs"  # Directory containing PDF documents
    documents, metadata = load_documents_from_directory(directory)
    
    # Add documents to the RAG system
    logging.info("Adding documents to the RAG system")
    rag.add_documents(documents, metadata)
    
    # Retrieve documents for a sample query
    # query = "How did the wildfires in California start?"
    query = "Where in California are the fires?"
    logging.info(f"Retrieving documents for query: {query}")
    results = rag.retrieve(query)
    
    # Generate a natural language answer
    logging.info("Generating a natural language answer")
    answer = generate_answer([doc for doc, _ in results], query)
    print(f"Answer: {answer}")