import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Embedding technique
from langchain.vectorstores import FAISS  # Vector Storage
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Google Generative AI with API Key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_path):
    """Extracts text from a single PDF file."""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits text into manageable chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_faiss_vector_store(pdf_path, faiss_index_path):
    """Creates a FAISS vector store from the PDF text and saves it locally."""
    # Extract text from the PDF
    raw_text = get_pdf_text(pdf_path)

    # Split text into chunks
    text_chunks = get_text_chunks(raw_text)

    # Initialize Google Generative AI embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)

    # Save FAISS index locally
    vector_store.save_local(faiss_index_path)
    print(f"FAISS vector store saved to {faiss_index_path}")

if __name__ == "__main__":
    # Replace 'your_pdf_file.pdf' with the path to your PDF file
    pdf_file = "E://10K reader SAP\JSS Published Paper.pdf"
    faiss_index_file = "faiss_index"

    # Create FAISS vector store from the PDF
    create_faiss_vector_store(pdf_file, faiss_index_file)
