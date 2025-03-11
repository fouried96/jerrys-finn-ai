import faiss
import ollama
import pdfplumber
import os
import glob
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document

# Load PDFs from data folder
def load_pdfs(data_path):
    text = ""
    pdf_files = glob.glob(os.path.join(data_path, "*.pdf"))
    
    if not pdf_files:
        raise ValueError("No PDF files found in the specified data directory.")

    print(f"Found {len(pdf_files)} PDFs.")  # Debugging print

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file}...")  # Debugging print
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                try:
                    extracted_text = page.extract_text()
                    if extracted_text is None:
                        print(f"Warning: No extractable text found in {pdf_file}. Skipping...")
                        continue  # Skip this PDF
                except Exception as e:
                    print(f"Error processing {pdf_file}: {e}")
                    continue  # Skip to the next PDF
			
                if extracted_text:
                    text += extracted_text + "\n"
                else:
                    print(f"Warning: Page {page.page_number} has no extractable text.")  # Debugging print
    
    print(f"Total extracted text length: {len(text)} characters")  # Debugging print
    return text

# Split text into smaller chunks for embedding
def split_text(text, chunk_size=1000, overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return text_splitter.split_text(text)

# Create FAISS index
def create_faiss_index(texts):
    if not texts:
        raise ValueError("No text chunks available for indexing.")
    
    print(f"Number of chunks to index: {len(texts)}")  

    # Define FAISS index path
    faiss_path = "faiss_index"

    # Check if faiss_index exists and remove it
    if os.path.exists(faiss_path):
        if os.path.isdir(faiss_path):
            shutil.rmtree(faiss_path)  # Remove folder
            print("Old FAISS index directory deleted.")
        else:
            os.remove(faiss_path)  # Remove file
            print("Old FAISS index file deleted.")

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Convert texts into Document objects
    documents = [Document(page_content=text) for text in texts]

    # Create FAISS index with proper docstore
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save the new FAISS index
    vectorstore.save_local(faiss_path)
    print("New FAISS index created successfully.")

# Query FAISS and Generate Response
def query_faiss(query):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load FAISS index with safe deserialization
    vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    results = vectorstore.similarity_search(query, k=3)  # Get top 3 relevant chunks
    retrieved_text = " ".join([doc.page_content for doc in results])

    response = ollama.chat(model="mistral", messages=[
        {"role": "system", "content": "You are an expert in pop punk mixing and Waves plugins."},
        {"role": "user", "content": f"Here are some reference materials: {retrieved_text}. Answer this: {query}"}
    ])

    return response['message']['content']

# Main execution
if __name__ == "__main__":
    data_path = "data/"
    print("Loading PDFs...")
    pdf_text = load_pdfs(data_path)

    print("Splitting text...")
    chunks = split_text(pdf_text)

    print("Creating FAISS index...")
    create_faiss_index(chunks)

    print("Setup complete. You can now query the RAG system!")