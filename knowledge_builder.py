import os
import glob
from typing import List, Dict, Any

# For PDF Parsing
import fitz  # PyMuPDF

# For RAG pipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings # Placeholder, user can swap
from langchain_community.llms import Ollama # Placeholder
from langchain.chains import RetrievalQA

# --- Configuration ---
# Adjust these paths and settings as needed
PDF_SOURCE_DIR = "./pdfs"  # Create a 'pdfs' directory and place your PDFs there
VECTORSTORE_DIR = "./vectorstore"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- 1. PDF Ingestion and Parsing ---

def parse_pdfs(pdf_dir: str) -> List[Document]:
    """
    Parses all PDF files in a directory, extracts text, and returns a list of LangChain Documents.
    Each Document represents a PDF and contains its text content and metadata.
    """
    print(f"Scanning for PDFs in '{pdf_dir}'...")
    pdf_files = glob.glob(os.path.join(pdf_dir, "*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in '{pdf_dir}'. Please add your PDFs there.")
        return []

    all_docs = []
    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}")
        try:
            with fitz.open(pdf_path) as doc_file:
                text = ""
                for page in doc_file:
                    text += page.get_text()
                
                # Create a LangChain Document
                # Metadata helps in tracking the source of information
                metadata = {"source": os.path.basename(pdf_path)}
                doc = Document(page_content=text, metadata=metadata)
                all_docs.append(doc)
        except Exception as e:
            print(f"Error processing {pdf_path}: {e}")
            
    print(f"Successfully parsed {len(all_docs)} PDF(s).")
    return all_docs

# --- 2. Text Chunking ---

def chunk_documents(docs: List[Document]) -> List[Document]:
    """
    Splits the loaded Documents into smaller chunks for effective embedding and retrieval.
    """
    print("Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunked_docs = text_splitter.split_documents(docs)
    print(f"Split {len(docs)} documents into {len(chunked_docs)} chunks.")
    return chunked_docs

# --- 3. Embedding and Indexing ---

def create_and_persist_vectorstore(chunked_docs: List[Document], persist_dir: str):
    """
    Creates embeddings for document chunks and stores them in a Chroma vector database.
    """
    if not chunked_docs:
        print("No documents to process. Skipping vector store creation.")
        return None

    print("Initializing embedding model...")
    # --- IMPORTANT: CHOOSE YOUR EMBEDDING MODEL ---
    # You can use OpenAI, HuggingFace, Google, or local models via Ollama.
    # Example for Ollama (run `ollama run mxbai-embed-large` first):
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    # Example for OpenAI (requires OPENAI_API_KEY env var):
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings()

    print(f"Creating and persisting vector store at '{persist_dir}'...")
    # This will create the vector store and save it to disk.
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    print("Vector store created successfully.")
    return vectorstore

# --- 4. Retrieval-Augmented Generation (RAG) ---

def setup_qa_chain(persist_dir: str):
    """
    Loads the persisted vector store and sets up a QA chain for querying.
    """
    if not os.path.exists(persist_dir):
        print(f"Vector store not found at '{persist_dir}'. Please run the ingestion process first.")
        return None

    print("Loading vector store and setting up QA chain...")
    # --- IMPORTANT: CHOOSE YOUR EMBEDDING AND LLM MODELS ---
    # The embedding model MUST be the same one used during indexing.
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    # from langchain_openai import OpenAIEmbeddings
    # embeddings = OpenAIEmbeddings()

    # Load the vector store from disk
    vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    # Initialize the LLM
    # Example for Ollama (run `ollama run llama3` first):
    llm = Ollama(model="llama3")
    # Example for OpenAI:
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # "stuff" is simple; others are "map_reduce", "refine"
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}), # Retrieve top 3 chunks
        return_source_documents=True,
    )
    print("QA chain is ready.")
    return qa_chain