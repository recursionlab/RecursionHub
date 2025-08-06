import streamlit as st
import os
import time
import fitz  # PyMuPDF

# Import functions from our knowledge_builder script
# This script now acts as a library of functions
from knowledge_builder import (
    chunk_documents,
    setup_qa_chain,
    VECTORSTORE_DIR
)
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

st.set_page_config(page_title="RecursionHub Knowledge UI", layout="wide")
st.title("📚 RecursionHub: Knowledge Library UI")
st.write("This app allows you to build and query a knowledge library from your PDF documents.")

# --- Function to load the QA chain, cached for performance ---
# The _resource suffix is important for objects that shouldn't be pickled, like DB connections.
@st.cache_resource
def load_qa_chain():
    """
    Loads the QA chain from the persisted vector store.
    Returns None if the vector store doesn't exist.
    """
    # Check if the vector store exists
    if not os.path.exists(VECTORSTORE_DIR) or not os.listdir(VECTORSTORE_DIR):
        st.session_state.db_exists = False
        return None
    
    st.session_state.db_exists = True
    # Use a spinner while loading the model and chain
    with st.spinner("Loading knowledge library... This may take a moment."):
        chain = setup_qa_chain(VECTORSTORE_DIR)
    return chain

# --- Sidebar for controls ---
with st.sidebar:
    st.header("Controls")
    
    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDFs to add them to the knowledge base. This will update the existing library."
    )

    if st.button("Process Uploaded PDFs", disabled=not uploaded_files):
        with st.spinner("Processing uploaded PDFs... This may take a while."):
            try:
                raw_docs = []
                for uploaded_file in uploaded_files:
                    # Read bytes from uploaded file
                    pdf_bytes = uploaded_file.getvalue()
                    # Use fitz to open from bytes and extract text
                    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc_file:
                        text = "".join(page.get_text() for page in doc_file)
                    
                    metadata = {"source": uploaded_file.name}
                    doc = Document(page_content=text, metadata=metadata)
                    raw_docs.append(doc)

                if raw_docs:
                    # Chunk the new documents
                    chunked_docs = chunk_documents(raw_docs)
                    
                    # Get embeddings model (must be same as used before)
                    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

                    # Load existing vector store or create a new one
                    if os.path.exists(VECTORSTORE_DIR) and os.listdir(VECTORSTORE_DIR):
                        st.write("Adding documents to existing knowledge base...")
                        vectorstore = Chroma(persist_directory=VECTORSTORE_DIR, embedding_function=embeddings)
                        vectorstore.add_documents(chunked_docs)
                    else:
                        st.write("Creating new knowledge base from uploaded documents...")
                        Chroma.from_documents(
                            documents=chunked_docs,
                            embedding=embeddings,
                            persist_directory=VECTORSTORE_DIR
                        )
                    
                    st.success("Knowledge base updated successfully!")
                    st.cache_resource.clear()
                    time.sleep(2)
                    st.rerun()
                else:
                    st.warning("Could not process any of the uploaded files.")
            except Exception as e:
                st.error(f"An error occurred during build: {e}")

# --- Main app logic ---

# Load the QA chain
qa_chain = load_qa_chain()

if qa_chain:
    st.header("Ask a Question")
    query = st.text_input("Enter your question about the documents:", key="query_input", placeholder="e.g., What is the main concept in the 'attention' paper?")

    if query:
        with st.spinner("Searching for the answer..."):
            response = qa_chain.invoke({"query": query})
            
            st.subheader("🤖 Answer")
            st.write(response["result"])
            
            st.subheader("📄 Sources")
            st.caption("The answer was generated from the following document snippets:")
            for source in response["source_documents"]:
                with st.expander(f"Source: {source.metadata['source']}"):
                    st.write(source.page_content)
else:
    st.info("The knowledge library is not yet available. Please upload PDF documents and click 'Process Uploaded PDFs' to build it.")