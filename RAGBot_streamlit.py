#Streamlit app for running that RAG model

import os
import streamlit as st
import tempfile
from RAG_pipeline import RAGPipeline

# Get API key from environment (Streamlit secrets inject this)
api_key = os.getenv("OPENAI_API_KEY")

# Initialize RAG pipeline in session state
if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline(api_key=api_key)

st.title("RAGBot")

# File uploader
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None and "vectorstore_created" not in st.session_state:
    with st.spinner("Processing PDF..."):
        # Save uploaded file to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        # Load and process document
        st.session_state.rag.load_documents(file_path=tmp_file_path)
        st.success("PDF processed successfully!")

        st.session_state.rag.create_vectorstore()
        st.success("Vector Store created successfully!")

        st.session_state.vectorstore_created = True

# Ask questions if vector store is ready
if "vectorstore_created" in st.session_state:
    query = st.text_input("Enter your question:")

    if query:
        with st.spinner("Searching answers..."):
            answer = st.session_state.rag.ask(query)
            st.write("Answer:", answer)
