#Streamlit app for running that RAG model

import streamlit as st
from RAG_pipeline import RAGPipeline
import tempfile

rag = RAGPipeline(api_key = '')

st.title("RAGBot")
uploaded_file = st.file_uploader("Upload your PDF file", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_file_path = tmp_file.name

        response = rag.load_documents(file_path=tmp_file_path)
        st.success("PDF processed successfully!")
        rag.create_vectorstore()
        st.success("Vector Store created successfully!")
    prompt = st.chat_input("Say something")

    if prompt:
        with st.spinner("Searching answers:"):
            answer = rag.ask(prompt)
            st.write("Answer:", answer)