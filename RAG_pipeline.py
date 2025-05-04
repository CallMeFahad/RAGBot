import langchain
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

class RAGPipeline:
    def __init__(self, api_key, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        # Initialize Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model, model_kwargs={'tokenizer_kwargs': {'clean_up_tokenization_spaces': True}})
        self.vectorstore = None
        # self.messages = [{'role': 'system', 'content': 'You are a helpful teacher.'}]
        self.api_key = api_key

    
    def load_documents(self, file_path, chunk_size = 1000, chunk_overlap = 200):
        #Loading the PDF documents
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size= chunk_size,
            chunk_overlap= chunk_overlap,
            length_function=len,
            is_separator_regex=True,
        )
        self.chunks = text_splitter.split_documents(pages)
        print(f"Document split into {len(self.chunks)} chunks.")

    def create_vectorstore(self):
        if not hasattr(self, 'chunks'):
            raise ValueError("Chunks not loaded. Run load_and_split() first.")
        self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
        print("FAISS Vectorstore created.")
    
    def retrieve(self, query, top_k=3):
        """Retrieve top-k relevant chunks."""
        if self.vectorstore is None:
            raise ValueError("Vectorstore not created. Run create_vectorstore() first.")
        results = self.vectorstore.similarity_search(query, k=top_k)
        return [doc.page_content for doc in results]
    
    def init_llm(self, api_key):
        self.llm = OpenAI(api_key = api_key, base_url = "https://generativelanguage.googleapis.com/v1beta/openai/")
        return self.llm
    
    def generate_answer(self, query, retrieved_docs):
        llm_model = self.init_llm(self.api_key)
        
        messages = [{'role' : 'system', 'content' : 'You are a helpful teacher.'},
                    {'role' : 'user', 'content' : f"""paraphrase the following text :{retrieved_docs},according to the query: {query},only that and no extra text."""}]
        # self.messages.append({'role' : 'user', 'content' : "paraphrase the following text, only that and no extra text: {retrieved_docs}. If provided text doesn't have enough context. Say so clearly."})
        response = llm_model.chat.completions.create(model = 'gemini-2.5-flash-preview-04-17', reasoning_effort="low",
                                        messages = messages)
        return response.choices[0].message.content


    def ask(self, query, top_k=3):
        """Full RAG flow: retrieve + generate."""
        retrieved_docs = self.retrieve(query, top_k=top_k)
        combined_text = "\n\n".join(retrieved_docs)
        answer = self.generate_answer(query, retrieved_docs)
        return answer
