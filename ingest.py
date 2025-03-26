import os 
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Import the new embeddings class
from langchain_huggingface import HuggingFaceEmbeddings

# Use FAISS vector store
from langchain.vectorstores import FAISS

# Load documents
loader = DirectoryLoader('data/', show_progress=True, loader_cls=PyPDFLoader)
documents = loader.load()

# Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# Create embeddings using the updated class
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': False}
)

# Initialize the FAISS vector store
vector_store = FAISS.from_documents(
    texts,
    embeddings
)

# Optionally, you can persist the FAISS vector store to disk
vector_store.save_local("stores/pet_faiss")

print("FAISS Vector Store Created.......")
