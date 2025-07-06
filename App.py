# app.py
import os
import streamlit as st

# PDF loader & splitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings & retriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from pinecone_text.sparse import BM25Encoder
from pinecone import Pinecone, ServerlessSpec
from langchain_community.retrievers import PineconeHybridSearchRetriever

# LLM & chain
from langchain.chains import RetrievalQA

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ------------ Config ------------
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "hybrid-search-langchain-pinecone-db"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PDF_PATH = "Path_to_your_pdf.pdf"

# Validate environment variables
if not all([PINECONE_API_KEY, PINECONE_ENV, GOOGLE_API_KEY]):
    st.error("Missing required environment variables. Please check your .env file.")
    st.stop()

# --------- Pinecone Setup ---------
pc = Pinecone(api_key=PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Correct dimension for Google's embedding-001 model
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
    )
index = pc.Index(INDEX_NAME)

# --------- Load & Split PDF ---------
@st.cache_data
def load_and_split_pdf(pdf_path):
    """Load and split PDF with caching to avoid reprocessing"""
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found at: {pdf_path}")
        st.stop()
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    return chunks

chunks = load_and_split_pdf(PDF_PATH)

# --------- Build Encoders & Index ---------
# Dense embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Sparse (BM25) encoder
bm25 = BM25Encoder().default()
texts = [chunk.page_content for chunk in chunks]
bm25.fit(texts)
bm25.dump("bm25_values.json")
# Note: For subsequent runs, you can load with: bm25 = BM25Encoder().load("bm25_values.json")

# Create hybrid search retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings,
    sparse_encoder=bm25,
    index=index,
    alpha=0.5,  # Balance between dense and sparse retrieval
    top_k=5,
)

# Check if index needs to be populated
index_stats = index.describe_index_stats()
if index_stats['total_vector_count'] == 0:
    with st.spinner("Indexing documents... This may take a moment."):
        retriever.add_texts(
            [c.page_content for c in chunks],
            ids=[str(i) for i in range(len(chunks))],
            metadatas=[{"source": PDF_PATH, "chunk": i} for i in range(len(chunks))],
        )
        st.success("Documents indexed successfully!")

# --------- LLM & QA Chain ----------
# THIS IS THE KEY FIX: Use LangChain's Google AI wrapper instead of direct Google AI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",  # Use the experimental model for better performance
    google_api_key=GOOGLE_API_KEY,
    temperature=0,  # For consistent, factual responses
    convert_system_message_to_human=True  # Helps with compatibility
)

# Create the retrieval-based QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    retriever=retriever, 
    return_source_documents=True,
    chain_type="stuff"  # Explicitly specify chain type
)

# --------- Streamlit UI ------------
st.title("📄 PDF Q&A with Pinecone Hybrid + Gemini")
st.write("Ask questions about your PDF document using advanced hybrid search!")

# Initialize chat history in session state
if "history" not in st.session_state:
    st.session_state.history = []

# Main query input
query = st.text_input("Ask a question about the document:", key="main_query")

if query:
    with st.spinner("Analyzing your question and searching for relevant information..."):
        try:
            result = qa({"query": query})
            
            # Display the answer
            st.write("**Answer:**")
            st.write(result["result"])
            
            # Display source documents
            st.write("---")
            st.write("**Sources:**")
            for i, doc in enumerate(result["source_documents"]):
                with st.expander(f"Source {i+1} - Chunk {doc.metadata['chunk']}"):
                    st.write(doc.page_content)
            
            # Add to history
            st.session_state.history.append((query, result["result"]))
            
        except Exception as e:
            st.error(f"Error processing your question: {e}")
            st.write("Please try rephrasing your question or check if the document was loaded correctly.")

# Sidebar with chat history
st.sidebar.header("Chat History")
if st.session_state.history:
    for i, (q, a) in enumerate(reversed(st.session_state.history)):
        with st.sidebar.expander(f"Q{len(st.session_state.history)-i}: {q[:50]}..."):
            st.write(f"**Q:** {q}")
            st.write(f"**A:** {a}")
    
    # Clear history button
    if st.sidebar.button("Clear History"):
        st.session_state.history = []
        st.rerun()
else:
    st.sidebar.write("No questions asked yet.")

# Display some helpful information
with st.sidebar:
    st.write("---")
    st.write("**About this app:**")
    st.write("This app uses hybrid search combining:")
    st.write("• Dense vectors (semantic similarity)")
    st.write("• Sparse vectors (keyword matching)")
    st.write("• Google's Gemini for answers")
    
    if chunks:
        st.write(f"**Document info:**")
        st.write(f"• {len(chunks)} text chunks")
        st.write(f"• Ready for questions!")
