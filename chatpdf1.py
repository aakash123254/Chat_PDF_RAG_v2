import os
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import google.generativeai as genai

# ✅ Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing. Check your .env file.")
    st.stop()

# ✅ Initialize Google Gemini API
genai.configure(api_key=GOOGLE_API_KEY)

# ✅ Initialize Google Generative AI Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# ✅ PDF Processing Function
def process_pdf(pdf_path):
    """Extracts text from a PDF and creates a FAISS vector store."""
    with fitz.open(pdf_path) as doc:
        text = "\n".join(page.get_text("text") for page in doc)

    # ✅ Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.create_documents([text])

    # ✅ Create FAISS vector store
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# ✅ Function to check if query is related to the PDF
def is_query_related_to_pdf(query, vector_store, threshold=0.7):
    """Determines if a query is related to the PDF based on similarity scores."""
    docs_with_scores = vector_store.similarity_search_with_score(query, k=1)
    
    if not docs_with_scores:
        return False  # No matches → Use Gemini Flash
    
    similarity_score = docs_with_scores[0][1]  # FAISS similarity score
    
    # ✅ Lower score = More relevant (FAISS works on distance, not similarity!)
    return similarity_score < threshold  # If close match, return True (use FAISS)

# ✅ Function to get answer from Gemini Flash
def get_gemini_answer(query):
    """Uses Google Gemini Flash to generate an answer for general queries."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(query)
    return response.text if response else "I'm not sure about that."

# ✅ Streamlit UI
st.title("📄 Hybrid PDF + Gemini Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_path = f"./pdfs/{uploaded_file.name}"
    
    # Save uploaded file
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("PDF uploaded successfully!")
    vector_store = process_pdf(pdf_path)
    st.session_state["vector_store"] = vector_store

# ✅ User Query Section
query = st.text_input("Ask a question:")

if query and "vector_store" in st.session_state:
    vector_store = st.session_state["vector_store"]

    if is_query_related_to_pdf(query, vector_store):
        # ✅ Retrieve answer from FAISS
        docs = vector_store.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])

        st.subheader("📄 Answer from PDF:")
        st.write(context)
    else:
        # ✅ Answer using Gemini Flash
        st.subheader("⚡ Answer from Google Gemini Flash:")
        st.write(get_gemini_answer(query))
