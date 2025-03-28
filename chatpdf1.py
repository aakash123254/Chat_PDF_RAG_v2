import os
import streamlit as st
import fitz  # PyMuPDF for PDF handling
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import google.generativeai as genai
import time

# ✅ Load API key from .env
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY is missing or invalid. Please check your .env file.")
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
    try:
        with fitz.open(pdf_path) as doc:
            text = "\n".join(page.get_text("text") for page in doc)

        # ✅ Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.create_documents([text])

        # ✅ Create FAISS vector store
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# ✅ Function to check if query is related to the PDF
def is_query_related_to_pdf(query, vector_store, threshold=0.7):
    """Determines if a query is related to the PDF based on similarity scores."""
    try:
        docs_with_scores = vector_store.similarity_search_with_score(query, k=1)
        
        if not docs_with_scores:
            return False  # No matches → Use Gemini Flash
        
        similarity_score = docs_with_scores[0][1]  # FAISS similarity score
        
        # ✅ Lower score = More relevant (FAISS works on distance, not similarity!)
        return similarity_score < threshold  # If close match, return True (use FAISS)
    except Exception as e:
        st.error(f"Error checking query relevance: {str(e)}")
        return False

# ✅ Function to get answer from Gemini Flash
def get_gemini_answer(query):
    """Uses Google Gemini Flash to generate an answer for general queries."""
    model = genai.GenerativeModel("gemini-1.5-flash")
    full_query = f"""
    Question:
    {query}
    """
    try:
        response = model.generate_content(full_query)
        if response and response.text.strip():
            return response.text.strip()
        else:
            return "I couldn't find an answer to that. Please try rephrasing your question."
    except Exception as e:
        # Retry after a short delay
        time.sleep(2)
        try:
            response = model.generate_content(full_query)
            return response.text.strip() if response else "I'm not sure about that."
        except Exception as retry_error:
            return f"Error after retry: {str(retry_error)}"

# ✅ Custom CSS for Styling
css = """
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
}
.chat-message.user {
    justify-content: flex-end;
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 20%;
}
.chat-message .avatar img {
    max-width: 78px;
    max-height: 78px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #fff;
}
</style>
"""

# ✅ HTML Templates for Messages
bot_template = """
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/6134/6134346.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
"""

user_template = """
<div class="chat-message user">
    <div class="message" style="text-align:right">{{MSG}}</div>
    <div class="avatar">
        <img src="https://png.pngtree.com/png-vector/20190321/ourmid/pngtree-vector-users-icon-png-image_856952.jpg" alt="User Avatar">
    </div>    
</div>
"""

# ✅ Streamlit UI
st.markdown(css, unsafe_allow_html=True)  # Apply custom CSS
st.title("📄 Hybrid PDF + Gemini Chatbot")

# Initialize session state for conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for PDF Upload
with st.sidebar:
    st.subheader("Upload Your PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        pdf_path = f"./pdfs/{uploaded_file.name}"
        
        # Ensure the ./pdfs directory exists
        os.makedirs("./pdfs", exist_ok=True)

        # Save uploaded file
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Processing PDF..."):
            vector_store = process_pdf(pdf_path)
            if vector_store:
                st.session_state["vector_store"] = vector_store
                st.success("PDF uploaded and processed successfully!")
            else:
                st.error("Failed to process the PDF.")

# Display Chat History
for message in st.session_state.chat_history:
    st.markdown(message, unsafe_allow_html=True)

# Input Box for User Query
query = st.text_input("Ask a question:", key="query_input", placeholder="Type your question here...", value="")

# Button to Submit Query
submit_button = st.button("Submit")

if submit_button and query:
    if "vector_store" in st.session_state:
        vector_store = st.session_state["vector_store"]

        # Generate Response
        if is_query_related_to_pdf(query, vector_store):
            # Retrieve answer from FAISS
            docs = vector_store.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            response = f"📄 Answer from PDF:\n{context}"
        else:
            # Get answer from Gemini Flash for unrelated queries
            response = get_gemini_answer(query)
    else:
        # If no PDF is uploaded, use Gemini Flash directly
        response = get_gemini_answer(query)

    # Update conversation history
    st.session_state.chat_history.append(user_template.replace("{{MSG}}", query))
    st.session_state.chat_history.append(bot_template.replace("{{MSG}}", response))

    # Clear the input box
    query = ""  # Reset the input field

    # Force Refresh to Update Chat History
    st.rerun()  # Refresh the app to reflect changes