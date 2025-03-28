import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
import google.generativeai as genai
import os
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from typing import Any, Dict, List

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=gemini_api_key)

# Custom template to rephrase follow-up questions
custom_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# Extract text from PDFs
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    if not text.strip():
        st.error("Uploaded PDFs are empty or could not be read.")
        return None
    print("Extracted Text:", text)  # Debug
    return text

# Split text into chunks
def get_chunks(raw_text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(raw_text)
    print("Text Chunks:", chunks)  # Debug
    return chunks

# Create vector store using FAISS and HuggingFace embeddings
def get_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    print("Vectorstore Created:", vectorstore)  # Debug
    return vectorstore

# Custom LLM wrapper for Gemini
class GeminiLLM(BaseChatModel):
    def _generate(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        # Convert messages to a single prompt string
        prompt = "\n".join([msg.content for msg in messages])

        # Use Gemini API to generate a response
        model = genai.GenerativeModel('models/gemini-1.5-flash')  # Updated to use gemini-1.5-flash
        response = model.generate_content(prompt)

        # Debugging: Print the response object to inspect its structure
        print("Response Object:", response)

        # Ensure we return the generated text
        return response.text  # Access the 'text' attribute of the response

    @property
    def _llm_type(self) -> str:
        return "gemini"

# Create conversation chain
def get_conversationchain(vectorstore):
    llm = GeminiLLM()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    print("LLM:", llm)  # Debug
    print("Vectorstore:", vectorstore)  # Debug

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    print("Conversation Chain Created:", conversation_chain)  # Debug
    return conversation_chain

# Handle user questions
def handle_question(question):
    if st.session_state.conversation is None:
        st.error("Please upload and process documents first.")
        return

    try:
        response = st.session_state.conversation.invoke({'question': question})  # Updated to use invoke
        st.session_state.chat_history = response["chat_history"]

        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Main function
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs 📚")
    question = st.text_input("Ask a question about your document:")
    if question:
        handle_question(question)

    with st.sidebar:
        st.subheader("Your documents")
        docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(docs)
                if raw_text is None:
                    return

                text_chunks = get_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversationchain(vectorstore)
                st.success("Documents processed successfully!")

if __name__ == '__main__':
    main()