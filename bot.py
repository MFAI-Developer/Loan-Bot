import os
import re
import pickle
import glob
import fitz  # PyMuPDF for handling PDFs
import faiss
import pandas as pd
import streamlit as st
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import Document  # Corrected import for Document class
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


# Ensure your OpenAI API key is securely stored in an environment variable
openai_api_key = os.getenv("OpenAI API key")

# Set the path to your data and PDF files
pdf_directory_path = '/home/mfai-developer/PycharmProjects/chatbot/data/pdf'
xlsx_file_path = '/home/mfai-developer/PycharmProjects/chatbot/data/FHA 2024 Loan Limits.xlsx'

# Load and Save Conversations
conversations_file = "conversations.pkl"


@st.cache_data()
def load_conversations():
    try:
        with open(conversations_file, "rb") as f:
            return pickle.load(f)
    except (FileNotFoundError, EOFError):
        return []


def save_conversations(conversations):
    with open(conversations_file, "wb") as f:
        pickle.dump(conversations, f)


def load_pdfs_from_directory(directory_path):
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    return pdf_files


def load_excel_file(xlsx_path):
    try:
        df = pd.read_excel(xlsx_path)
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None


# Custom function to process PDF and include page numbers
def process_pdf_with_page_numbers(pdf_path):
    reader = fitz.open(pdf_path)  # Use PyMuPDF to open the PDF
    documents = []
    text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=100)

    for page_number in range(len(reader)):
        page = reader.load_page(page_number)
        text = page.get_text("text")
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "pdf_file": pdf_path,
                    "page_number": page_number + 1
                }
            )
            documents.append(doc)

    return documents


# Function to build FAISS index with references
def build_faiss_index(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore


# Function to query FAISS index using GPT-4
def query_faiss(query, vectorstore):
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4", temperature=0.0, openai_api_key=openai_api_key)

    qa_chain = load_qa_chain(llm, chain_type="stuff")
    result = qa_chain({"input_documents": retriever.get_relevant_documents(query), "question": query})
    return result


# Function to extract the page number and display the relevant page from the PDF
def extract_and_display_pdf_page(documents, pdf_file):
    for doc in documents:
        if doc.metadata["pdf_file"] == pdf_file:
            page_number = doc.metadata.get("page_number", None)
            if page_number:
                display_pdf_page(pdf_file, page_number)
                return  # Exit after displaying the relevant page

    st.warning("No specific page number found in the metadata.")


# Function to display the relevant PDF page
def display_pdf_page(pdf_path, page_num):
    try:
        images = convert_from_path(pdf_path, first_page=page_num, last_page=page_num)
        for image in images:
            st.image(image=image)
    except Exception as e:
        st.error(f"Error displaying PDF page: {e}")


# Function to truncate conversation history
def truncate_conversation(conversation, max_tokens=4000):
    """Keep the conversation history within the token limit."""
    tokens = 0
    truncated_conversation = []

    for message in reversed(conversation):
        message_tokens = len(message['content'].split())  # Rough token estimate
        if tokens + message_tokens > max_tokens:
            break
        truncated_conversation.insert(0, message)
        tokens += message_tokens

    return truncated_conversation


# Initialize session state variables if they don't exist
if 'pdf_files' not in st.session_state:
    with st.spinner("Loading PDF files..."):
        st.session_state.pdf_files = load_pdfs_from_directory(pdf_directory_path)
        if not st.session_state.pdf_files:
            st.error("No PDF files found in the specified directory.")

if 'xlsx_data' not in st.session_state:
    with st.spinner("Loading Excel file..."):
        st.session_state.xlsx_data = load_excel_file(xlsx_file_path)
        if st.session_state.xlsx_data is None:
            st.error("Failed to load Excel data.")

if 'vectorstore' not in st.session_state:
    with st.spinner("Building FAISS index..."):
        documents = []
        for pdf in st.session_state.pdf_files:
            documents.extend(process_pdf_with_page_numbers(pdf))
        st.session_state.vectorstore = build_faiss_index(documents)

if 'conversations' not in st.session_state:
    st.session_state.conversations = load_conversations()

if 'current_conversation' not in st.session_state:
    st.session_state.current_conversation = [
        {"role": "assistant", "content": "Welcome! Please type your query or ask for help."}]


def display_chats_sidebar():
    with st.sidebar:
        st.header('Settings')
        if st.button('Start New Conversation'):
            with st.spinner("Initializing new conversation..."):
                st.session_state.current_conversation = [
                    {"role": "assistant", "content": "Welcome! Please type your query."}]
                st.session_state.conversations.append(st.session_state.current_conversation)

        if st.button('Clear All Conversations'):
            st.session_state.conversations = []
            st.session_state.current_conversation = []

        st.header('Past Conversations')
        for idx, conversation in enumerate(st.session_state.conversations):
            if conversation:
                chat_title_raw = next((msg["content"] for msg in conversation if msg["role"] == "user"),
                                      "New Conversation")
                chat_title = chat_title_raw[:30] + "..." if len(chat_title_raw) > 30 else chat_title_raw
                if st.button(f"Conversation {idx + 1}: {chat_title}"):
                    st.session_state.current_conversation = conversation


# Main app function for displaying the chat interface
def main_app():
    for message in st.session_state.current_conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    query = st.chat_input("Send a query:")
    if query:
        # Handle simple greetings separately
        if query.lower() in ["hi", "hello", "hey"]:
            st.session_state.current_conversation.append(
                {"role": "assistant", "content": "Hello! How can I assist you today?"})
            st.chat_message("assistant").write("Hello! How can I assist you today?")
            return

        # Truncate conversation history to prevent exceeding token limit
        st.session_state.current_conversation = truncate_conversation(st.session_state.current_conversation)

        st.session_state.current_conversation.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = query_faiss(query, st.session_state.vectorstore)
                    answer = result['output_text']
                    st.write(answer)
                    st.session_state.current_conversation.append({"role": "assistant", "content": answer})
                    save_conversations(st.session_state.conversations)

                    # Display the PDF page from where the answer was retrieved
                    if result and result['input_documents']:
                        extract_and_display_pdf_page(result['input_documents'], st.session_state.pdf_files[0])

                except Exception as e:
                    st.error(f"Error getting Bot Response: {e}")


display_chats_sidebar()
main_app()
