import streamlit as st
import pdfplumber
from docx import Document
import re

# Page setup
st.set_page_config(page_title="RydenNet Intelligence", layout="wide")
st.markdown(
    """
    <style>
    body {background-color: #f0f0f0; color: #111;}
    .stTextArea textarea {background-color: #fff; color: #000;}
    </style>
    """,
    unsafe_allow_html=True
)
st.title("RydenNet – Behavioural & Intelligence AI")

# Sidebar
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True
    )

    st.header("Search Text/Entities")
    search_query = st.text_input("Enter keyword, email, phone number...")

# Functions
def extract_text(file):
    try:
        if file.type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file)
            return "\n".join(p.text for p in doc.paragraphs)
        elif file.type == "text/plain":
            return file.read().decode("utf-8")
        else:
            return ""
    except Exception as e:
        st.error(f"Failed to extract text from {file.name}: {e}")
        return ""

def find_entities(text):
    emails = re.findall(r"\b[\w.-]+?@\w+?\.\w+?\b", text)
    phones = re.findall(r"\b\d{10,15}\b", text)
    return emails, phones

# Main display
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"File: {file.name}")
        text_content = extract_text(file)
        st.text_area("Extracted Text", text_content, height=300)

        emails, phones = find_entities(text_content)
        st.write("Extracted Emails:", emails or "None")
        st.write("Extracted Phone Numbers:", phones or "None")

        if search_query:
            matches = [line for line in text_content.splitlines() if search_query.lower() in line.lower()]
            st.write(f"Search Results for '{search_query}':", matches or "No matches found")
else:
    st.info("Upload files using the sidebar to extract text and search for entities.")
