import streamlit as st
import pdfplumber
from docx import Document
import re

st.set_page_config(page_title="RydenNet Intelligence", layout="wide")
st.title("RydenNet – Behavioural & Intelligence AI")

# Sidebar Upload
with st.sidebar:
    st.header("Upload Files")
    uploaded_files = st.file_uploader(
        "Choose PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    st.header("Search Text/Entities")
    search_query = st.text_input("Enter keyword, email, phone number...")

# Helper functions
def extract_text(file):
    if file.type == "application/pdf":
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return text
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        text = "\n".join(p.text for p in doc.paragraphs)
        return text
    elif file.type == "text/plain":
        return file.read().decode("utf-8")
    else:
        return ""

def find_entities(text):
    emails = re.findall(r"\b[\w.-]+?@\w+?\.\w+?\b", text)
    phones = re.findall(r"\b\d{10,15}\b", text)
    return emails, phones

# Display uploaded files and extracted text
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"File: {file.name}")
        text_content = extract_text(file)
        st.text_area("Extracted Text", text_content, height=200)
        
        emails, phones = find_entities(text_content)
        st.write("Extracted Emails:", emails)
        st.write("Extracted Phone Numbers:", phones)

        if search_query:
            matches = [line for line in text_content.splitlines() if search_query.lower() in line.lower()]
            st.write(f"Search Results for '{search_query}':", matches)
else:
    st.info("Upload files using the sidebar to extract text and search for entities.")
