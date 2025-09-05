import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
from docx import Document
import os
import openai

# -----------------------
# APP SETTINGS
# -----------------------
st.set_page_config(
    page_title="RydenNet Intelligence Suite",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
# RydenNet – Behavioural & Intelligence AI
Upload files to extract text, analyze content, and interact with the intelligence assistant.
""")

# Sidebar for uploads
with st.sidebar:
    st.subheader("Upload your files")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, Images, Audio (MP3/WAV)",
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "docx"]
    )

# -----------------------
# FUNCTIONS
# -----------------------
def extract_text(file):
    ext = file.name.split('.')[-1].lower()
    text_content = ""
    
    try:
        if ext == "pdf":
            with pdfplumber.open(file) as pdf:
                text_content = "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif ext in ["png", "jpg", "jpeg"]:
            img = Image.open(file)
            text_content = pytesseract.image_to_string(img)
        elif ext == "docx":
            doc = Document(file)
            text_content = "\n".join([p.text for p in doc.paragraphs])
        elif ext in ["mp3", "wav"]:
            text_content = "[Audio uploaded – transcription module only works locally]"
        else:
            text_content = "[Unsupported file type]"
    except Exception as e:
        text_content = f"[Error extracting text: {e}]"
    
    return text_content

def chatgpt_query(prompt_text):
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are RydenNet, an AI intelligence analyst. Extract insights, behavioral patterns, fraud signals, and credibility notes from text."},
                {"role": "user", "content": prompt_text}
            ],
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ChatGPT query failed: {e}]"

# -----------------------
# PROCESS FILES
# -----------------------
all_text_content = ""
if uploaded_files:
    st.subheader("Uploaded Files & Extracted Text")
    for file in uploaded_files:
        st.markdown(f"### {file.name}")
        text_content = extract_text(file)
        st.text_area("Extracted Text", text_content, height=200)
        all_text_content += "\n" + text_content

# -----------------------
# SEARCH & CHATGPT INTERACTION
# -----------------------
st.subheader("Search / Ask Intelligence Agent")

search_query = st.text_input("Type a query, name, email, phone, or instruction for RydenNet:")

if st.button("Submit Query") and search_query:
    if all_text_content:
        # Combine uploaded text with user query for context
        prompt = f"Analyze the following text and respond to the query.\n\nText:\n{all_text_content}\n\nQuery: {search_query}"
        result = chatgpt_query(prompt)
        st.markdown("### RydenNet Response")
        st.write(result)
    else:
        st.warning("Please upload files first to give RydenNet content to analyze.")
