import streamlit as st
import pdfplumber
from PIL import Image
from pydub import AudioSegment
import pytesseract
import openai
from docx import Document
import os

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
Upload files to extract text, analyze content, and generate intelligence.
""")

# Sidebar upload
with st.sidebar:
    st.subheader("Upload your files")
    uploaded_files = st.file_uploader(
        "PDF, DOCX, Images, Audio (MP3/WAV)",
        accept_multiple_files=True,
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "docx"]
    )

    search_query = st.text_input("Search for names, emails, phone numbers...")

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

def generate_intelligence(text):
    # Lightweight example using OpenAI (requires OPENAI_API_KEY)
    if not text.strip():
        return "No extractable text available."
    
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an intelligence agent analyzing text for behavioural, financial, and credibility information."},
                {"role": "user", "content": text}
            ],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Intelligence generation failed: {e}]"

# -----------------------
# PROCESS FILES
# -----------------------
if uploaded_files:
    st.subheader("Uploaded Files & Extracted Text")
    
    for file in uploaded_files:
        st.markdown(f"### {file.name}")
        text_content = extract_text(file)
        st.text_area("Extracted Text", text_content, height=200)
        
        st.subheader("Generated Intelligence")
        intelligence = generate_intelligence(text_content)
        st.write(intelligence)

# -----------------------
# SEARCH FUNCTIONALITY
# -----------------------
if search_query and uploaded_files:
    st.subheader(f"Search results for '{search_query}'")
    for file in uploaded_files:
        text_content = extract_text(file)
        if search_query.lower() in text_content.lower():
            st.markdown(f"- Found in **{file.name}**")
        else:
            st.markdown(f"- Not found in **{file.name}**")
