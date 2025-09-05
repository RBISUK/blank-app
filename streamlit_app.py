import streamlit as st
import pdfplumber
import docx
import os
import tempfile
import openai
import re
import easyocr
from PIL import Image
import whisper
import pandas as pd

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(
    page_title="RydenNet – Master AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🕵️‍♂️ RydenNet – Behavioural & Intelligence AI Cockpit")

# Sidebar Upload
st.sidebar.header("📂 Upload Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs, Word docs, Images, or Audio",
    type=["pdf", "docx", "png", "jpg", "jpeg", "mp3", "wav", "opus"],
    accept_multiple_files=True
)

# Whisper model for audio
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")

whisper_model = load_whisper()

# EasyOCR
reader = easyocr.Reader(["en"])

# -------------------------------
# TEXT EXTRACTION
# -------------------------------
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.endswith((".png", ".jpg", ".jpeg")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        text = " ".join(reader.readtext(tmp_path, detail=0))
        os.remove(tmp_path)
    elif file.name.endswith((".mp3", ".wav", ".opus")):
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path)
        text = result["text"]
        os.remove(tmp_path)
    return text.strip()

# -------------------------------
# ENTITY EXTRACTION
# -------------------------------
def extract_entities(text):
    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    amounts = re.findall(r"£\d+(?:,\d{3})*(?:\.\d{2})?", text)
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"\+?\d[\d\-\s]{7,}\d", text)
    names = re.findall(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", text)
    return dates, amounts, names, emails, phones

# -------------------------------
# BEHAVIOURAL & CREDIBILITY AI
# -------------------------------
def behavioural_analysis(text):
    prompt = f"""
    You are a behavioural intelligence analyst. 
    Analyse the following text for:
    - Emotional tone
    - Credibility markers
    - Signs of exaggeration or manipulation
    - Overall behavioural risk score (0–100)

    Text:
    {text[:2000]}  # limit text
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in behavioural analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Behavioural analysis failed: {e}"

# -------------------------------
# MAIN UI
# -------------------------------
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📄 File: {file.name}")
        text_content = extract_text(file)

        if not text_content:
            st.warning("No text could be extracted.")
            continue

        st.markdown("### 🔎 Extracted Text")
        st.text_area("Raw Content", text_content[:3000], height=200)

        # Entities
        dates, amounts, names, emails, phones = extract_entities(text_content)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📅 Dates:**")
            st.write(dates if dates else "None")
            st.markdown("**💷 Amounts:**")
            st.write(amounts if amounts else "None")
        with col2:
            st.markdown("**👤 Names:**")
            st.write(names if names else "None")
        with col3:
            st.markdown("**📧 Emails:**")
            st.write(emails if emails else "None")
            st.markdown("**📞 Phones:**")
            st.write(phones if phones else "None")

        # Behavioural Intelligence
        st.markdown("### 🧠 Behavioural & Credibility Analysis")
        behavioural = behavioural_analysis(text_content)
        st.write(behavioural)

else:
    st.info("Upload files from the sidebar to begin.")
