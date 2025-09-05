# streamlit_app.py
import streamlit as st
import pdfplumber
import docx
import os
from PIL import Image
import easyocr
from pydub import AudioSegment
import whisper
import re

# ------------------ UI & Layout ------------------
st.set_page_config(page_title="RydenNet Intelligence Cockpit", layout="wide")

# Sidebar
st.sidebar.title("RydenNet Cockpit")
uploaded_files = st.sidebar.file_uploader(
    "Upload files (PDF, DOCX, TXT, PNG, JPG, JPEG, MP3, WAV, OPUS)", 
    type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "mp3", "wav", "opus"], 
    accept_multiple_files=True
)
search_query = st.sidebar.text_input("Search uploaded content...")

# ------------------ Helpers ------------------
def extract_text(file):
    text_content = ""
    name = file.name.lower()
    if name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text_content += page.extract_text() + "\n"
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text_content += para.text + "\n"
    elif name.endswith(".txt"):
        text_content += file.read().decode("utf-8")
    elif name.endswith((".mp3", ".wav", ".opus")):
        try:
            audio = AudioSegment.from_file(file)
            audio.export("temp.wav", format="wav")
            whisper_model = whisper.load_model("base")
            result = whisper_model.transcribe("temp.wav")
            text_content += result["text"]
            os.remove("temp.wav")
        except Exception as e:
            st.error(f"Audio processing failed: {e}")
    return text_content

# EasyOCR setup
reader = easyocr.Reader(["en"], gpu=False)

def extract_entities(text):
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?\d[\d -]{8,}\d", text)
    dates = re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text)
    amounts = re.findall(r"£\d+[.,]?\d*", text)
    return {"emails": emails, "phones": phones, "dates": dates, "amounts": amounts}

# ------------------ Main Display ------------------
st.title("RydenNet Intelligence Cockpit")
st.markdown(
    """
    Upload files, analyze content, extract entities, and generate intelligence.
    """
)

# Process uploaded files
all_text = ""
file_entities = {}
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"File: {file.name}")
        if file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file)
            st.image(img, caption=file.name, use_column_width=True)
            ocr_text = " ".join(reader.readtext(file, detail=0))
            st.text_area("OCR Extracted Text", ocr_text, height=150)
            all_text += ocr_text + "\n"
        else:
            text_content = extract_text(file)
            st.text_area("Extracted Text", text_content, height=150)
            all_text += text_content + "\n"

        file_entities[file.name] = extract_entities(all_text)

# ------------------ Search Function ------------------
if search_query:
    st.subheader("Search Results")
    matches = [line for line in all_text.split("\n") if search_query.lower() in line.lower()]
    if matches:
        for m in matches:
            st.write(f"- {m}")
    else:
        st.write("No matches found.")

# ------------------ Intelligence Summary ------------------
st.subheader("Entity Summary")
for fname, entities in file_entities.items():
    st.markdown(f"**{fname}**")
    st.write("Emails:", entities["emails"])
    st.write("Phones:", entities["phones"])
    st.write("Dates:", entities["dates"])
    st.write("Amounts:", entities["amounts"])

st.markdown("---")
st.info("Behavioural and AI intelligence modules are under development for credibility, tone, and fraud detection.")
