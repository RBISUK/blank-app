import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
import re
import os
import openai
from pydub import AudioSegment
import io

st.set_page_config(page_title="RydenNet", layout="wide")

# UI styling
st.markdown("""
    <style>
        .stApp {background-color: #ffffff; color: #000000;}
        .stSidebar .sidebar-content {background-color: #f0f0f0; color: #000000;}
        .stButton>button {background-color: #006400; color:white;}
    </style>
""", unsafe_allow_html=True)

# OpenAI API key
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY
else:
    st.sidebar.warning("⚠️ Set OPENAI_API_KEY environment variable.")

# File upload
uploaded_files = st.sidebar.file_uploader("Upload PDF/Images/Audio", accept_multiple_files=True,
                                          type=['pdf','png','jpg','jpeg','mp3','wav','opus'])

search_query = st.sidebar.text_input("Search Text/Entities")

data_store = []

def extract_text(file):
    try:
        if file.name.lower().endswith((".png",".jpg",".jpeg")):
            img = Image.open(file)
            text = pytesseract.image_to_string(img)
            return text
        elif file.name.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
            return text
        elif file.name.lower().endswith((".mp3",".wav",".opus")):
            # Convert audio to wav in-memory
            try:
                audio = AudioSegment.from_file(file)
                buf = io.BytesIO()
                audio.export(buf, format="wav")
                buf.seek(0)
                return "Audio transcription placeholder (Whisper to be added)"
            except FileNotFoundError:
                return "⚠️ Audio processing failed: ffmpeg not found"
    except Exception as e:
        return f"⚠️ Failed to extract text: {e}"
    return ""

def extract_entities(text):
    return {
        "Names": re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', text),
        "Emails": re.findall(r'[\w\.-]+@[\w\.-]+', text),
        "Phones": re.findall(r'(?:\+44\s?|0)\d{10,11}', text),
        "Dates": re.findall(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', text),
        "Amounts": re.findall(r'£\d+(?:\.\d+)?', text)
    }

def generate_ai_insights(text):
    if not OPENAI_KEY:
        return "OpenAI key missing"
    prompt = f"Analyze for behavioral, credibility, and fraud insights:\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# Process uploads
if uploaded_files:
    for file in uploaded_files:
        text_content = extract_text(file)
        entities = extract_entities(text_content)
        insights = generate_ai_insights(text_content)
        data_store.append({'filename': file.name, 'text': text_content, 'entities': entities, 'insights': insights})

# Display
st.title("RydenNet – Behavioural & Intelligence AI")
if data_store:
    for item in data_store:
        st.subheader(item['filename'])
        st.text_area("Extracted Text", value=item['text'][:500]+"...", height=150)
        st.write("Entities:", item['entities'])
        st.text_area("Intelligence Insights", value=item['insights'], height=150)
        if search_query:
            matches = [line for line in item['text'].splitlines() if search_query.lower() in line.lower()]
            if matches:
                st.write("Search Matches:", matches)
else:
    st.info("Upload files via the sidebar to begin analysis.")
