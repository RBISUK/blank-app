import streamlit as st
from pathlib import Path
from PIL import Image
import os
import io
import easyocr
import pdfplumber
import openai
from collections import defaultdict
import re
import whisper
from pydub import AudioSegment

# ---------------------------
# App Setup
# ---------------------------
st.set_page_config(page_title="RydenNet – Behavioural & Intelligence AI", layout="wide")

# Sidebar style
st.markdown("""
    <style>
        .sidebar .sidebar-content {
            background-color: #0b0b0b;
            color: #00ff00;
        }
        .stApp {
            background-color: #0a0a0a;
            color: #00ff00;
        }
        .stButton>button {
            background-color: #006400;
            color: white;
        }
        .stTextInput>div>input {
            background-color: #111;
            color: #00ff00;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# OpenAI API Key Handling
# ---------------------------
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    st.sidebar.error("⚠️ OpenAI API Key not found! Set OPENAI_API_KEY environment variable.")
else:
    openai.api_key = OPENAI_KEY

# ---------------------------
# Sidebar Uploads & Search
# ---------------------------
st.sidebar.header("RydenNet Control Panel")
uploaded_files = st.sidebar.file_uploader(
    "Upload Documents, Images, or Audio (PDF, PNG, JPG, JPEG, MP3, WAV, OPUS)",
    accept_multiple_files=True,
    type=['pdf','png','jpg','jpeg','mp3','wav','opus']
)

search_query = st.sidebar.text_input("Search Entities or Keywords")

# ---------------------------
# Initialize OCR and Whisper
# ---------------------------
reader = easyocr.Reader(['en'])
@st.cache_resource
def load_whisper():
    return whisper.load_model("base")
whisper_model = load_whisper()

# ---------------------------
# Intelligence Data Storage
# ---------------------------
data_store = defaultdict(list)  # Stores extracted text/audio/entities

# ---------------------------
# Utility Functions
# ---------------------------
def extract_text(file):
    if file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(file)
        text = " ".join(reader.readtext(np.array(img), detail=0))
        return text
    elif file.name.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file.name.lower().endswith((".mp3", ".wav", ".opus")):
        # Convert to WAV if needed
        audio = AudioSegment.from_file(file)
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        result = whisper_model.transcribe(buffer)
        return result['text']
    return ""

def extract_entities(text):
    # Simple regex extraction for demo
    entities = {
        "Names": re.findall(r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)?\b', text),
        "Emails": re.findall(r'\b[\w\.-]+@[\w\.-]+\b', text),
        "Phones": re.findall(r'\b(?:\+44\s?|0)\d{10,11}\b', text),
        "Dates": re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text),
        "Amounts": re.findall(r'£\d+(?:\.\d+)?', text)
    }
    return entities

def generate_ai_insights(text):
    if not OPENAI_KEY:
        return "OpenAI key not set, cannot generate intelligence."
    prompt = f"Analyze this content and provide behavioral, credibility, and fraud insights:\n\n{text}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role":"user","content":prompt}],
        temperature=0
    )
    return response['choices'][0]['message']['content']

# ---------------------------
# Process Files
# ---------------------------
if uploaded_files:
    for file in uploaded_files:
        text_content = extract_text(file)
        data_store['text'].append({'filename': file.name, 'content': text_content})
        data_store['entities'].append({'filename': file.name, 'entities': extract_entities(text_content)})
        data_store['insights'].append({'filename': file.name, 'intelligence': generate_ai_insights(text_content)})

# ---------------------------
# Search & Display
# ---------------------------
st.title("RydenNet – Behavioural & Intelligence AI")
st.subheader("Upload files, analyze content, extract entities, and generate intelligence.")

if uploaded_files:
    for idx, item in enumerate(data_store['text']):
        st.markdown(f"### {item['filename']}")
        st.markdown(f"**Extracted Text:**\n```\n{item['content'][:500]}...\n```")  # Show first 500 chars
        st.markdown("**Entities:**")
        entities = data_store['entities'][idx]['entities']
        for k,v in entities.items():
            st.markdown(f"{k}: {', '.join(v)}")
        st.markdown("**Intelligence:**")
        st.markdown(f"```\n{data_store['insights'][idx]['intelligence']}\n```")

        # Highlight search matches
        if search_query:
            matches = [line for line in item['content'].splitlines() if search_query.lower() in line.lower()]
            if matches:
                st.markdown("**Search Results:**")
                for m in matches:
                    st.markdown(f"> {m}")

else:
    st.write("Upload files using the sidebar to begin analysis.")
