import streamlit as st
from PIL import Image
import easyocr
import pdfplumber
from pdf2image import convert_from_bytes
from pydub import AudioSegment
import tempfile
import librosa
import whisper
import matplotlib.pyplot as plt
import re
import os
from datetime import datetime
import numpy as np
import openai  # For ChatGPT AI agent

# -----------------------
# CONFIGURATION
# -----------------------
st.set_page_config(page_title="RBIS CSI Cockpit", layout="wide")
st.markdown("""
<style>
body {background-color: #0f0f0f; color: #00ff00; font-family: 'Courier New', monospace;}
.stButton>button {background-color:#111; color:#00ff00;}
.stTextArea textarea {background-color:#111; color:#00ff00; font-family:'Courier New', monospace;}
.stFileUploader>div {background-color:#111; color:#00ff00; padding:5px; border-radius:5px;}
.stMetric {background-color:#111; color:#00ff00; border:1px solid #00ff00; border-radius:5px; padding:5px;}
</style>
""", unsafe_allow_html=True)
st.title("RBIS CSI Intelligence Cockpit")

# -----------------------
# INITIALIZE MODELS
# -----------------------
reader = easyocr.Reader(['en'])
whisper_model = whisper.load_model("base")
openai.api_key = st.secrets.get("OPENAI_API_KEY")  # Set your OpenAI API key in Streamlit Secrets

# -----------------------
# STORAGE
# -----------------------
terminal_logs = []
intelligence_data = []

# -----------------------
# FUNCTIONS
# -----------------------
def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    terminal_logs.append(f"[{timestamp}] {msg}")

def behavioural_score(text):
    uncertainty_words = ['maybe', 'possibly', 'uncertain', 'believe', 'guess']
    score = 100
    text_lower = text.lower()
    for word in uncertainty_words:
        if word in text_lower.split():
            score -= 5
    return max(score,0)

def vocal_tone_score(audio_path=None):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        score = int(min(max(rms*1000 + tempo/2, 0), 100))
        stress = int(min(max(rms*500 + np.std(y)*100, 0), 100))
        return score, stress
    except:
        return 0, 0

def transcribe_audio(file_path):
    result = whisper_model.transcribe(file_path)
    return result['text']

def extract_entities(text):
    dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text)
    amounts = re.findall(r"£\d+(?:\.\d{2})?", text)
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?\d[\d -]{7,}\d", text)
    names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
    return dates, amounts, names, emails, phones

def ai_agent_query(query, intelligence_data):
    context = ""
    for data in intelligence_data:
        context += f"\nFile: {data['file']}\nText: {data['text']}\nTranscript: {data.get('transcript','')}\n"
    prompt = f"Based on the following intelligence data, answer the query accurately and concisely:\n{context}\n\nQuery: {query}\nAnswer:"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=500
    )
    return response.choices[0].text.strip()

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.header("Control Panel")
    uploaded_files = st.file_uploader(
        "Upload Files (PDF, PNG, JPG, JPEG, MP3/WAV/OPUS)", 
        accept_multiple_files=True
    )
    search_query = st.text_input("Search across files")
    st.markdown("---")
    st.subheader("AI Agent")
    agent_query = st.text_input("Ask the AI about your files:")
    if agent_query and st.button("Get Intelligence"):
        if intelligence_data:
            answer = ai_agent_query(agent_query, intelligence_data)
            st.markdown(f"<div style='background-color:#111; padding:10px; color:#00ff00;'>{answer}</div>", unsafe_allow_html=True)
        else:
            st.warning("Upload files first!")

# -----------------------
# FILE PROCESSING
# -----------------------
file_info_list = []
if uploaded_files:
    os.makedirs("temp", exist_ok=True)
    for file in uploaded_files:
        path = os.path.join("temp", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        file_info_list.append({"name": file.name, "path": path})
        log(f"Uploaded file: {file.name}")

# -----------------------
# INTELLIGENCE EXTRACTION
# -----------------------
for info in file_info_list:
    text_content = ""
    dates, amounts, names, emails, phones = [], [], [], [], []
    transcript = ""
    vocal_score, stress_score = 0, 0
    audio_path = None

    # PDF
    if info['name'].lower().endswith(".pdf"):
        try:
            with pdfplumber.open(info['path']) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
            if not text_content.strip():
                images = convert_from_bytes(open(info['path'], 'rb').read())
                for img in images:
                    text_content += " ".join(reader.readtext(img, detail=0)) + "\n"
            log(f"Processed PDF: {info['name']}")
        except Exception as e:
            st.error(f"PDF error: {e}")
            log(f"Error processing PDF: {info['name']}")

    # Image
    elif info['name'].lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(info['path'])
        st.image(img, caption=info['name'], use_column_width=True)
        text_content = " ".join(reader.readtext(str(info['path']), detail=0))
        log(f"Processed Image: {info['name']}")

    # Audio
    elif info['name'].lower().endswith((".mp3", ".wav", ".opus")):
        audio_path = info['path']
        if info['name'].lower().endswith(".opus"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_path = tmp_wav.name
            AudioSegment.from_file(audio_path, format="opus").export(tmp_path, format="wav")
            audio_path = tmp_path
        vocal_score, stress_score = vocal_tone_score(audio_path)
        transcript = transcribe_audio(audio_path)
        log(f"Processed Audio: {info['name']}")

    dates, amounts, names, emails, phones = extract_entities(text_content)
    behavioural = behavioural_score(text_content)

    intelligence_data.append({
        "file": info['name'],
        "text": text_content,
        "dates": dates,
        "amounts": amounts,
        "names": names,
        "emails": emails,
        "phones": phones,
        "behavioural": behavioural,
        "vocal_score": vocal_score,
        "stress_score": stress_score,
        "transcript": transcript,
        "audio_path": audio_path
    })

# -----------------------
# SEARCH RESULTS
# -----------------------
if search_query:
    st.subheader(f"Search Results for: {search_query}")
    for data in intelligence_data:
        matches = []
        for field in ['names','emails','phones','text','transcript']:
            if isinstance(data[field], list):
                matches += [v for v in data[field] if search_query.lower() in str(v).lower()]
            elif isinstance(data[field], str):
                if search_query.lower() in data[field].lower():
                    matches.append(search_query)
        if matches:
            st.markdown(f"**{data['file']}**: {matches}")

# -----------------------
# DISPLAY INTELLIGENCE
# -----------------------
for data in intelligence_data:
    st.markdown("---")
    st.subheader(f"Intelligence Report – {data['file']}")
    st.metric("Behavioural Score", data['behavioural'])
    st.metric("Vocal Tone", data['vocal_score'])
    st.metric("Stress Score", data['stress_score'])
    st.write("Extracted Dates:", data['dates'])
    st.write("Extracted Amounts:", data['amounts'])
    st.write("Names:", data['names'])
    st.write("Emails:", data['emails'])
    st.write("Phones:", data['phones'])
    if data['audio_path']:
        st.audio(data['audio_path'])
        y, sr = librosa.load(data['audio_path'], sr=None)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        fig, ax = plt.subplots(figsize=(10,3))
        librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap='magma')
        ax.set_title('Audio Spectrogram')
        st.pyplot(fig)
        st.text_area("Transcript", data['transcript'], height=150)

# -----------------------
# TERMINAL LOG
# -----------------------
st.markdown("---")
st.subheader("Terminal Log")
for log_msg in terminal_logs:
    st.markdown(f"<span style='color:#00ff00'>{log_msg}</span>", unsafe_allow_html=True)
