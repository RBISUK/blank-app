import streamlit as st
from PIL import Image
import easyocr
import pdfplumber
from pdf2image import convert_from_bytes
from pydub import AudioSegment
import tempfile
import librosa
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from datetime import datetime
import openai

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(page_title="RBIS CSI Cockpit", layout="wide", initial_sidebar_state="expanded")

# Custom CSI / futuristic styles
st.markdown("""
<style>
body {background-color: #0a0a0a; color: #00ff00; font-family: 'Courier New', monospace;}
.stButton>button {background-color:#111; color:#00ff00; border:none;}
.stTextArea textarea {background-color:#111; color:#00ff00; font-family:'Courier New', monospace;}
.stFileUploader>div {background-color:#111; color:#00ff00; padding:5px; border-radius:5px;}
.stMetric {background-color:#111; color:#00ff00; border:1px solid #00ff00; border-radius:5px; padding:5px;}
</style>
""", unsafe_allow_html=True)

st.title("RBIS Intelligence Cockpit")
st.markdown("Upload files, analyze content, extract entities, and generate intelligence.")

# -----------------------
# INITIALIZE MODELS
# -----------------------
reader = easyocr.Reader(['en'])
openai.api_key = st.secrets.get("OPENAI_API_KEY")  # Set your OpenAI key in Streamlit secrets

# -----------------------
# DATA STORAGE
# -----------------------
terminal_logs = []
intelligence_data = []

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    terminal_logs.append(f"[{timestamp}] {msg}")

def extract_entities(text):
    dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text)
    amounts = re.findall(r"£\d+(?:\.\d{2})?", text)
    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?\d[\d -]{7,}\d", text)
    names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text)
    return dates, amounts, names, emails, phones

def behavioural_analysis(text):
    prompt = f"Analyze this text for credibility, detect exaggeration, contradictions, hedging, or suspicious statements:\n{text}\nSummarize the intelligence findings in bullet points."
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=400
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"Behavioural analysis failed: {e}"

def vocal_tone_analysis(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        rms = np.mean(librosa.feature.rms(y=y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        stress = int(min(max(rms*500 + np.std(y)*100, 0), 100))
        summary = f"RMS: {rms:.3f}, Tempo: {tempo:.1f}, Stress Index: {stress}"
        return stress, summary
    except Exception as e:
        return 0, f"Error analyzing audio: {e}"

def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            result = openai.Audio.transcriptions.create(
                file=audio_file,
                model="whisper-1"
            )
        return result.text
    except Exception as e:
        return f"Transcription failed: {e}"

def ai_agent_query(query, intelligence_data):
    context = ""
    for data in intelligence_data:
        context += f"\nFile: {data['file']}\nText: {data['text']}\nTranscript: {data.get('transcript','')}\n"
    prompt = f"Based on the following intelligence data, provide a concise, actionable intelligence report answering this query:\n{context}\nQuery: {query}\nAnswer:"
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].text.strip()
    except Exception as e:
        return f"AI Agent failed: {e}"

# -----------------------
# SIDEBAR
# -----------------------
with st.sidebar:
    st.header("Control Panel")
    uploaded_files = st.file_uploader("Upload Files (PDF, PNG, JPG, JPEG, MP3/WAV/OPUS)", accept_multiple_files=True)
    search_query = st.text_input("Search across all files")
    st.markdown("---")
    st.subheader("AI Agent Query")
    agent_query = st.text_input("Ask AI about your files:")
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
        log(f"Uploaded: {file.name}")

# -----------------------
# INTELLIGENCE EXTRACTION
# -----------------------
for info in file_info_list:
    text_content = ""
    dates, amounts, names, emails, phones = [], [], [], [], []
    transcript = ""
    vocal_score, vocal_summary = 0, ""

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
            log(f"Error processing PDF: {info['name']}: {e}")

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
        vocal_score, vocal_summary = vocal_tone_analysis(audio_path)
        transcript = transcribe_audio(audio_path)
        log(f"Processed Audio: {info['name']}")

    dates, amounts, names, emails, phones = extract_entities(text_content)
    behavioural_summary = behavioural_analysis(text_content)

    intelligence_data.append({
        "file": info['name'],
        "text": text_content,
        "dates": dates,
        "amounts": amounts,
        "names": names,
        "emails": emails,
        "phones": phones,
        "behavioural_summary": behavioural_summary,
        "vocal_score": vocal_score,
        "vocal_summary": vocal_summary,
        "transcript": transcript
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
    st.markdown(f"**Behavioural & Credibility Analysis:**\n{data['behavioural_summary']}")
    st.markdown(f"**Vocal Score:** {data['vocal_score']} | **Tone Summary:** {data['vocal_summary']}")
    st.write("Extracted Dates:", data['dates'])
    st.write("Extracted Amounts:", data['amounts'])
    st.write("Names:", data['names'])
    st.write("Emails:", data['emails'])
    st.write("Phones:", data['phones'])
    if data.get('transcript'):
        st.text_area("Transcript", data['transcript'], height=250, max_chars=None)
    st.markdown("---")

# -----------------------
# TERMINAL LOGS (CSI STYLE)
# -----------------------
st.sidebar.subheader("Terminal Logs")
for line in terminal_logs[-20:]:
    st.sidebar.markdown(f"`{line}`")
