import streamlit as st
from PIL import Image
import easyocr
import pdfplumber
from pdf2image import convert_from_bytes
from fpdf import FPDF
from datetime import datetime
import os
import re
import numpy as np
from pydub import AudioSegment
import tempfile
import librosa

# -----------------------
# PAGE SETUP
# -----------------------
st.set_page_config(page_title="RBIS CSI Cockpit", layout="wide")
st.markdown("""
<style>
body {background-color: #0f0f0f; color: #00ff00; font-family: 'Courier New', monospace;}
.stButton>button {background-color:#111; color:#00ff00;}
.stTextArea textarea {background-color:#111; color:#00ff00; font-family:'Courier New', monospace;}
</style>
""", unsafe_allow_html=True)

st.title("RBIS Intelligence Cockpit – Behavioural & Document AI")
st.write("Upload files (PDF, PNG, JPG, JPEG, MP3/WAV/OPUS) to extract, analyze, and generate intelligence.")

# -----------------------
# INITIALIZE OCR
# -----------------------
reader = easyocr.Reader(['en'])

# -----------------------
# MASTER STORAGE
# -----------------------
master_index = []
terminal_logs = []

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

def detect_anomalies(amounts):
    anomalies = []
    seen = set()
    for amt in amounts:
        if amt in seen:
            anomalies.append(amt)
        else:
            seen.add(amt)
    return anomalies

def fraud_risk_score(text):
    fraud_keywords = ['fake', 'fraud', 'lie', 'forged', 'tampered']
    score = 0
    text_lower = text.lower()
    for kw in fraud_keywords:
        if kw in text_lower:
            score += 20
    return min(score, 100)

def check_cross_file_anomalies(file_data):
    anomalies = []
    for entry in master_index:
        for amt in file_data['amounts']:
            if amt in entry['amounts']:
                anomalies.append(f"Repeated amount: {amt}")
        for name in file_data['names']:
            if name not in entry['names']:
                anomalies.append(f"New name detected: {name}")
    master_index.append(file_data)
    return anomalies

# -----------------------
# FILE UPLOAD
# -----------------------
uploaded_files = st.file_uploader(
    "Upload your files", accept_multiple_files=True, 
    type=['pdf','png','jpg','jpeg','mp3','wav','opus']
)
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
# PROCESS FILES
# -----------------------
intelligence_data = []

for info in file_info_list:
    text_content = ""
    dates, amounts, names, anomalies, vocal_score, stress_score = [], [], [], [], 0, 0

    # PDF processing
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

    # Image processing
    elif info['name'].lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(info['path'])
        st.image(img, caption=info['name'], use_column_width=True)
        text_content = " ".join(reader.readtext(str(info['path']), detail=0))
        log(f"Processed Image: {info['name']}")

    # Audio processing (WhatsApp .opus)
    elif info['name'].lower().endswith((".mp3", ".wav", ".opus")):
        audio_path = info['path']
        if info['name'].lower().endswith(".opus"):
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
                tmp_path = tmp_wav.name
            try:
                AudioSegment.from_file(audio_path, format="opus").export(tmp_path, format="wav")
                audio_path = tmp_path
            except Exception as e:
                st.error(f"Failed to convert .opus file: {e}")
        vocal_score, stress_score = vocal_tone_score(audio_path)
        log(f"Processed Audio: {info['name']}")

    # Extract entities
    dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text_content)
    amounts = re.findall(r"£\d+(?:\.\d{2})?", text_content)
    names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text_content)
    anomalies = detect_anomalies(amounts)
    behavioural = behavioural_score(text_content)
    fraud_score = fraud_risk_score(text_content)
    cross_anomalies = check_cross_file_anomalies({
        "dates": dates,
        "amounts": amounts,
        "names": names,
        "text": text_content
    })

    intelligence_data.append({
        "file": info['name'],
        "dates": dates,
        "amounts": amounts,
        "names": names,
        "anomalies": anomalies,
        "behavioural": behavioural,
        "vocal_tone": vocal_score,
        "stress_score": stress_score,
        "fraud_score": fraud_score,
        "cross_anomalies": cross_anomalies,
        "text": text_content,
        "audio_path": audio_path if info['name'].lower().endswith((".mp3", ".wav", ".opus")) else None
    })

# -----------------------
# DASHBOARD LAYOUT
# -----------------------
left_col, right_col = st.columns([1,3])

with left_col:
    st.markdown("### Terminal Log")
    for log_msg in terminal_logs:
        st.markdown(f"<span style='color:#00ff00'>{log_msg}</span>", unsafe_allow_html=True)

with right_col:
    for data in intelligence_data:
        with st.expander(f"File: {data['file']}"):
            tab1, tab2, tab3 = st.tabs(["Behavioural", "Vocal & Audio", "Fraud & OCR"])

            with tab1:
                st.metric("Behavioural Score", data['behavioural'])
                st.text_area("Extracted Text", data['text'], height=150)

            with tab2:
                st.metric("Vocal Tone", data['vocal_tone'])
                st.metric("Stress", data['stress_score'])
                if data['audio_path']:
                    st.audio(data['audio_path'])

            with tab3:
                st.metric("Fraud Risk Score", data['fraud_score'])
                st.markdown("### OCR Preview")
                if data['text']:
                    st.text_area("OCR Text", data['text'], height=150)

# -----------------------
# PDF REPORT
# -----------------------
st.subheader("Generate Full Intelligence PDF")
if st.button("Create PDF Report"):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(0, 0, 0)
    pdf.set_text_color(0, 255, 0)
    pdf.set_font("Courier", 'B', 18)
    pdf.cell(0, 12, "RBIS Intelligence Report", ln=True, align="C", fill=True)
    pdf.ln(8)
    pdf.set_font("Courier", '', 12)
    for data in intelligence_data:
        pdf.multi_cell(0, 6, f"File: {data['file']}")
        pdf.multi_cell(0, 6, f"Behavioural Score: {data['behavioural']}")
        pdf.multi_cell(0, 6, f"Vocal Tone: {data['vocal_tone']}")
        pdf.multi_cell(0, 6, f"Stress: {data['stress_score']}")
        pdf.multi_cell(0, 6, f"Fraud Risk Score: {data['fraud_score']}")
        pdf.multi_cell(0, 6, f"Anomalies: {data['anomalies'] if data['anomalies'] else 'None'}")
        pdf.multi_cell(0, 6, f"Cross-File Anomalies: {data['cross_anomalies'] if data['cross_anomalies'] else 'None'}")
        pdf.multi_cell(0, 6, "---------------------------------------------")
    pdf_file = "RBIS_Intelligence_Report.pdf"
    pdf.output(pdf_file)
    st.success(f"PDF Report Generated: {pdf_file}")
    st.download_button("Download PDF", pdf_file)
