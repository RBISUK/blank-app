import streamlit as st
from PIL import Image
import easyocr
import pdfplumber
from pdf2image import convert_from_bytes
from fpdf import FPDF
from datetime import datetime
from textblob import TextBlob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# -----------------------
# PAGE SETUP
# -----------------------
st.set_page_config(page_title="RBIS Intelligence Cockpit", layout="wide")
st.markdown("""
<style>
body {background-color: #0f0f0f; color: #00ff00; font-family: 'Courier New', monospace;}
.stButton>button {background-color:#111; color:#00ff00;}
.stTextArea textarea {background-color:#111; color:#00ff00; font-family:'Courier New', monospace;}
</style>
""", unsafe_allow_html=True)

st.title("RBIS Intelligence Cockpit – Behavioural & Document AI")
st.write("Upload files (PDF, PNG, JPG, JPEG, MP3/WAV) to extract, analyze, and generate intelligence.")

# -----------------------
# INITIALIZE OCR
# -----------------------
reader = easyocr.Reader(['en'])

# -----------------------
# MASTER DATA STORAGE
# -----------------------
master_index = []

# -----------------------
# FUNCTIONS
# -----------------------
def behavioural_score(text):
    uncertainty_words = ['maybe', 'possibly', 'uncertain', 'believe', 'guess']
    score = 100
    blob = TextBlob(text.lower())
    for word in uncertainty_words:
        if word in blob.words:
            score -= 5
    return max(score, 0)

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
uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True, type=['pdf','png','jpg','jpeg','mp3','wav'])
file_info_list = []

if uploaded_files:
    os.makedirs("temp", exist_ok=True)
    for file in uploaded_files:
        path = os.path.join("temp", file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        file_info_list.append({"name": file.name, "path": path})

# -----------------------
# INTELLIGENCE PROCESSING
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
        except Exception as e:
            st.error(f"PDF error: {e}")

    # Image processing
    elif info['name'].lower().endswith((".png", ".jpg", ".jpeg")):
        img = Image.open(info['path'])
        st.image(img, caption=info['name'], use_column_width=True)
        text_content = " ".join(reader.readtext(str(info['path']), detail=0))

    # Audio processing
    elif info['name'].lower().endswith((".mp3",".wav")):
        vocal_score, stress_score = vocal_tone_score(info['path'])

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
        "text": text_content
    })

# -----------------------
# INTERACTIVE DASHBOARD
# -----------------------
st.subheader("Futuristic CSI Terminal")
for data in intelligence_data:
    with st.expander(f"File: {data['file']}"):
        st.markdown(f"**Dates:** {', '.join(data['dates']) if data['dates'] else 'None'}")
        st.markdown(f"**Amounts (Anomalies in red):**")
        for amt in data['amounts']:
            if amt in data['anomalies'] or any(f"Repeated amount: {amt}" in a for a in data['cross_anomalies']):
                st.markdown(f"<span style='color:red'>{amt}</span>", unsafe_allow_html=True)
            else:
                st.markdown(f"<span style='color:#00ff00'>{amt}</span>", unsafe_allow_html=True)
        st.markdown(f"**Names:** {', '.join(data['names']) if data['names'] else 'None'}")
        st.markdown(f"**Behavioural Score:** {data['behavioural']}/100")
        st.markdown(f"**Vocal Tone Score:** {data['vocal_tone']}/100")
        st.markdown(f"**Stress Score:** {data['stress_score']}/100")
        st.markdown(f"**Fraud Risk Score:** {data['fraud_score']}/100")
        st.markdown(f"**Cross-file Anomalies:** {', '.join(data['cross_anomalies']) if data['cross_anomalies'] else 'None'}")
        st.text_area("Raw Extracted Text", data['text'], height=150)

# -----------------------
# ANALYTICS CHARTS
# -----------------------
if intelligence_data:
    st.subheader("Analytics Dashboard")
    all_amounts = []
    for data in intelligence_data:
        for amt in data['amounts']:
            all_amounts.append(float(amt.replace("£","")))
    if all_amounts:
        df = pd.DataFrame({"Amounts": all_amounts})
        st.bar_chart(df)

# -----------------------
# PDF REPORT GENERATION
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
    pdf.set_font("Courier", '', 10)
    pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", ln=True)
    pdf.ln(5)

    for data in intelligence_data:
        pdf.set_fill_color(20, 20, 20)
        pdf.set_text_color(0, 255, 0)
        pdf.set_font("Courier", 'B', 14)
        pdf.cell(0, 10, f"File: {data['file']}", ln=True, fill=True)
        pdf.ln(2)

        pdf.set_font("Courier", '', 12)
        pdf.multi_cell(0, 8, f"Dates: {', '.join(data['dates']) if data['dates'] else 'None'}")

        amounts_str = ""
        for amt in data['amounts']:
            if amt in data['anomalies'] or any(f"Repeated amount: {amt}" in a for a in data['cross_anomalies']):
                amounts_str += f"[!]{amt} "
            else:
                amounts_str += f"{amt} "
        pdf.set_text_color(255, 0, 0)
        pdf.multi_cell(0, 8, f"Amounts (Anomalies in red): {amounts_str.strip()}")
        pdf.set_text_color(0, 255, 0)

        pdf.multi_cell(0, 8, f"Names: {', '.join(data['names']) if data['names'] else 'None'}")
        pdf.multi_cell(0, 8, f"Behavioural Score: {data['behavioural']}/100")
        pdf.multi_cell(0, 8, f"Vocal Tone Score: {data['vocal_tone']}/100")
        pdf.multi_cell(0, 8, f"Stress Score: {data['stress_score']}/100")
        pdf.multi_cell(0, 8, f"Fraud Risk Score: {data['fraud_score']}/100")
        pdf.multi_cell(0, 8, f"Cross-file Anomalies: {', '.join(data['cross_anomalies']) if data['cross_anomalies'] else 'None'}")
        pdf.ln(5)

    pdf_bytes = pdf.output(dest='S').encode('latin1')
    st.download_button(
        label="Download Full Intelligence PDF",
        data=pdf_bytes,
        file_name="RBIS_Futuristic_Intelligence_Report.pdf",
        mime="application/pdf"
    )
    st.success("Full Intelligence PDF generated!")
