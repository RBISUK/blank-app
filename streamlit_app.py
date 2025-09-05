import streamlit as st
import pdfplumber
import docx
import tempfile
import os
import re
import openai
import nltk
from textblob import TextBlob
from PIL import Image

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="🕵️ RydenNet – Behavioural & Intelligence AI Cockpit",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# TERMINAL / CSI STYLE
# -------------------------------
st.markdown("""
<style>
body, .stApp {
    background-color: #000000;
    color: #00FF00;
}
.stButton>button {
    background-color: #222222;
    color: #00FF00;
    border-radius:5px;
}
.stTextInput>div>div>input {
    background-color: #111111;
    color: #00FF00;
}
textarea, .stTextArea>div>div>textarea {
    background-color: #111111;
    color: #00FF00;
}
</style>
""", unsafe_allow_html=True)

st.title("🕵️‍♂️ RydenNet – Behavioural & Intelligence AI Cockpit")
st.markdown("Upload files, analyze content, extract entities, and generate intelligence.")

# -------------------------------
# DOWNLOAD NLP CORPORA
# -------------------------------
with st.spinner("Downloading NLP corpora for TextBlob..."):
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('brown')

# -------------------------------
# SIDEBAR UPLOAD + SEARCH
# -------------------------------
st.sidebar.header("📂 Upload & Search")
uploaded_files = st.sidebar.file_uploader(
    "Upload files",
    type=["pdf", "docx", "png", "jpg", "jpeg", "mp3", "wav", "opus"],
    accept_multiple_files=True
)
search_query = st.sidebar.text_input("🔎 Search entities")

# -------------------------------
# FUNCTIONS
# -------------------------------
def extract_text(file):
    text = ""
    if file.name.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif file.name.lower().endswith((".png", ".jpg", ".jpeg")):
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        results = reader.readtext(tmp_path, detail=0)
        os.remove(tmp_path)
        text = " ".join(results)
    elif file.name.lower().endswith((".mp3", ".wav", ".opus")):
        text = transcribe_audio(file)
    return text

def transcribe_audio(file):
    try:
        import whisper
        with st.spinner("Loading Whisper model..."):
            model = whisper.load_model("base")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.name) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name
        result = model.transcribe(tmp_path)
        os.remove(tmp_path)
        return result["text"]
    except Exception:
        try:
            file.seek(0)
            transcript = openai.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=file
            )
            return transcript.text
        except Exception as e2:
            return f"⚠️ Audio transcription failed: {e2}"

def extract_entities(text):
    dates = re.findall(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", text)
    amounts = re.findall(r"£?\d+(?:\.\d{1,2})?", text)
    emails = re.findall(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phones = re.findall(r"\+?\d[\d\s-]{7,}\d", text)

    try:
        blob = TextBlob(text)
        names = [word for word, tag in blob.tags if tag in ("NNP", "NNPS")]
    except Exception:
        names = []

    return {
        "Dates": dates,
        "Finance": amounts,
        "Persons": names,
        "Contacts": emails + phones
    }

def behavioural_score(text):
    try:
        blob = TextBlob(text)
        sentiment = blob.sentiment.polarity
        score = round((sentiment + 1) * 50, 2)
        return score, f"Sentiment polarity {sentiment:.2f}"
    except Exception as e:
        return 0, f"⚠️ Behavioural analysis failed: {e}"

# -------------------------------
# MAIN DASHBOARD
# -------------------------------
if uploaded_files:
    for file in uploaded_files:
        st.subheader(f"📑 File: {file.name}")
        text_content = extract_text(file)

        if not text_content.strip():
            st.warning("⚠️ No readable content extracted.")
            continue

        # ENTITY + BEHAVIOUR
        entities = extract_entities(text_content)
        behaviour, behaviour_summary = behavioural_score(text_content)

        # SEARCH FILTER
        if search_query and search_query.lower() not in text_content.lower():
            st.info(f"Search term '{search_query}' not found in {file.name}.")
            continue

        # DISPLAY REPORT
        with st.expander("📜 Full Transcript / Extracted Text", expanded=False):
            st.text_area("Extracted Text", text_content, height=250)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Behavioural & Credibility Analysis")
            st.write(f"**Behavioural Score:** {behaviour}")
            st.write(f"**Summary:** {behaviour_summary}")

            st.markdown("### Extracted Entities")
            for cat, vals in entities.items():
                st.write(f"**{cat}:** {vals}")

        with col2:
            st.markdown("### Intelligence Insights")
            try:
                insights = openai.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an intelligence analyst. Categorise names, places, organisations, and financial figures. Identify credibility issues or anomalies."},
                        {"role": "user", "content": text_content[:4000]}
                    ]
                )
                st.write(insights.choices[0].message.content)
            except Exception as e:
                st.error(f"⚠️ Intelligence generation failed: {e}")
