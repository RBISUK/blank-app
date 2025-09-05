import streamlit as st
from PIL import Image
import easyocr
import pdfplumber
import re
import openai
import spacy
from pydub import AudioSegment
import tempfile

# --------------------
# Config
# --------------------
st.set_page_config(
    page_title="RydenNet Intelligence Cockpit",
    layout="wide",
    page_icon="🤖"
)
st.markdown("<h1 style='text-align:center'>RydenNet – Behavioural & Intelligence AI</h1>", unsafe_allow_html=True)

# OpenAI key
openai.api_key = st.secrets.get("OPENAI_API_KEY")

# --------------------
# Sidebar
# --------------------
st.sidebar.title("RydenNet Control Panel")
uploaded_files = st.sidebar.file_uploader(
    "Upload files", 
    type=["pdf","png","jpg","jpeg","mp3","wav","opus"], 
    accept_multiple_files=True
)
search_query = st.sidebar.text_input("Search intelligence...")
st.sidebar.markdown("---")
st.sidebar.markdown("**Logs:**")
terminal_logs = []

# --------------------
# OCR & PDF text extraction
# --------------------
reader = easyocr.Reader(['en'])
nlp = spacy.load("en_core_web_sm")

def extract_text(file):
    try:
        if file.name.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file)
            text = " ".join(reader.readtext(file, detail=0))
        elif file.name.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
        elif file.name.lower().endswith((".mp3", ".wav", ".opus")):
            text = f"[Audio file '{file.name}' ready for transcription module]"
        else:
            text = ""
        terminal_logs.append(f"Extracted text from {file.name}")
        return text
    except Exception as e:
        terminal_logs.append(f"Failed to extract from {file.name}: {e}")
        return ""

# --------------------
# Behavioural & credibility analysis
# --------------------
def behavioural_analysis(text):
    prompt = f"""
    You are an expert behavioural intelligence analyst.
    Analyze the following text for credibility, exaggeration, contradictions, hedging, suspicious statements.
    Return results as concise bullet points.
    {text}
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role":"system","content":"You are a behavioural intelligence analyst."},
                {"role":"user","content":prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        terminal_logs.append(f"Behavioural analysis failed: {e}")
        return f"Behavioural analysis failed: {e}"

# --------------------
# Text classification / entity extraction
# --------------------
def classify_text_entities(text):
    doc = nlp(text)
    names, places, orgs, dates, amounts, emails, phones = [], [], [], [], [], [], []

    for ent in doc.ents:
        if ent.label_ == "PERSON": names.append(ent.text)
        elif ent.label_ in ["GPE","LOC"]: places.append(ent.text)
        elif ent.label_ == "ORG": orgs.append(ent.text)
        elif ent.label_ in ["DATE","TIME"]: dates.append(ent.text)
        elif ent.label_ == "MONEY": amounts.append(ent.text)

    emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
    phones = re.findall(r"\+?\d[\d -]{7,}\d", text)

    return {
        "names": list(set(names)),
        "places": list(set(places)),
        "orgs": list(set(orgs)),
        "dates": list(set(dates)),
        "amounts": list(set(amounts)),
        "emails": list(set(emails)),
        "phones": list(set(phones))
    }

# --------------------
# Process uploads
# --------------------
intelligence_data = []

if uploaded_files:
    for file in uploaded_files:
        text_content = extract_text(file)
        if text_content:
            entities = classify_text_entities(text_content)
            behaviour = behavioural_analysis(text_content)
            intelligence_data.append({
                "file": file.name,
                "text": text_content,
                "behaviour": behaviour,
                **entities
            })

# --------------------
# Search function
# --------------------
def search_intelligence(query, data_list):
    results = []
    query_lower = query.lower()
    for data in data_list:
        combined_text = " ".join([
            data['text'],
            " ".join(data['names']),
            " ".join(data['places']),
            " ".join(data['orgs']),
            " ".join(data['dates']),
            " ".join(data['amounts']),
            " ".join(data['emails']),
            " ".join(data['phones'])
        ]).lower()
        if query_lower in combined_text:
            results.append(data)
    return results

if search_query:
    intelligence_data = search_intelligence(search_query, intelligence_data)
    st.sidebar.markdown(f"**{len(intelligence_data)} results found for '{search_query}'**")

# --------------------
# Display intelligence
# --------------------
for data in intelligence_data:
    st.markdown(f"### File: {data['file']}")
    st.markdown(f"**Behavioural & Credibility Analysis:**\n{data['behaviour']}")
    st.write("**Entities:**")
    st.write(f"Names: {data['names']}")
    st.write(f"Places: {data['places']}")
    st.write(f"Organisations: {data['orgs']}")
    st.write(f"Dates: {data['dates']}")
    st.write(f"Amounts: {data['amounts']}")
    st.write(f"Emails: {data['emails']}")
    st.write(f"Phones: {data['phones']}")
    st.markdown("---")

# --------------------
# Sidebar logs
# --------------------
st.sidebar.subheader("RydenNet Logs")
for line in terminal_logs[-20:]:
    st.sidebar.markdown(f"`{line}`")

