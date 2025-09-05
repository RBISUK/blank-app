# streamlit_app.py
import streamlit as st
import pdfplumber
from docx import Document
import re
import openai

# -------------------
# PAGE CONFIG
# -------------------
st.set_page_config(page_title="RydenNet Intelligence AI", layout="wide")
st.markdown(
    """
    <style>
    /* Global background */
    body { background-color: #0d1117; color: #00ff00; font-family: 'Courier New', monospace; }

    /* Sidebar style */
    .css-1d391kg {background-color: #0d1117;}
    .css-1v3fvcr {color: #00ff00;}
    .css-ffhzg2 {color: #00ff00;}
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 { color: #00ff00; }

    /* Text area styling */
    textarea { background-color: #0d1117; color: #00ff00; border: 1px solid #00ff00; }

    /* Buttons */
    div.stButton > button:first-child { background-color: #00ff00; color: #0d1117; }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("RydenNet – Behavioural & Intelligence AI")

# -------------------
# OPENAI SETUP
# -------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    st.warning("⚠️ Set OPENAI_API_KEY in Streamlit secrets to enable intelligence features.")
openai.api_key = OPENAI_API_KEY

# -------------------
# SIDEBAR
# -------------------
st.sidebar.header("Upload Files & Ask AI")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDFs/DOCX", type=['pdf', 'docx'], accept_multiple_files=True
)
search_query = st.sidebar.text_input("Search text/entities...")
user_prompt = st.sidebar.text_area("Ask RydenNet AI")

# -------------------
# UTILS
# -------------------
def extract_text(file):
    if file.name.endswith('.pdf'):
        text = ""
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    elif file.name.endswith('.docx'):
        doc = Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""

def extract_entities(text):
    dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
    amounts = re.findall(r'£\d+(?:,\d{3})*(?:\.\d{2})?', text)
    emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w{2,4}\b', text)
    phones = re.findall(r'\b(?:\+44\s?\d{10}|\d{10})\b', text)
    names = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', text)
    return {"Dates": dates, "Amounts": amounts, "Emails": emails, "Phones": phones, "Names": names}

def ask_chatgpt(prompt):
    if not OPENAI_API_KEY:
        return "OpenAI API key not set."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# -------------------
# DISPLAY FILES & TEXT
# -------------------
st.header("Uploaded Files & Extracted Text")
file_texts = {}
for file in uploaded_files:
    text = extract_text(file)
    file_texts[file.name] = text
    st.subheader(file.name)
    st.text_area("Extracted Text", text, height=200)

# -------------------
# ENTITY EXTRACTION
# -------------------
st.header("Extracted Entities")
for fname, text in file_texts.items():
    entities = extract_entities(text)
    st.subheader(fname)
    st.json(entities)

# -------------------
# SEARCH FUNCTION
# -------------------
if search_query:
    st.header(f"Search Results for '{search_query}'")
    for fname, text in file_texts.items():
        matches = [line for line in text.split("\n") if search_query.lower() in line.lower()]
        st.subheader(fname)
        if matches:
            st.write("\n".join(matches))
        else:
            st.write("No matches found.")

# -------------------
# CHATGPT
# -------------------
if user_prompt:
    st.header("RydenNet AI Response")
    answer = ask_chatgpt(user_prompt)
    st.write(answer)
