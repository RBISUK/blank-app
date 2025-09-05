# streamlit_app.py

import streamlit as st
from PIL import Image
import easyocr
import os

# Initialize EasyOCR once
reader = easyocr.Reader(['en'])

st.set_page_config(page_title="RBIS Dashboard", layout="wide")

st.title("RBIS Dashboard – Behavioural & Intelligence Services")
st.markdown("Upload documents or images to extract text and analyse.")

# File uploader
uploaded_files = st.file_uploader("Upload files (PDF, PNG, JPG, JPEG)", accept_multiple_files=True)

# Store uploaded file info
file_info_list = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save file temporarily
        file_path = os.path.join("temp", uploaded_file.name)
        os.makedirs("temp", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        file_info_list.append({
            "name": uploaded_file.name,
            "path": file_path
        })

# Tabs for UI
tabs = st.tabs(["Uploaded Files", "Image OCR"])

# 1️⃣ Uploaded Files tab
with tabs[0]:
    st.subheader("Uploaded Files")
    if file_info_list:
        for info in file_info_list:
            st.write(f"- {info['name']}")
    else:
        st.write("No files uploaded yet.")

# 2️⃣ Image OCR tab
with tabs[1]:
    st.subheader("Image Review & OCR")
    if file_info_list:
        for info in file_info_list:
            if info['name'].lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(info['path'])
                    st.image(img, caption=info['name'], use_column_width=True)
                    
                    # EasyOCR extraction
                    ocr_text = " ".join(reader.readtext(str(info['path']), detail=0))
                    st.write("OCR Extracted Text:", ocr_text if ocr_text.strip() else "None")
                except Exception as e:
                    st.error(f"Could not process {info['name']}: {e}")
    else:
        st.write("No image files uploaded.")
