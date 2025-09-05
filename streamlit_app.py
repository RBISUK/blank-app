import streamlit as st
from PIL import Image
import easyocr
import os
import pdfplumber
from pdf2image import convert_from_bytes
import re
from fpdf import FPDF
import io

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

st.set_page_config(page_title="RBIS Dashboard", layout="wide")
st.title("RBIS Dashboard – Behavioural & Intelligence Services")
st.markdown("Upload documents or images to extract text and generate intelligence insights.")

# Upload files
uploaded_files = st.file_uploader("Upload files (PDF, PNG, JPG, JPEG)", accept_multiple_files=True)

file_info_list = []
if uploaded_files:
    for uploaded_file in uploaded_files:
        os.makedirs("temp", exist_ok=True)
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_info_list.append({"name": uploaded_file.name, "path": file_path})

tabs = st.tabs(["Uploaded Files", "Intelligence Output", "Generate PDF Report"])

# -------------------------------
# Uploaded Files Tab
# -------------------------------
with tabs[0]:
    st.subheader("Uploaded Files")
    if file_info_list:
        for info in file_info_list:
            st.write(f"- {info['name']}")
    else:
        st.write("No files uploaded yet.")

# -------------------------------
# Intelligence Output Tab
# -------------------------------
consolidated_text = ""
intelligence_data = []

with tabs[1]:
    st.subheader("Document Intelligence")
    if not file_info_list:
        st.write("Upload files to generate intelligence insights.")
    else:
        for info in file_info_list:
            st.markdown(f"### {info['name']}")
            text_content = ""

            # --- PDF Handling ---
            if info['name'].lower().endswith(".pdf"):
                try:
                    with pdfplumber.open(info['path']) as pdf:
                        for page in pdf.pages:
                            page_text = page.extract_text()
                            if page_text:
                                text_content += page_text + "\n"
                    if not text_content.strip():  # Fallback OCR
                        images = convert_from_bytes(open(info['path'], 'rb').read())
                        for img in images:
                            text_content += " ".join(reader.readtext(img, detail=0)) + "\n"
                except Exception as e:
                    st.error(f"Failed PDF processing: {e}")

            # --- Image Handling ---
            elif info['name'].lower().endswith((".png", ".jpg", ".jpeg")):
                try:
                    img = Image.open(info['path'])
                    st.image(img, caption=info['name'], use_column_width=True)
                    text_content = " ".join(reader.readtext(str(info['path']), detail=0))
                except Exception as e:
                    st.error(f"Failed image processing: {e}")

            # Add to consolidated text
            consolidated_text += f"\n--- {info['name']} ---\n{text_content}\n"

            # --- Intelligence Processing ---
            dates = re.findall(r"\d{1,2}/\d{1,2}/\d{4}", text_content)
            amounts = re.findall(r"£\d+(?:\.\d{2})?", text_content)
            names = re.findall(r"[A-Z][a-z]+ [A-Z][a-z]+", text_content)

            intelligence_data.append({
                "file": info['name'],
                "dates": dates,
                "amounts": amounts,
                "names": names
            })

            # Display intelligence
            st.markdown("**Highlights:**")
            st.write("• Dates:", dates or "None")
            st.write("• Amounts:", amounts or "None")
            st.write("• Names:", names or "None")
            st.write("• Behavioural / Vocal Tone Analysis: [Coming soon]")
            st.write("• Anomaly / Discrepancy Detection: [Coming soon]")

# -------------------------------
# Generate PDF Report Tab
# -------------------------------
with tabs[2]:
    st.subheader("Generate Intelligence PDF Report")

    if st.button("Create PDF"):
        if not file_info_list:
            st.warning("Upload files first!")
        else:
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", "B", 16)
            pdf.cell(0, 10, "RBIS Dashboard Intelligence Report", ln=True, align="C")
            pdf.ln(10)

            for data in intelligence_data:
                pdf.set_font("Arial", "B", 12)
                pdf.cell(0, 10, f"File: {data['file']}", ln=True)
                pdf.set_font("Arial", "", 12)
                pdf.multi_cell(0, 8, f"Dates: {', '.join(data['dates']) if data['dates'] else 'None'}")
                pdf.multi_cell(0, 8, f"Amounts: {', '.join(data['amounts']) if data['amounts'] else 'None'}")
                pdf.multi_cell(0, 8, f"Names: {', '.join(data['names']) if data['names'] else 'None'}")
                pdf.multi_cell(0, 8, f"Behavioural / Vocal Tone: [Coming Soon]")
                pdf.multi_cell(0, 8, f"Anomalies / Discrepancies: [Coming Soon]")
                pdf.ln(5)

            # Output PDF to download
            pdf_bytes = pdf.output(dest='S').encode('latin1')
            st.download_button(
                label="Download Intelligence Report PDF",
                data=pdf_bytes,
                file_name="RBIS_Intelligence_Report.pdf",
                mime="application/pdf"
            )
            st.success("PDF generated successfully!")
