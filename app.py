import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image

# --- 1. Bring back your main title ---
st.title("🤖 Multi-LLM Model Orchestrator") # Change this text to whatever your original heading was!

# --- 2. File Uploader Section ---
st.subheader("📎 Attach Files for Review")
uploaded_files = st.file_uploader(
    "Upload PDFs, CSVs, Word Documents, or Screenshots/Images",
    type=["pdf", "csv", "doc", "docx", "png", "jpg", "jpeg"],
    accept_multiple_files=True
)

file_contexts = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        file_extension = uploaded_file.name.split(".")[-1].lower()
        
        try:
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
                file_contexts.append(f"--- File: {uploaded_file.name} (CSV) ---\n{df.to_string(index=False)}")
            elif file_extension == "pdf":
                reader = PdfReader(uploaded_file)
                pdf_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                file_contexts.append(f"--- File: {uploaded_file.name} (PDF) ---\n{pdf_text}")
            elif file_extension in ["doc", "docx"]:
                doc = Document(uploaded_file)
                doc_text = "\n".join([para.text for para in doc.paragraphs if para.text])
                file_contexts.append(f"--- File: {uploaded_file.name} (Word Doc) ---\n{doc_text}")
            elif file_extension in ["png", "jpg", "jpeg"]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=300)
                file_contexts.append(f"--- File: {uploaded_file.name} (Image Uploaded) ---")
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

user_prompt = st.text_area("Enter your prompt:", "e.g., Review the attached document and summarize key insights.")

if st.button("🚀 Run Multi-LLM Engine with Ollama"):
    full_payload = user_prompt + "\n\n" + "\n".join(file_contexts)
    st.write("Processing query with attached context...")
