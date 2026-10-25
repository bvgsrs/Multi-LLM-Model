import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image

# --- Page Configuration ---
st.set_page_config(page_title="Multi-LLM Model Orchestrator", layout="wide")

# ==========================================
# SIDEBAR: Models, Weights, and API Keys
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration & Models")
    
    # 1. API Keys Input with automatic Secrets fallback
    st.subheader("API Keys & Links")
    
    # Check server secrets first, default to empty string if not found
    default_openai = st.secrets.get("OPENAI_API_KEY", "")
    default_anthropic = st.secrets.get("ANTHROPIC_API_KEY", "")
    default_gemini = st.secrets.get("GEMINI_API_KEY", "")
    default_groq = st.secrets.get("GROQ_KEY", "")
    
    openai_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
    st.markdown("[Get OpenAI API Key](https://openai.com)")
    
    anthropic_key = st.text_input("Anthropic API Key", value=default_anthropic, type="password")
    st.markdown("[Get Anthropic API Key](https://www.anthropic.com)")
    
    gemini_key = st.text_input("Gemini API Key", value=default_gemini, type="password")
    st.markdown("[Get Gemini API Key](https://ai.google.dev)")
    
    groq_key = st.text_input("Groq API Key", value=default_groq, type="password")
    st.markdown("[Get Groq API Key](https://ollama.com)")
    
    st.divider()
    
    # 2. Model Selections
    st.subheader("Select Models")
    use_gpt = st.checkbox("GPT-4o ([OpenAI Site](https://openai.com))", value=True)
    use_claude = st.checkbox("Claude 3.5 Sonnet ([Anthropic Site](https://www.anthropic.com))", value=True)
    use_gemini = st.checkbox("Gemini ([Google AI Site](https://ai.google.dev))", value=True)
    use_llama = st.checkbox("Llama 3 / Ollama ([Ollama Site](https://ollama.com))", value=True)
    
    st.divider()
    
    # 3. Consensus / Model Weights
    st.subheader("Model Weights")
    weight_gpt = st.slider("GPT Weight", 0.0, 1.0, 0.25)
    weight_claude = st.slider("Claude Weight", 0.0, 1.0, 0.25)
    weight_gemini = st.slider("Gemini Weight", 0.0, 1.0, 0.25)
    weight_llama = st.slider("Llama Weight", 0.0, 1.0, 0.25)

# ==========================================
# MAIN PAGE: Title, File Uploads, & Prompts
# ==========================================
st.title("🤖 Multi-LLM Model Orchestrator")

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
            # Handle CSV Files
            if file_extension == "csv":
                df = pd.read_csv(uploaded_file)
                file_contexts.append(f"--- File: {uploaded_file.name} (CSV) ---\n{df.to_string(index=False)}")
            
            # Handle PDF Files
            elif file_extension == "pdf":
                reader = PdfReader(uploaded_file)
                pdf_text = "".join([page.extract_text() for page in reader.pages if page.extract_text()])
                file_contexts.append(f"--- File: {uploaded_file.name} (PDF) ---\n{pdf_text}")
            
            # Handle Word Documents (.docx)
            elif file_extension in ["doc", "docx"]:
                doc = Document(uploaded_file)
                doc_text = "\n".join([para.text for para in doc.paragraphs if para.text])
                file_contexts.append(f"--- File: {uploaded_file.name} (Word Doc) ---\n{doc_text}")
            
            # Handle Images / Screenshots
            elif file_extension in ["png", "jpg", "jpeg"]:
                image = Image.open(uploaded_file)
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", width=300)
                file_contexts.append(f"--- File: {uploaded_file.name} (Image Uploaded) ---")
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

# Prompt Input Area
user_prompt = st.text_area("Enter your prompt:", "e.g., Review the attached document and summarize key insights.")

if st.button("🚀 Run Multi-LLM Engine"):
    # Combine prompt text and contents extracted from documents/files
    full_payload = user_prompt + "\n\n" + "\n".join(file_contexts)
    st.write("Processing query with attached context across selected models...")
