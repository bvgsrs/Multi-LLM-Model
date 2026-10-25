import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image

# Import official model libraries
from openai import OpenAI
import anthropic
import google.generativeai as genai
from groq import Groq

# --- Page Configuration ---
st.set_page_config(page_title="Multi-LLM Model Orchestrator", layout="wide")

# ==========================================
# SIDEBAR: Models, Weights, and API Keys
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration & Models")
    
    st.subheader("API Keys & Links")
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
    
    st.subheader("Select Models")
    use_gpt = st.checkbox("GPT-4o ([OpenAI Site](https://openai.com))", value=True)
    use_claude = st.checkbox("Claude 3.5 Sonnet ([Anthropic Site](https://www.anthropic.com))", value=True)
    use_gemini = st.checkbox("Gemini ([Google AI Site](https://ai.google.dev))", value=True)
    use_llama = st.checkbox("Llama 3 / Groq ([Ollama Site](https://ollama.com))", value=True)
    
    st.divider()
    
    st.subheader("Model Weights")
    weight_gpt = st.slider("GPT Weight", 0.0, 1.0, 0.25)
    weight_claude = st.slider("Claude Weight", 0.0, 1.0, 0.25)
    weight_gemini = st.slider("Gemini Weight", 0.0, 1.0, 0.25)
    weight_llama = st.slider("Llama Weight", 0.0, 1.0, 0.25)

# ==========================================
# MAIN PAGE: Title & File Uploads (Optional)
# ==========================================
st.title("🤖 Multi-LLM Model Orchestrator")

st.subheader("📎 Attach Files for Review (Optional)")
uploaded_files = st.file_uploader(
    "Upload PDFs, CSVs, Word Documents, or Screenshots/Images for context",
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

st.divider()

# ==========================================
# CHAT INTERFACE & MULTI-LLM EXECUTION LOGIC
# ==========================================
st.subheader("💬 Chat & Consensus Output")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_prompt := st.chat_input("Type a message or ask a question about your files..."):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.status("Running multi-model consensus engine...", expanded=True) as status:
            
            # Prepare payload
            if file_contexts:
                st.write("Including attached document context...")
                full_payload = f"Context Documents:\n" + "\n".join(file_contexts) + f"\n\nUser Question: {user_prompt}"
            else:
                full_payload = user_prompt
            
            model_responses = {}
            successful_response = None
            
            # Define models and execution lambdas
            models_to_try = [
                ("GPT-4o", use_gpt, openai_key, lambda: OpenAI(api_key=openai_key).chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": full_payload}]).choices[0].message.content),
                ("Claude 3.5 Sonnet", use_claude, anthropic_key, lambda: anthropic.Anthropic(api_key=anthropic_key).messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[{"role": "user", "content": full_payload}]).content[0].text),
                ("Gemini", use_gemini, gemini_key, lambda: (genai.configure(api_key=gemini_key), genai.GenerativeModel("gemini-1.5-flash").generate_content(full_payload).text)[1]),
                ("Llama 3 (Groq)", use_llama, groq_key, lambda: Groq(api_key=groq_key).chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": full_payload}]).choices[0].message.content)
            ]

            for name, is_selected, key, api_call in models_to_try:
                if is_selected and key:
                    st.write(f"Attempting query with {name}...")
                    try:
                        res_text = api_call()
                        model_responses[name] = res_text
                        successful_response = res_text
                        st.success(f"Successfully received response from {name}!")
                    except Exception as e:
                        error_msg = str(e)
                        model_responses[name] = f"Error: {error_msg}"
                        st.warning(f"{name} failed.")

            status.update(label="Execution sequence complete!", state="complete", expanded=False)
        
        # Format and display results
        if model_responses:
            final_output_markdown = "### 🌐 Multi-LLM Execution & Fallback Results\n\n"
            for name, text in model_responses.items():
                final_output_markdown += f"#### **{name}**\n{text}\n\n---\n"
        else:
            final_output_markdown = "⚠️ **Warning:** No models were successfully queried. Check your API keys and balances."

        st.markdown(final_output_markdown)
        st.session_state.messages.append({"role": "assistant", "content": final_output_markdown})
