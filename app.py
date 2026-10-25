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
# SESSION STATE: Prompt Indexing Library
# ==========================================
if "prompt_library" not in st.session_state:
    st.session_state.prompt_library = {
        1: "Summarize the key takeaways and action items.",
        2: "Review the attached context for any logical errors or risks.",
        3: "Write a professional email response based on this input."
    }

# ==========================================
# SIDEBAR: Configuration, Models, & Indexing
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration & Models")
    
    st.subheader("API Keys")
    default_openai = st.secrets.get("OPENAI_API_KEY", "")
    default_anthropic = st.secrets.get("ANTHROPIC_API_KEY", "")
    default_gemini = st.secrets.get("GEMINI_API_KEY", "")
    default_groq = st.secrets.get("GROQ_KEY", "")
    
    openai_key = st.text_input("OpenAI API Key", value=default_openai, type="password")
    anthropic_key = st.text_input("Anthropic API Key", value=default_anthropic, type="password")
    gemini_key = st.text_input("Gemini API Key", value=default_gemini, type="password")
    groq_key = st.text_input("Groq API Key", value=default_groq, type="password")
    
    st.divider()
    
    st.subheader("Select Models")
    use_llama = st.checkbox("Groq (Llama 3.1)", value=True)
    use_gemini = st.checkbox("Gemini (1.5 Flash)", value=True)
    use_gpt = st.checkbox("GPT-4o (Paid)", value=False)
    use_claude = st.checkbox("Claude 3.5 Sonnet (Paid)", value=False)
    
    st.divider()
    
    st.subheader("📌 Prompt Indexing Library")
    st.markdown("Use index numbers (e.g., `1`) in the chat box to send saved prompts instantly.")
    
    new_idx = len(st.session_state.prompt_library) + 1
    with st.form("add_index_form", clear_on_submit=True):
        custom_prompt_text = st.text_area("Prompt text to index:")
        submitted = st.form_submit_button("Save to Index")
        if submitted and custom_prompt_text.strip():
            st.session_state.prompt_library[new_idx] = custom_prompt_text.strip()
            st.success(f"Saved as Index [{new_idx}]!")
            st.rerun()
            
    st.write("**Current Index Mapping:**")
    for idx, p_text in st.session_state.prompt_library.items():
        st.code(f"[{idx}]: {p_text}", language="text")

# ==========================================
# MAIN PAGE: Title & File Uploads
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
# CHAT INTERFACE & EXECUTION LOGIC
# ==========================================
st.subheader("💬 Chat & Consensus Output")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

raw_input = st.chat_input("Type a message, question, or index number (e.g., 1)...")

if raw_input:
    clean_input = raw_input.strip()
    # Check if user entered an index number (e.g. "1", "2")
    if clean_input.isdigit() and int(clean_input) in st.session_state.prompt_library:
        user_prompt = st.session_state.prompt_library[int(clean_input)]
        st.toast(f"Resolved Index [{clean_input}] to: '{user_prompt}'", icon="📌")
    else:
        user_prompt = raw_input

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.status("Running multi-model consensus engine...", expanded=True) as status:
            
            if file_contexts:
                st.write("Including attached document context...")
                full_payload = f"Context Documents:\n" + "\n".join(file_contexts) + f"\n\nUser Question: {user_prompt}"
            else:
                full_payload = user_prompt
            
            model_responses = {}
            
            def call_gemini(key, prompt):
                genai.configure(api_key=key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)
                return response.text

            # Prioritize fast, working free-tier models first so it doesn't lag
            models_to_try = [
                ("Groq (Llama 3.1)", use_llama, groq_key, lambda: Groq(api_key=groq_key).chat.completions.create(model="llama-3.1-8b-instant", messages=[{"role": "user", "content": full_payload}]).choices[0].message.content),
                ("Gemini (1.5 Flash)", use_gemini, gemini_key, lambda: call_gemini(gemini_key, full_payload)),
                ("GPT-4o", use_gpt, openai_key, lambda: OpenAI(api_key=openai_key).chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": full_payload}]).choices[0].message.content),
                ("Claude 3.5 Sonnet", use_claude, anthropic_key, lambda: anthropic.Anthropic(api_key=anthropic_key).messages.create(model="claude-3-5-sonnet-20241022", max_tokens=1024, messages=[{"role": "user", "content": full_payload}]).content[0].text)
            ]

            for name, is_selected, key, api_call in models_to_try:
                if is_selected and key:
                    st.write(f"Attempting query with {name}...")
                    try:
                        res_text = api_call()
                        model_responses[name] = res_text
                        st.success(f"Successfully received response from {name}!")
                    except Exception as e:
                        error_msg = str(e)
                        model_responses[name] = f"Error: {error_msg}"
                        st.warning(f"{name} failed.")

            status.update(label="Execution sequence complete!", state="complete", expanded=False)
        
        if model_responses:
            final_output_markdown = "### 🌐 Multi-LLM Execution Results\n\n"
            for name, text in model_responses.items():
                final_output_markdown += f"#### **{name}**\n{text}\n\n---\n"
        else:
            final_output_markdown = "⚠️ **Warning:** No models were successfully queried. Please check your selected models and API keys."

        st.markdown(final_output_markdown)
        st.session_state.messages.append({"role": "assistant", "content": final_output_markdown})
