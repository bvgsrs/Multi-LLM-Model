import streamlit as st
import pandas as pd
from pypdf import PdfReader
from docx import Document
from PIL import Image
import concurrent.futures

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
# SIDEBAR: Configuration & Keys
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
    
    st.subheader("Select Active Models")
    use_groq = st.checkbox("Groq (Llama 3)", value=True)
    use_gemini = st.checkbox("Gemini (Pro)", value=True)
    use_gpt = st.checkbox("GPT-4o (Paid)", value=False)
    use_claude = st.checkbox("Claude 3.5 Sonnet (Paid)", value=False)
    
    st.divider()
    
    st.subheader("📌 Manage Prompt Index")
    with st.form("add_index_form", clear_on_submit=True):
        custom_prompt_text = st.text_area("Add new prompt to index:")
        submitted = st.form_submit_button("Save to Index")
        if submitted and custom_prompt_text.strip():
            new_idx = len(st.session_state.prompt_library) + 1
            st.session_state.prompt_library[new_idx] = custom_prompt_text.strip()
            st.success(f"Saved as Index [{new_idx}]!")
            st.rerun()

# ==========================================
# MAIN PAGE: Title & File Uploads
# ==========================================
st.title("🤖 Multi-LLM Model Orchestrator & Synthesizer")

# Visible Index Quick-Reference Bar on the main canvas
with st.expander("📌 View Quick-Index Library (Type number in chat)", expanded=True):
    cols = st.columns(len(st.session_state.prompt_library))
    for i, (idx, p_text) in enumerate(st.session_state.prompt_library.items()):
        with cols[i]:
            st.markdown(f"**Index [{idx}]**")
            st.caption(p_text)

st.subheader("📎 Attach Files for Review (Optional)")
uploaded_files = st.file_uploader(
    "Upload PDFs, CSVs, Word Documents, or Images",
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
# CHAT INTERFACE & CONCURRENT SYNTHESIS
# ==========================================
st.subheader("💬 Chat & Consensus Output")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

raw_input = st.chat_input("Type a message or enter an index number (e.g., 1)...")

if raw_input:
    clean_input = raw_input.strip()
    
    # Resolve Index Number
    if clean_input.isdigit() and int(clean_input) in st.session_state.prompt_library:
        resolved_text = st.session_state.prompt_library[int(clean_input)]
        user_prompt = f"**[Index Resolved: #{clean_input}]** {resolved_text}"
        st.toast(f"Resolved Index [{clean_input}] to: '{resolved_text}'", icon="📌")
    else:
        user_prompt = f"**[Custom Input]** {raw_input}"

    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.status("Running parallel models and synthesizing consensus...", expanded=True) as status:
            
            if file_contexts:
                full_payload = f"Context Documents:\n" + "\n".join(file_contexts) + f"\n\nUser Request: {user_prompt}"
            else:
                full_payload = user_prompt
            
            # 1. Parallel Workers with stable, active model identifiers
            def query_groq():
                try:
                    client = Groq(api_key=groq_key)
                    res = client.chat.completions.create(
                        model="llama3-70b-8192",
                        messages=[{"role": "user", "content": full_payload}]
                    )
                    return "Groq (Llama 3)", res.choices[0].message.content
                except Exception as e:
                    return "Groq (Llama 3)", f"Error: {e}"

            def query_gemini():
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-pro")
                    res = model.generate_content(full_payload)
                    return "Gemini (Pro)", res.text
                except Exception as e:
                    return "Gemini (Pro)", f"Error: {e}"

            def query_gpt():
                try:
                    client = OpenAI(api_key=openai_key)
                    res = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": full_payload}]
                    )
                    return "GPT-4o", res.choices[0].message.content
                except Exception as e:
                    return "GPT-4o", f"Error: {e}"

            def query_claude():
                try:
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    res = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": full_payload}]
                    )
                    return "Claude 3.5 Sonnet", res.content[0].text
                except Exception as e:
                    return "Claude 3.5 Sonnet", f"Error: {e}"

            tasks = []
            if use_groq and groq_key:
                tasks.append(query_groq)
            if use_gemini and gemini_key:
                tasks.append(query_gemini)
            if use_gpt and openai_key:
                tasks.append(query_gpt)
            if use_claude and anthropic_key:
                tasks.append(query_claude)

            model_responses = {}
            
            # Execute concurrently via threads
            if tasks:
                st.write(f"Launching {len(tasks)} model threads concurrently...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_model = {executor.submit(task): task for task in tasks}
                    for future in concurrent.futures.as_completed(future_to_model):
                        name, result = future.result()
                        model_responses[name] = result
                        st.write(f"Completed initial pass: {name}")
            else:
                st.warning("No models selected or API keys missing.")

            # 2. Stage 2: Cross-Synthesis / Refinement Pass
            consolidated_output = ""
            valid_responses = {k: v for k, v in model_responses.items() if not v.startswith("Error:")}
            
            if len(valid_responses) > 0:
                st.write("Synthesizing and cross-refining outputs into a final consolidated version...")
                
                synthesis_prompt = f"""You are the Master Orchestrator AI. 
The user asked: '{user_prompt}'

Below are the independent responses generated by multiple AI models. Compare them, reconcile any contradictions, merge their best insights, and write a single, highly refined, comprehensive consolidated response.

Individual Model Outputs:
"""
                for m_name, m_text in valid_responses.items():
                    synthesis_prompt += f"\n--- [{m_name}] ---\n{m_text}\n"

                # Use Groq or Gemini as the synthesizer engine
                try:
                    if groq_key and use_groq:
                        synth_client = Groq(api_key=groq_key)
                        synth_res = synth_client.chat.completions.create(
                            model="llama3-70b-8192",
                            messages=[{"role": "user", "content": synthesis_prompt}]
                        )
                        consolidated_output = synth_res.choices[0].message.content
                    elif gemini_key and use_gemini:
                        genai.configure(api_key=gemini_key)
                        synth_model = genai.GenerativeModel("gemini-pro")
                        synth_res = synth_model.generate_content(synthesis_prompt)
                        consolidated_output = synth_res.text
                    else:
                        # Fallback to whatever valid response we have
                        consolidated_output = list(valid_responses.values())[0]
                except Exception as e:
                    consolidated_output = f"Consolidation synthesis skipped due to error: {e}"
            else:
                consolidated_output = "⚠️ Could not generate a consolidated synthesis because all active model queries encountered errors."

            status.update(label="Pipeline execution complete!", state="complete", expanded=False)
        
        # Format and display the final compiled markdown
        final_output_markdown = ""
        
        if consolidated_output:
            final_output_markdown += f"### 🏆 Final Consolidated & Refined Synthesis\n\n{consolidated_output}\n\n---\n\n"

        if model_responses:
            final_output_markdown += "### 🌐 Individual Model Responses:\n\n"
            for name, text in model_responses.items():
                final_output_markdown += f"#### **{name}**\n{text}\n\n---\n"
        else:
            final_output_markdown = "⚠️ **Warning:** No models were queried. Check your sidebar settings."

        st.markdown(final_output_markdown)
        st.session_state.messages.append({"role": "assistant", "content": final_output_markdown})
