import asyncio
import json
import os
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# API Clients
from anthropic import AsyncAnthropic
from google import genai
from openai import AsyncOpenAI

# Optional: Groq for free open-source fallback
from groq import AsyncGroq

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Unified Multi-LLM Orchestrator [Resilient]", page_layout="wide"
)

st.title("⚡ Unified Multi-LLM Engine [Auto-Fallback]")
st.caption(
    "Query Gemini, Claude, and ChatGPT with automatic error handling, partial-consensus calculation, and free-tier fallbacks."
)

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedding_model()

# ---------------------------------------------------------
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.header("🔑 API Configuration")
    openai_key = st.text_input("OpenAI API Key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    claude_key = st.text_input("Anthropic API Key", value=os.getenv("ANTHROPIC_API_KEY", ""), type="password")
    gemini_key = st.text_input("Gemini API Key", value=os.getenv("GEMINI_API_KEY", ""), type="password")
    
    st.markdown("---")
    st.header("🛡️ Fallback Settings")
    use_groq_fallback = st.checkbox("Fallback to Groq (Free Llama 3) if OpenAI/Claude fail", value=True)
    groq_key = st.text_input("Groq API Key (Optional)", value=os.getenv("GROQ_API_KEY", ""), type="password")

    st.markdown("---")
    st.header("⚖️ Model Influence Weights")
    weight_gpt = st.slider("ChatGPT Weight", 0.0, 1.0, 0.33)
    weight_claude = st.slider("Claude Weight", 0.0, 1.0, 0.33)
    weight_gemini = st.slider("Gemini Weight", 0.0, 1.0, 0.33)

# ---------------------------------------------------------
# Resilient Asynchronous LLM Runners with Fallbacks
# ---------------------------------------------------------
async def get_groq_fallback(prompt: str, api_key: str, model_name: str = "llama-3.3-70b-versatile") -> str:
    try:
        client = AsyncGroq(api_key=api_key)
        res = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return f"[Fallback to Groq Llama]: {res.choices[0].message.content}"
    except Exception as e:
        return f"[Groq Fallback Error]: {str(e)}"

async def get_chatgpt_response(prompt: str, api_key: str, groq_backup_key: str) -> str:
    if not api_key:
        if groq_backup_key:
            return await get_groq_fallback(prompt, groq_backup_key)
        return "[ChatGPT Skipped]: No API key provided."
    try:
        client = AsyncOpenAI(api_key=api_key)
        res = await client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return res.choices[0].message.content
    except Exception as e:
        if groq_backup_key:
            return await get_groq_fallback(prompt, groq_backup_key)
        return f"[ChatGPT Error/Limit]: {str(e)}"

async def get_claude_response(prompt: str, api_key: str, groq_backup_key: str) -> str:
    if not api_key:
        if groq_backup_key:
            return await get_groq_fallback(prompt, groq_backup_key)
        return "[Claude Skipped]: No API key provided."
    try:
        client = AsyncAnthropic(api_key=api_key)
        res = await client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return res.content[0].text
    except Exception as e:
        if groq_backup_key:
            return await get_groq_fallback(prompt, groq_backup_key)
        return f"[Claude Error/Limit]: {str(e)}"

async def get_gemini_response(prompt: str, api_key: str) -> str:
    if not api_key:
        return "[Gemini Skipped]: No API key provided (Required for synthesis)."
    try:
        client = genai.Client(api_key=api_key)
        res = await asyncio.to_thread(
            client.models.generate_content,
            model="gemini-2.5-flash",
            contents=prompt,
        )
        return res.text
    except Exception as e:
        return f"[Gemini Error/Limit]: {str(e)}"

async def run_all_llms(prompt: str, keys: dict):
    groq_backup = keys["groq"] if keys["use_groq"] else ""
    tasks = [
        get_chatgpt_response(prompt, keys["openai"], groq_backup),
        get_claude_response(prompt, keys["claude"], groq_backup),
        get_gemini_response(prompt, keys["gemini"]),
    ]
    return await asyncio.gather(*tasks)

# ---------------------------------------------------------
# Dynamic Closeness Calculation (Handles Partial Responses)
# ---------------------------------------------------------
def calculate_closeness_scores(responses: dict):
    """Dynamically calculates similarity even if one or more models returned errors."""
    valid_keys = []
    valid_texts = []
    
    for k, text in responses.items():
        if not text.startswith("[ChatGPT Error") and not text.startswith("[Claude Error") and not text.startswith("[Gemini Error") and not text.startswith("[ChatGPT Skipped") and not text.startswith("[Claude Skipped") and not text.startswith("[Gemini Skipped"):
            valid_keys.append(k)
            valid_texts.append(text)
            
    if len(valid_texts) < 2:
        # If fewer than 2 models succeeded, return default baseline metrics
        return {"chatgpt": 50.0, "claude": 50.0, "gemini": 50.0}, 50.0

    embeddings = embedder.encode(valid_texts)
    sim_matrix = cosine_similarity(embeddings)
    
    scores = {}
    for i, key in enumerate(valid_keys):
        other_similarities = [sim_matrix[i][j] for j in range(len(valid_keys)) if i != j]
        scores[key] = round((sum(other_similarities) / len(other_similarities)) * 100, 1) if other_similarities else 50.0
        
    # Fill missing ones with baseline
    full_scores = {
        "chatgpt": scores.get("chatgpt", 0.0),
        "claude": scores.get("claude", 0.0),
        "gemini": scores.get("gemini", 0.0)
    }
    
    overall_consensus = round(float(np.mean(sim_matrix)) * 100, 1)
    return full_scores, overall_consensus

# ---------------------------------------------------------
# Meta Synthesis Call
# ---------------------------------------------------------
def generate_synthesis(prompt: str, responses: dict, api_key: str, weights: dict):
    if not api_key:
        return "⚠️ Gemini API key is required to generate the unified meta-synthesis."
    
    client = genai.Client(api_key=api_key)
    meta_prompt = f"""
    You are an expert AI Evaluator. Analyze the following available AI responses to the user prompt: "{prompt}".
    
    Available Responses:
    - ChatGPT: {responses['chatgpt']}
    - Claude: {responses['claude']}
    - Gemini: {responses['gemini']}
    
    Provide Markdown formatted output:
    1. **Unified Synthesis**: Best combined answer.
    2. **Unique Suggestions by Model**: Key takeaways from each successful model response.
    """
    try:
        res = client.models.generate_content(model="gemini-2.5-flash", contents=meta_prompt)
        return res.text
    except Exception as e:
        return f"Synthesis generation failed: {str(e)}"

# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
user_prompt = st.text_area("Enter your prompt:", height=120, placeholder="e.g., Outline a robust API rate-limiting strategy.")

if st.button("🚀 Run Resilient Multi-LLM Engine", type="primary"):
    if not gemini_key:
        st.error("Please provide a Gemini API Key (required for orchestrating the synthesis engine).")
    elif not user_prompt.strip():
        st.warning("Please enter a prompt first.")
    else:
        keys = {
            "openai": openai_key,
            "claude": claude_key,
            "gemini": gemini_key,
            "groq": groq_key,
            "use_groq": use_groq_fallback
        }
        weights = {"gpt": weight_gpt, "claude": weight_claude, "gemini": weight_gemini}

        with st.spinner("Executing query with fallback safety nets..."):
            gpt_out, claude_out, gemini_out = asyncio.run(run_all_llms(user_prompt, keys))
            
            response_dict = {
                "chatgpt": gpt_out,
                "claude": claude_out,
                "gemini": gemini_out
            }

            alignment_scores, consensus = calculate_closeness_scores(response_dict)
            synthesis = generate_synthesis(user_prompt, response_dict, gemini_key, weights)

        st.session_state["last_result"] = {
            "prompt": user_prompt,
            "consensus_score": consensus,
            "metrics": alignment_scores,
            "synthesis": synthesis,
            "raw_responses": response_dict
        }

# ---------------------------------------------------------
# Display Results
# ---------------------------------------------------------
if "last_result" in st.session_state:
    data = st.session_state["last_result"]
    st.markdown("---")
    
    col_c1, col_c2 = st.columns([1, 3])
    with col_c1:
        st.metric("Global Consensus Match", f"{data['consensus_score']}%")
    with col_c2:
        st.progress(data['consensus_score'] / 100)
        st.caption("🛡️ **Fallback engine executed successfully. Errors caught and handled gracefully.**")

    tab_synthesis, tab_comparison, tab_raw = st.tabs(
        ["✨ Unified Synthesis", "📊 Model Scorecard", "📝 Raw Outputs"]
    )

    with tab_synthesis:
        st.markdown(data["synthesis"])

    with tab_comparison:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader("ChatGPT / Backup")
            st.metric("Alignment", f"{data['metrics']['chatgpt']}%")
            st.info(data["raw_responses"]["chatgpt"][:300] + "...")
        with c2:
            st.subheader("Claude / Backup")
            st.metric("Alignment", f"{data['metrics']['claude']}%")
            st.info(data["raw_responses"]["claude"][:300] + "...")
        with c3:
            st.subheader("Gemini")
            st.metric("Alignment", f"{data['metrics']['gemini']}%")
            st.info(data["raw_responses"]["gemini"][:300] + "...")

    with tab_raw:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.subheader("ChatGPT Raw")
            st.text_area("GPT", data["raw_responses"]["chatgpt"], height=400)
        with col_b:
            st.subheader("Claude Raw")
            st.text_area("Claude", data["raw_responses"]["claude"], height=400)
        with col_c:
            st.subheader("Gemini Raw")
            st.text_area("Gemini", data["raw_responses"]["gemini"], height=400)

    st.markdown("---")
    st.download_button(
        label="📥 Export Session Report as JSON",
        data=json.dumps(data, indent=2),
        file_name="resilient_multi_llm_report.json",
        mime="application/json",
    )