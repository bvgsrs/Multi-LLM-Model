model_responses = {}
            successful_response = None

            # Define a fallback order of models to try sequentially if one fails
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
                        break # Stop trying once we get a successful fallback response!
                    except Exception as e:
                        error_msg = str(e)
                        model_responses[name] = f"Error: {error_msg}"
                        st.warning(f"{name} failed, trying next available model...")

            status.update(label="Fallback sequence complete!", state="complete", expanded=False)
        
        # Format and display results
        if model_responses:
            final_output_markdown = "### 🌐 Multi-LLM Execution & Fallback Results\n\n"
            for name, text in model_responses.items():
                final_output_markdown += f"#### **{name}**\n{text}\n\n---\n"
        else:
            final_output_markdown = "⚠️ **Warning:** No models were successfully queried. Check your API keys and balances."

        st.markdown(final_output_markdown)
        st.session_state.messages.append({"role": "assistant", "content": final_output_markdown})
