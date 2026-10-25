model_responses = {}

            # 1. OpenAI Query
            if use_gpt and openai_key:
                st.write("Querying GPT-4o...")
                try:
                    client = OpenAI(api_key=openai_key)
                    res = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": full_payload}]
                    )
                    model_responses["GPT-4o"] = res.choices[0].message.content
                except Exception as e:
                    model_responses["GPT-4o"] = f"Error: {e}"

            # 2. Anthropic Query
            if use_claude and anthropic_key:
                st.write("Querying Claude 3.5 Sonnet...")
                try:
                    client = anthropic.Anthropic(api_key=anthropic_key)
                    res = client.messages.create(
                        model="claude-3-5-sonnet-20241022",
                        max_tokens=1024,
                        messages=[{"role": "user", "content": full_payload}]
                    )
                    model_responses["Claude 3.5 Sonnet"] = res.content[0].text
                except Exception as e:
                    model_responses["Claude 3.5 Sonnet"] = f"Error: {e}"

            # 3. Gemini Query
            if use_gemini and gemini_key:
                st.write("Querying Gemini...")
                try:
                    genai.configure(api_key=gemini_key)
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    res = model.generate_content(full_payload)
                    model_responses["Gemini"] = res.text
                except Exception as e:
                    model_responses["Gemini"] = f"Error: {e}"

            # 4. Groq / Llama Query (Updated to active identifier)
            if use_llama and groq_key:
                st.write("Querying Llama via Groq...")
                try:
                    client = Groq(api_key=groq_key)
                    res = client.chat.completions.create(
                        model="llama-3.1-70b-versatile",
                        messages=[{"role": "user", "content": full_payload}]
                    )
                    model_responses["Llama 3 (Groq)"] = res.choices[0].message.content
                except Exception as e:
                    model_responses["Llama 3 (Groq)"] = f"Error: {e}"
