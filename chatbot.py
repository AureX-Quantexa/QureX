import streamlit as st
import os
import json

try:
    import openai
except ImportError:
    openai = None

def render_chat_interface(context_data: dict = None) -> None:
    """Render the Interactive AI Co-Pilot chat on a standalone page."""
    st.markdown("## 🤖 COPS Assistant")
    st.caption("Ask me to explain the traffic data or any concepts simply.")

    api_key = "rc_30a7d546f5ef59472d9772924d8c4b1a3ac069af344551cfa828979ae8240961"
    
    if openai is None:
        st.error("OpenAI library not installed.")
        return
        
    # Use Featherless API Base URL
    client = openai.OpenAI(
        api_key=api_key, 
        base_url="https://api.featherless.ai/v1"
    )

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Load live context data
    c_delay, q_delay, q_throughput, q_fuel, q_co2 = 0, 0, 0, 0, 0
    anomaly = "No"
    try:
        with open("live_sumo_state.json", "r") as f:
            data = json.load(f)
            c_delay = data.get("kpi", {}).get("c_delay", 0)
            q_delay = data.get("kpi", {}).get("q_delay", 0)
            q_throughput = data.get("kpi", {}).get("q_throughput", 0)
            q_fuel = data.get("kpi", {}).get("q_fuel", 0)
            q_co2 = data.get("kpi", {}).get("q_co2", 0)
            q_state = data.get("q_state", {})
            if any(d.get("queue", 0) > 15 for d in q_state.values()):
                anomaly = "Yes (Queue > 15)"
    except Exception:
        pass

    # Create dynamic system prompt with LIVE data
    system_prompt = f"""
You are the COPS Assistant, an expert traffic AI. 
Explain the dashboard numbers comparing Classical vs Quantum routing in extremely simple terms (like explaining to a school kid).
Current Live Data:
- Classical Delay: {c_delay:.1f}s
- Quantum Delay: {q_delay:.1f}s (Lower is better)
- Quantum Throughput: {q_throughput:.0f} veh/h
- Fuel Saved: {q_fuel:.2f} gal
- CO2 Reduced: {q_co2:.2f} kg
- Anomaly / Traffic Jam detected: {anomaly}

Make your answers short, professional, and easy to understand. Do NOT use emojis.
"""

    # Accept user input using chat_input
    prompt = st.chat_input("Ask a question about the data or traffic concepts...")
    if prompt:
        # Display user message in chat message container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Build messages list for API
        api_messages = [{"role": "system", "content": system_prompt}]
        api_messages.extend(st.session_state.messages)
        
        # Call OpenAI API
        with chat_container:
            with st.chat_message("assistant"):
                    try:
                        response = client.chat.completions.create(
                            model="Qwen/Qwen2-7B-Instruct",
                            messages=api_messages,
                            max_tokens=250,
                            temperature=0.7
                        )
                        assistant_response = response.choices[0].message.content
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    except Exception as e:
                        st.error(f"Error communicating with AI: {e}")
                        
    # If there's no chat history, provide a button to auto-explain the current data
    if len(st.session_state.messages) == 0:
        if st.button("Generate Explanation of Current Data"):
            with chat_container:
                with st.chat_message("user"):
                    st.markdown("Please explain the current traffic data simply.")
                st.session_state.messages.append({"role": "user", "content": "Please explain the current traffic data simply."})
                
                with st.chat_message("assistant"):
                    try:
                        api_messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": "Please explain the current traffic data simply."}]
                        response = client.chat.completions.create(
                            model="Qwen/Qwen2-7B-Instruct",
                            messages=api_messages,
                            max_tokens=250,
                            temperature=0.7
                        )
                        assistant_response = response.choices[0].message.content
                        st.markdown(assistant_response)
                        st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                    except Exception as e:
                        st.error(f"Error communicating with AI: {e}")
