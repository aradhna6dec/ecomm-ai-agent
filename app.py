import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import your compiled LangGraph workflow from the previous file
from agent_workflow import app as agent_app

# --- UI Configuration ---
st.set_page_config(page_title="AI E-Commerce Agent", page_icon="🛒", layout="centered")
st.title("🛒 Next-Gen E-Commerce Assistant")
st.markdown("*Powered by Local Llama 3, Qdrant Vector DB, & LangGraph*")

# --- Session State Management ---
# We use Streamlit's session_state to remember the chat history across reruns
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi there! I'm your AI shopping assistant. Are you looking for any specific products today?"}
    ]

# Display all previous messages in the chat UI
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input & AI Processing ---
if prompt := st.chat_input("Ask about gaming monitors, chairs, keyboards..."):
    
    # 1. Display the user's message in the UI immediately
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Convert the Streamlit chat history into LangChain message objects
    langchain_messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))

    # 3. Prepare the state payload for your LangGraph workflow
    agent_state = {
        "messages": langchain_messages,
        "next_step": "",
        "context": ""
    }

    # 4. Invoke the AI and stream the response UI
    with st.chat_message("assistant"):
        with st.spinner("Agent routing & searching database..."):
            try:
                # Call your LangGraph architecture!
                result = agent_app.invoke(agent_state)
                
                # Extract the final AI response from the graph's state
                ai_response = result["messages"][-1].content
                st.markdown(ai_response)
                
                # Save the AI's response to the session history
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            
            except Exception as e:
                st.error(f"An error occurred: {e}. Please ensure Ollama and Qdrant are running.")