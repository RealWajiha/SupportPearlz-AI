import streamlit as st
import os

# 1. Page Configuration (Sab se pehle rehna chahiye)
st.set_page_config(page_title="SupportPearlz AI", page_icon="🤖", layout="wide")

# 2. Session State Initialization
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

# 3. Center Screen API Key Input Box
if not st.session_state["openai_api_key"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.write("")
        st.title("🤖 SupportPearlz AI")
        st.subheader("Welcome! Enter your OpenAI API Key to start.")
        
        api_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        
        if st.button("Enter App", use_container_width=True):
            if api_input.strip():
                st.session_state["openai_api_key"] = api_input.strip()
                st.rerun()  # Script refresh karke main UI load karega
            else:
                st.error("Please enter a valid OpenAI API Key.")
    
    st.stop()  # Iske niche wala UI hide rahega jab tak key na daali jaye

# ------------------------------------------------------------------
# 4. MAIN APP INTERFACE (Yeh tab chalega jab Key enter ho chuki ho)
# ------------------------------------------------------------------

openai_api_key = st.session_state["openai_api_key"]

# --- SIDEBAR UI ---
with st.sidebar:
    st.header("⚙️ Settings & Indexing")
    st.success("API Key Active ✅")
    
    # Key reset / Logout button
    if st.button("Change API Key"):
        st.session_state["openai_api_key"] = ""
        st.rerun()
        
    st.subheader("Vector Database Management")
    if st.button("🔨 Rebuild Vector Index"):
        # Aapka build_vector_index function yahan call hoga
        st.info("Rebuilding index...")

# --- MAIN PAGE CHAT INTERFACE ---
st.title("🤖 SupportPearlz Customer Support Agent")
st.caption("LangChain-powered RAG Knowledge Agent")

# Aapka chat input aur history render karne ka code yahan aayega
user_query = st.chat_input("Ask a question about Pearlz products...")

if user_query:
    st.chat_message("user").write(user_query)
    # Aapka RAG response function yahan call hoga
