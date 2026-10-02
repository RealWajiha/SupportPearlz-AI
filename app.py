import streamlit as st

st.set_page_config(page_title="SupportPearlz AI", layout="wide")

# 1. Session state check karein
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""

# 2. Agar API Key nahi mili, toh sirf Center Screen dikhayein
if not st.session_state["openai_api_key"]:
    # Columns ki madad se box ko center mein layein
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.write("")
        st.write("")
        st.title("🤖 SupportPearlz AI")
        st.subheader("Welcome! Please enter your OpenAI API key to access the app.")
        
        api_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        
        if st.button("Enter App", use_container_width=True):
            if api_input.strip():
                st.session_state["openai_api_key"] = api_input
                st.rerun()  # Page refresh karke main app dikhayein
            else:
                st.error("Please enter a valid OpenAI API Key.")
    
    # Iske aage ka koi bhi UI/Sidebar execute nahi hoga jab tak key na daali jaye
    st.stop()

# 3. Main App Interface (API Key milne ke baad yeh run hoga)
openai_api_key = st.session_state["openai_api_key"]

# Aapka baaki ka Sidebar aur Main Chat interface code yahan niche aayega...
