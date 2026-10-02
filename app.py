import os
import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="SupportPearlz AI", page_icon="🤖", layout="wide")

# 2. Session State Setup
if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

# 3. Center Screen - API Key Input Box
if not st.session_state["openai_api_key"]:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.write("")
        st.write("")
        st.title("🤖 SupportPearlz AI")
        st.subheader("Welcome! Please enter your OpenAI API key to access the app.")
        
        api_input = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
        if st.button("Enter App", use_container_width=True):
            if api_input.strip().startswith("sk-"):
                st.session_state["openai_api_key"] = api_input.strip()
                st.rerun()
            else:
                st.error("Please enter a valid OpenAI API Key starting with sk-")
    st.stop()

# 4. Helper Function: Build or Load Vector Store
def load_or_build_vector_store(api_key):
    if st.session_state["vector_store"] is None:
        kb_dir = "./data/knowledge_base"
        if not os.path.exists(kb_dir):
            return None
        
        loader = DirectoryLoader(kb_dir, glob="**/*.*")
        raw_docs = loader.load()
        if not raw_docs:
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(raw_docs)
        
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        st.session_state["vector_store"] = FAISS.from_documents(chunks, embeddings)
        
    return st.session_state["vector_store"]

# ------------------------------------------------------------------
# 5. MAIN APP INTERFACE (API Key Enter hone ke baad)
# ------------------------------------------------------------------
api_key = st.session_state["openai_api_key"]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings & Indexing")
    st.success("API Key Active ✅")
    
    if st.button("Change API Key"):
        st.session_state["openai_api_key"] = ""
        st.session_state["vector_store"] = None
        st.session_state["messages"] = []
        st.rerun()
        
    st.subheader("Vector Database Management")
    if st.button("🔨 Rebuild Vector Index"):
        st.session_state["vector_store"] = None
        with st.spinner("Rebuilding Index..."):
            vs = load_or_build_vector_store(api_key)
            if vs:
                st.success("Index rebuilt successfully!")
            else:
                st.error("No documents found in `./data/knowledge_base`.")

# Main Chat Header
st.title("🤖 SupportPearlz Customer Support Agent")
st.caption("LangChain-powered RAG Knowledge Agent")

# Load Vector Store
vector_store = load_or_build_vector_store(api_key)

if vector_store is None:
    st.warning("⚠️ No documents found in `./data/knowledge_base`. Please add KB files and click 'Rebuild Vector Index'.")

# Display Chat History
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Chat Input & Response Logic
if user_query := st.chat_input("Ask a question about Pearlz products or knowledge base..."):
    # User message display karein
    st.session_state["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Answer generate karein
    if vector_store is None:
        answer = "Knowledge base empty hai. Pehle file upload karein aur Index Rebuild karein."
        with st.chat_message("assistant"):
            st.error(answer)
        st.session_state["messages"].append({"role": "assistant", "content": answer})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                try:
                    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key, temperature=0.2)
                    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
                    
                    system_prompt = (
                        "You are a helpful customer support and AI assistant. "
                        "Use the following context to answer the user question accurately. "
                        "If you don't know the answer based on context, state politely that information is not available.\n\n"
                        "Context:\n{context}"
                    )
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", system_prompt),
                        ("human", "{input}"),
                    ])
                    
                    question_answer_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                    
                    response = rag_chain.invoke({"input": user_query})
                    answer = response.get("answer", "No response generated.")
                    
                    st.markdown(answer)
                    st.session_state["messages"].append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error: {str(e)}")
