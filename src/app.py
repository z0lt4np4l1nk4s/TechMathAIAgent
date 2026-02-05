import streamlit as st
from core.engine import Engine

# --- PAGE SETUP ---
# Configure the browser tab title and set the layout to wide mode for better visibility
st.set_page_config(page_title="Dev & Math AI Agent", layout="wide")

# --- ENGINE PERSISTENCE ---
# Use Streamlit's session_state to ensure the Engine and models are loaded only once.
# This prevents the application from re-initializing heavy AI models on every user interaction.
if "engine" not in st.session_state:
    # Initialize the core RAG engine; clear_on_start ensures a fresh index if needed
    st.session_state.engine = Engine(clear_on_start=True)
    # Store chat history to maintain conversation context across reruns
    st.session_state.messages = []

# Create a local reference to the persistent engine object
engine = st.session_state.engine

# Track whether the model is currently generating a response to manage UI locking
if "generating" not in st.session_state:
    st.session_state.generating = False

# --- INITIALIZATION ---
# Handle the initial booting process of the AI models and system components
if "booted" not in st.session_state:
    with st.status("Pokretanje sustava...", expanded=False) as status:
        # Load LLM, embedding models, and vector stores into memory
        engine.boot()
        st.session_state.booted = True
        status.update(label="Sustav je spreman!", state="complete")

# --- SIDEBAR: Knowledge Base ---
with st.sidebar:
    st.header("📁 Baza znanja")
    # File uploader allows users to ingest PDFs and images for RAG context
    uploaded_files = st.file_uploader(
        "Prenesi dokumente (PDF/Slike)", 
        accept_multiple_files=True
    )

    # Trigger indexing process if new files are uploaded and system is idle
    if uploaded_files and not engine.is_indexing:
        if engine.process_uploads(uploaded_files):
            st.toast("Baza znanja je ažurirana!")
            st.rerun()  # Refresh the UI to reflect changes in the knowledge base
            
    st.divider()
    # Button to reset the conversation and clear internal engine buffers
    if st.button("🗑️ Očisti razgovor"):
        engine.clear_chat()
        st.session_state.messages = []
        st.rerun()

# --- MAIN UI ---
st.title("🚀 Tech-Math AI Agent")
st.caption("Multimodalni RAG sustav za analizu dokumentacije i koda")

# Render the chat history from the session state
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # If the message is from the assistant and used RAG, display source citations
        if msg["role"] == "assistant" and msg.get("source_used") and msg.get("sources"):
            st.markdown("---")
            st.markdown("##### 📚 Izvor:")
            # Extract unique source names from the metadata retrieved by FAISS
            source_names = {res.get('source', 'Nepoznato') for res in msg["sources"]}
            cols = st.columns(max(len(source_names), 1))
            for i, name in enumerate(source_names):
                with cols[i]:
                    st.caption(f"📄 {name}")

# --- CHAT LOGIC ---
# Determine if the input should be disabled based on system status (indexing or generating)
is_locked = engine.is_indexing or st.session_state.generating
chat_placeholder = "Postavite pitanje..." if not is_locked else "Molimo pričekajte..."

# Handle new user input via the chat bar
if prompt := st.chat_input(chat_placeholder, disabled=is_locked):
    st.session_state.generating = True
    
    # Append and display the user's message immediately
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display the assistant's response
    with st.chat_message("assistant"):
        with st.spinner("Analiziram..."):
            # The engine handles Intent Detection -> Retrieval -> Generation
            answer, sources, source_used = engine.generate_response(prompt)
            
            st.markdown(answer)

            # If retrieval was successful and information was found, show attribution
            if sources and source_used:
                st.markdown("---")
                st.markdown("##### 📚 Izvor:")
                source_names = {res.get('source', 'Nepoznato') for res in sources}
                cols = st.columns(max(len(source_names), 1))
                for i, name in enumerate(source_names):
                    with cols[i]:
                        st.caption(f"📄 {name}")
        
            # Store the final assistant response along with metadata in the history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": answer, 
                "sources": sources, 
                "source_used": source_used
            })
            st.session_state.generating = False