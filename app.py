import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Book Assistant", page_icon="📚", layout="wide")

st.title("📚 RAG Book Assistant")

# 1. Session State Initialization
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# 2. Sidebar Controls & File Upload
with st.sidebar:
    st.header("⚙️ Document Setup")
    uploaded_file = st.file_uploader("Upload a PDF book", type="pdf")

    chunk_size = st.number_input("Chunk Size", min_value=100, max_value=2000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk Overlap", min_value=0, max_value=500, value=200, step=50)

    process_btn = st.button("Index Document", type="primary", disabled=not uploaded_file)

# 3. Document Processing Pipeline
if process_btn and uploaded_file:
    with st.spinner("Processing document and generating vector embeddings..."):
        file_path = None
        try:
            # Safely create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                file_path = tmp_file.name

            # Load and split
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            chunks = splitter.split_documents(docs)

            embeddings = OpenAIEmbeddings()

            # Store in session state (in-memory Chroma instance)
            st.session_state.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            st.sidebar.success(f"Indexed {len(docs)} pages into {len(chunks)} chunks!")
            st.session_state.messages = []  # Reset chat on new document load
        except Exception as e:
            st.sidebar.error(f"Failed to process document: {e}")
        finally:
            # Cleanup temporary file from disk
            if file_path and os.path.exists(file_path):
                os.unlink(file_path)

# 4. Display Existing Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("🔍 View Retrieved Sources"):
                for idx, src in enumerate(message["sources"], 1):
                    page_num = src.metadata.get("page", 0) + 1
                    st.markdown(f"**Source {idx} (Page {page_num}):**")
                    st.caption(src.page_content)

# 5. Interactive Chat Input & Retrieval Pipeline
if query := st.chat_input("Ask a question about the book..."):
    if not st.session_state.vectorstore:
        st.warning("Please upload and index a PDF document in the sidebar first.")
    else:
        # Display user query
        st.chat_message("user").markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})

        # Retrieve relevant chunks
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5}
        )
        docs = retriever.invoke(query)

        # Format context with page metadata
        formatted_context = "\n\n".join([
            f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}"
            for doc in docs
        ])

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful AI assistant. Use ONLY the provided context to answer the question.\n"
                "If the answer is not present in the context, say: 'I could not find the answer in the document.'"
            ),
            (
                "human",
                "Context:\n{context}\n\nQuestion:\n{question}"
            )
        ])

        llm = ChatMistralAI(model="mistral-small-2506")
        chain = prompt | llm

        # Stream assistant response
        with st.chat_message("assistant"):
            response_stream = chain.stream({"context": formatted_context, "question": query})
            full_response = st.write_stream(response_stream)

            # Show expandable source attribution
            with st.expander("🔍 View Retrieved Sources"):
                for idx, doc in enumerate(docs, 1):
                    page_num = doc.metadata.get("page", 0) + 1
                    st.markdown(f"**Source {idx} (Page {page_num}):**")
                    st.caption(doc.page_content)

        # Store assistant response and sources in history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response,
            "sources": docs
        })
