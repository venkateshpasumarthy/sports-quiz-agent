"""
AI-Powered Automated HR Support Chatbot (Beginner Version)
------------------------------------------------------------
Stack: Python + Streamlit + LangChain + FAISS + Gemini API

How it works (RAG = Retrieval Augmented Generation):
1. The HR policy document (data/hr_policies.txt) is split into small chunks.
2. Each chunk is converted into a vector embedding using Gemini's embedding model.
3. All embeddings are stored in a FAISS vector index for fast similarity search.
4. When an employee asks a question, we embed the question, find the most
   relevant policy chunks (the "context"), and pass them + the question to
   the Gemini chat model so it can answer using only that context.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ----------------------------------------------------------------------
# 1. Basic setup
# ----------------------------------------------------------------------
load_dotenv()  # reads GOOGLE_API_KEY from a local .env file, if present

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "hr_policies.txt")
INDEX_PATH = os.path.join(os.path.dirname(__file__), "faiss_index")

# Model names (Gemini, as of 2026). You can change these later — see README.
CHAT_MODEL = "gemini-2.5-flash"          # fast + cheap, good for Q&A
EMBEDDING_MODEL = "gemini-embedding-001"  # stable text embedding model

st.set_page_config(page_title="HR Support Chatbot", page_icon="🤖", layout="centered")

# ----------------------------------------------------------------------
# 2. Build (or load) the FAISS vector store — cached so it only runs once
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Setting up the knowledge base...")
def get_vectorstore():
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    # Reuse a previously built index if it exists, so we don't re-embed
    # the document (and burn API quota) every time the app restarts.
    if os.path.exists(INDEX_PATH):
        return FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )

    loader = TextLoader(DATA_PATH, encoding="utf-8")
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=80,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(raw_documents)

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(INDEX_PATH)
    return vectorstore


# ----------------------------------------------------------------------
# 3. Build the LLM + prompt template — also cached
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_llm_and_prompt():
    llm = ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_template(
        """You are a friendly, professional HR Support Assistant for this company.
Answer the employee's question using ONLY the HR policy context provided below.
If the context does not contain the answer, say you don't have that information
and suggest the employee contact the HR team directly. Keep answers concise and
easy to read.

HR Policy Context:
{context}

Employee Question:
{question}

Answer:"""
    )

    return llm, prompt


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer_question(question: str, vectorstore, llm, prompt):
    """Retrieve relevant policy chunks, then ask Gemini to answer using them."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(question)
    context = format_docs(retrieved_docs)

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, retrieved_docs


# ----------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------
st.title("🤖 AI-Powered HR Support Chatbot")
st.caption("Ask me about leave, payroll, benefits, work-from-home policy, and more.")

if not GOOGLE_API_KEY:
    st.error(
        "⚠️ No Gemini API key found. Create a `.env` file (see `.env.example`) "
        "with `GOOGLE_API_KEY=your_key_here`, then restart the app."
    )
    st.stop()

vectorstore = get_vectorstore()
llm, prompt = get_llm_and_prompt()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hi! I'm your HR assistant. Ask me anything about company "
            "policies — leave, payroll, benefits, work hours, and more.",
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources used for this answer"):
                for i, doc in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Snippet {i}:** {doc}")

# Chat input
user_input = st.chat_input("Type your HR question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Checking the HR handbook..."):
            try:
                answer, sources = answer_question(user_input, vectorstore, llm, prompt)
            except Exception as e:
                answer = (
                    "Sorry, something went wrong while contacting the AI model. "
                    f"Details: {e}"
                )
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("📄 Sources used for this answer"):
                for i, doc in enumerate(sources, start=1):
                    st.markdown(f"**Snippet {i}:** {doc.page_content}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [d.page_content for d in sources] if sources else [],
        }
    )

# ----------------------------------------------------------------------
# 5. Sidebar
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("About")
    st.write(
        "This chatbot answers HR questions using Retrieval Augmented Generation "
        "(RAG): it searches the company's HR policy document for relevant "
        "passages with FAISS, then asks Google's Gemini model to answer based "
        "only on those passages."
    )
    st.markdown(f"**Chat model:** `{CHAT_MODEL}`")
    st.markdown(f"**Embedding model:** `{EMBEDDING_MODEL}`")

    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(
        "Tip: edit `data/hr_policies.txt` with your own company's policies, "
        "then delete the `faiss_index` folder so it gets rebuilt."
    )
