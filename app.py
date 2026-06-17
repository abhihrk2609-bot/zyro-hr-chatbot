
import os
import streamlit as st

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Zyro Dynamics HR Assistant")

st.title("🤖 Zyro Dynamics HR Assistant")

# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS index
vectorstore = FAISS.load_local(
    ".",
    embeddings,
    allow_dangerous_deserialization=True
)

# Retriever
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "fetch_k": 20
    }
)

# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=st.secrets["GROQ_API_KEY"]
)

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are Zyro Dynamics HR Assistant.

Answer ONLY using the provided HR policy context.

If the answer is not found in the context, reply exactly:

I can only answer questions based on Zyro Dynamics HR policy documents.

Context:
{context}

Question:
{question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# RAG Chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)

user_question = st.chat_input("Ask an HR question")

if user_question:

    st.chat_message("user").write(user_question)

    answer = rag_chain.invoke(user_question)

    st.chat_message("assistant").write(answer)
