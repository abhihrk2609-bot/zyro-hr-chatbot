
import streamlit as st

st.set_page_config(page_title="Zyro Dynamics HR Assistant")

st.title("🤖 Zyro Dynamics HR Assistant")

user_question = st.chat_input("Ask an HR question")

if user_question:

    st.chat_message("user").write(user_question)

    answer = rag_chain.invoke(user_question)

    st.chat_message("assistant").write(answer)
