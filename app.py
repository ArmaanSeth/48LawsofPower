# importing dependencies
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Cassandra
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationKGMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from htmlTemplates import css, bot_template, user_template
import cassio

# creating custom template to guide llm model
prompt_template1="""
Answer the question based only on given context:
Context: \n{context}\n
Question: \n{question}\n
Elaborate the answer giving all the information possible related to the question from the context in english
Dont state about the context in the answer.
Answer:
"""
prompt1=PromptTemplate(template=prompt_template1,input_variables=["context","question"])

def get_vectorstore():
    cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"), database_id=os.getenv("ASTRA_DB_ID"))
    astra_vector_store=Cassandra(
        embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device':'cpu'}),
        table_name="laws_of_power",
        session=None,
        keyspace=None
    )
    return astra_vector_store

# generating conversation chain  
def get_conversationchain():
    vectorstore=get_vectorstore()
    llm=ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True)
    memory = ConversationKGMemory(llm=llm,memory_key='chat_history', return_messages=True) # using conversation buffer memory to hold past information
    chain = ConversationalRetrievalChain.from_llm(llm=llm, memory=memory, verbose=True, combine_docs_chain_kwargs={"prompt":prompt1},retriever=vectorstore.as_retriever())
    return chain

# generating response from user queries and displaying them accordingly
def handle_question(question):
    try:
        res=st.session_state.conversation.invoke(question)
    except Exception:
        try:    
            res=st.session_state.conversation.invoke(question)
        except Exception as e:
            st.write("Please resubmit the prompt, an error has occured")
            return
    # res=st.session_state.conversation.invoke({"input":question})
    st.session_state.chat_history.append([question,res["answer"]])
    for conversation in st.session_state.chat_history:
        st.write(user_template.replace("{{MSG}}",conversation[0]),unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}",conversation[1]),unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="48 Laws of Power",page_icon=":closed_book:")
    st.write(css,unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation=get_conversationchain()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    
    st.header(":closed_book: 48 Laws of Power ")
    question=st.text_input("Ask question from the book:")
    if question:
        handle_question(question)


if __name__ == '__main__':
    main()