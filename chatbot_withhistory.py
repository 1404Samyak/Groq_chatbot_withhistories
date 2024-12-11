from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import streamlit as st

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
langchain_api_key=os.getenv("LANGCHAIN_API_KEY")
groq_api_key=os.getenv("GROQ_API_KEY")


from langchain_groq import ChatGroq
model=ChatGroq(model='Llama3-8b-8192') #llm model

st.title("Groq Chatbot with chat message histories")
st.write("Kindly enter a seesion id and start interacting with the chatbot and to start a fresh talk pls change the session_id")

session_id=st.text_input("Session ID",value="default_session")

if 'store' not in st.session_state:
    st.session_state.store={}

prompt_template=ChatPromptTemplate.from_messages(
    [("system","Kindly answer to the queries of user in detail"),
     MessagesPlaceholder('chat_history'),
     ("user","{input}")]
)

parser=StrOutputParser()
chain=prompt_template|model|parser

#Adding chat history
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

def get_session_history(session:str)->BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id]=ChatMessageHistory()
        return st.session_state.store[session_id]

#Now modifying the chain to run with message history 
conversational_chain=RunnableWithMessageHistory(
    chain,get_session_history,
     input_messages_key="input",
     history_messages_key="chat_history"
)

user_input=st.text_input("You:")
if user_input:
    response=conversational_chain.invoke({"input":user_input},config={'configurable':{'session_id':session_id}})
    st.write(response)
    