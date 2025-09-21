from typing import Annotated,TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import BaseMessage
from langgraph.graph import START,StateGraph,END
from langgraph.graph.message import add_messages
import os
from groq import Groq
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

#load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
# llm groq
llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
# sate intiallization
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]



def Chat_node(state:ChatState):

    message = state['messages']
    response = llm.invoke(message) 
    return {'messages':response}   

cons = sqlite3.connect(database='chatbot.db',check_same_thread=False)
# build grpah
graph = StateGraph(ChatState)

graph.add_node('Chat_node',Chat_node)


graph.add_edge(START,'Chat_node')

graph.add_edge('Chat_node',END)

checkpointer = SqliteSaver(conn=cons)
chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threds():
    all_thredas = set()
    for checkpoint in checkpointer.list(None):
        all_thredas.add(checkpoint.config['configurable']['thread_id'])

    return list(all_thredas)


