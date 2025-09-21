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
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode,tools_condition
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
import requests
import random

#load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
api_key_google = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
# llm groq
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
#llm = ChatGroq(api_key=api_key, model="llama-3.1-8b-instant")
# sate intiallization
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# tools 
search_tool = TavilySearchResults(api_key=tavily_api_key)
search_dugduggo = DuckDuckGoSearchRun()


@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}
        
        return {"first_num": first_num, "second_num": second_num, "operation": operation, "result": result}
    except Exception as e:
        return {"error": str(e)}




@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=RKMBSQTELA2SCMS3"
    r = requests.get(url)
    return r.json()        

tools = [get_stock_price,calculator,search_dugduggo,search_tool]
llm_with_tools = llm.bind_tools(tools)

def Chat_node(state:ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)
 

cons = sqlite3.connect(database='chatbot.db',check_same_thread=False)
checkpointer = SqliteSaver(conn=cons)
# build grpah
graph = StateGraph(ChatState)
graph.add_node("chat_node", Chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools', 'chat_node')

chatbot = graph.compile(checkpointer=checkpointer)

def retrive_all_threds():
    all_thredas = set()
    for checkpoint in checkpointer.list(None):
        all_thredas.add(checkpoint.config['configurable']['thread_id'])

    return list(all_thredas)


