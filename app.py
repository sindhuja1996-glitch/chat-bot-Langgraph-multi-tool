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
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests
import random

#load env
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
api_key_google = os.getenv("GOOGLE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
API_KEY = os.getenv("API_KEY")
# llm groq
# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
#llm = ChatOpenAI()
llm = ChatGroq(api_key=api_key, model="meta-llama/llama-4-scout-17b-16e-instruct")
imagellm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-image-preview")
# state initialization
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

# tools 
search_tool = TavilySearchResults()
#search_dugduggo = DuckDuckGoSearchRun()


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

def draw_image(description: str) -> str:
    """Generate an image based on the provided description using DALL-E."""
    dalle_url = "https://api.openai.com/v1/images/generations"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key_google}"
    }
    data = {
        "model": "dall-e-3",
        "prompt": description,
        "n": 1,
        "size": "1024x1024"
    }
    
    response = requests.post(dalle_url, headers=headers, json=data)
    
    if response.status_code == 200:
        image_url = response.json()['data'][0]['url']
        print(f"Generated image URL: {image_url}")
        return image_url
    else:
        return f"Error: {response.status_code}, {response.text}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=RKMBSQTELA2SCMS3"
    r = requests.get(url)
    return r.json()

@tool
def get_weather(city: str) -> str:
    """Get current weather for a given city using OpenWeatherMap."""
    BASE_URL = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(BASE_URL)

    if response.status_code == 200:
        data = response.json()
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The weather in {city} is {description}, {temp}°C."
    else:
        return f"Could not fetch weather for {city}."
        

tools = [search_tool,get_stock_price,calculator,get_weather,draw_image]
llm_with_tools = llm.bind_tools(tools)

def Chat_node(state:ChatState):
    """As chat node receives the messages from the state and pass it to llm with respective tools"""
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


