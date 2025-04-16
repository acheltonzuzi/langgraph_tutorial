from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

def node1(state: State):
    # Return a LangChain message object, not an integer
    return {"messages": ["Bom dia"]}

def node2(state: State):
    # Add another message to the list
    return {"messages": [state["messages"][-1]]}

graph = StateGraph(State)
graph.add_node("node1", node1)
graph.add_node("node2", node2)
graph.add_edge(START, "node1")
graph.add_edge("node1", "node2")
graph.add_edge("node2", END)
graph = graph.compile()

res = graph.invoke({"messages": []})  # Initialize with empty list
print(res["messages"][-1].content)  # Should print "Response from node2"
