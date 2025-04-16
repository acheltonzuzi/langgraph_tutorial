from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing import Annotated

from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools.tavily_search import TavilySearchResults

import json

from langchain_core.messages import ToolMessage

load_dotenv()

tool = TavilySearchResults(max_results=2)
tools = [tool]

llm=ChatGroq(model="deepseek-r1-distill-llama-70b",temperature=1)
llm_with_tools = llm.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=json.dumps(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}



def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    print("DEBUG - Model response:", response)
    return {"messages": [response]}


tool_node = BasicToolNode(tools=[tool])
graph_builder = StateGraph(State)

def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END



# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_node("chatbot",chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile()

graph = graph_builder.compile()
res=graph.invoke({"messages": [{"role": "user", "content": "Quem e o actual presidente dos estados unidos ?"}]})
print(res["messages"][-1].content)

