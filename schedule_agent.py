from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from typing_extensions import TypedDict
from typing import Annotated, Dict, Literal
import datetime
load_dotenv()

llm = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=1)

tasks = []

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def addTask(task: str) -> str:
    """Add a task to the list."""
    if task in tasks:
        return f"Task '{task}' already exists."
    else:
        tasks.append(task)
    return f"Task '{task}' added."

@tool
def getCurrentDate() -> str:
    """Get the current date and time."""
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")
tools = [addTask,getCurrentDate]
llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

# Configurar o grafo
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)

# Correto: passar a lista de ferramentas ao ToolNode
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

# Corrigir o tools_condition para roteamento adequado
def router(state: State) -> Literal["tools", END]:
    messages = state["messages"]
    last_message = messages[-1]
    
    # Verificar se há chamadas de ferramentas na resposta do modelo
    if "tool_calls" in last_message.additional_kwargs:
        return "tools"
    return END

# Adicionar as bordas condicionais corretamente
graph_builder.add_conditional_edges("chatbot", router)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")

graph = graph_builder.compile()

while True:
    input_message = input("User: ")
    res = graph.invoke({"messages": [{"role": "user", "content": input_message}]})
    print(res["messages"][-1].content)
    print("=======================================")
    print(tasks)