"""
Agent Workflow Example

This workflow implements a simple agent that performs arithmetic using tool calls.
The agent decides whether to call a tool (for arithmetic) based on the input message.
"""

from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph_workflows.llm_setup import get_llm_instance

# Define the state schema for the agent workflow using messages
class AgentWorkflowState(TypedDict):
    messages: list  # A list of messages exchanged in the agent workflow

# Define simple arithmetic tools
@tool
def multiply_numbers(first_number: int, second_number: int) -> int:
    """Multiply two integers and return the result."""
    return first_number * second_number

@tool
def add_numbers(first_number: int, second_number: int) -> int:
    """Add two integers and return the sum."""
    return first_number + second_number

@tool
def divide_numbers(first_number: int, second_number: int) -> float:
    """Divide the first integer by the second and return the quotient."""
    return first_number / second_number

# Create a list of tools and a dictionary for lookup by name
arithmetic_tools = [add_numbers, multiply_numbers, divide_numbers]
tools_lookup = {tool_instance.name: tool_instance for tool_instance in arithmetic_tools}

# Get the LLM instance and bind it with arithmetic tools
llm = get_llm_instance()
llm_with_tools = llm.bind_tools(arithmetic_tools)

def agent_llm_call(state: AgentWorkflowState):
    """Agent node: the LLM processes messages and may decide to call a tool."""
    # The agent sends a system message along with the conversation history
    combined_messages = [SystemMessage(content="You are a helpful arithmetic assistant.")] + state["messages"]
    response_message = llm_with_tools.invoke(combined_messages)
    return {"messages": [response_message]}

def agent_tool_execution(state: AgentWorkflowState):
    """Tool node: execute the arithmetic tool call requested by the agent."""
    tool_messages = []
    last_agent_message = state["messages"][-1]
    # Process each tool call in the last agent message
    for tool_call in getattr(last_agent_message, "tool_calls", []):
        tool_function = tools_lookup.get(tool_call["name"])
        if tool_function:
            result = tool_function.invoke(tool_call["args"])
            # Wrap the tool result in a ToolMessage for display
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

def decide_agent_route(state: AgentWorkflowState) -> Literal["execute_tool", END]:
    """Decide whether to call the tool node or to end the loop based on the agent's output."""
    last_message = state["messages"][-1]
    # If there are tool calls in the last message, route to tool execution
    if getattr(last_message, "tool_calls", []):
        return "execute_tool"
    return END

def run_agent_workflow():
    """Build, compile, and run the agent workflow."""
    agent_graph = StateGraph(AgentWorkflowState)
    
    # Add nodes for the LLM call and tool execution
    agent_graph.add_node("agent_llm_call", agent_llm_call)
    agent_graph.add_node("agent_tool_execution", agent_tool_execution)
    
    # Set up edges: start with agent LLM call, then conditionally route based on tool calls
    agent_graph.add_edge(START, "agent_llm_call")
    agent_graph.add_conditional_edges(
        "agent_llm_call",
        decide_agent_route,
        {"execute_tool": "agent_tool_execution", END: END}
    )
    agent_graph.add_edge("agent_tool_execution", "agent_llm_call")
    
    # Compile the agent workflow
    agent_workflow = agent_graph.compile()
    
    # Prepare an initial message for an arithmetic query
    from langchain_core.messages import HumanMessage
    initial_messages = [HumanMessage(content="Add 3 and 4.")]
    initial_state = {"messages": initial_messages}
    
    # Invoke the workflow and print the resulting messages
    final_state = agent_workflow.invoke(initial_state)
    for message in final_state["messages"]:
        message.pretty_print()

if __name__ == "__main__":
    run_agent_workflow()
