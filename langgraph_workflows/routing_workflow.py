from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_workflows.llm_setup import get_llm_instance

class RouteOutput(BaseModel):
    step: Literal["poem", "story", "joke"] = Field(None,
                                                   description="The next step in the routing process"
                                                  )

class RoutingWorkflowState(TypedDict):
    user_input: str # The original input from the user
    routing_decision: str # The decision output from the routing LLM
    final_output: str # The final generated output

# Augment the LLM with structured output using the Pydantic model
llm = get_llm_instance()
routing_llm = llm.with_structured_output(RouteOutput)

def route_input(state: RoutingWorkflowState):
    """Route the input using structured output from the augmented LLM."""
    decision = routing_llm.invoke([
        SystemMessage(content="Based on the user's request, decide whether to generate a story, joke, or poem."),
        HumanMessage(content=state["user_input"])
    ])
    return {"routing_decision": decision.step}

def generate_story_output(state: RoutingWorkflowState):
    """LLM call to generate a story based on the user input."""
    response = llm.invoke(f"Write a story based on: {state['user_input']}")
    return {"final_output": response.content}

def generate_joke_output(state: RoutingWorkflowState):
    """LLM call to generate a joke based on the user input."""
    response = llm.invoke(f"Write a joke based on: {state['user_input']}")
    return {"final_output": response.content}

def generate_poem_output(state: RoutingWorkflowState):
    """LLM call to generate a poem based on the user input."""
    response = llm.invoke(f"Write a poem based on: {state['user_input']}")
    return {"final_output": response.content}

def decide_next_node(state: RoutingWorkflowState):
    """Determine which node to route to based on the routing decision."""
    decision = state["routing_decision"]
    if decision == "story":
        return "generate_story_output"
    elif decision == "joke":
        return "generate_joke_output"
    elif decision == "poem":
        return "generate_poem_output"

def run_routing_workflow():
    """Build, compile, and run the routing workflow."""
    routing_graph = StateGraph(RoutingWorkflowState)
    
    # Add nodes for routing and output generation
    routing_graph.add_node("route_input", route_input)
    routing_graph.add_node("generate_story_output", generate_story_output)
    routing_graph.add_node("generate_joke_output", generate_joke_output)
    routing_graph.add_node("generate_poem_output", generate_poem_output)
    
    # Set up edges:
    routing_graph.add_edge(START, "route_input")
    routing_graph.add_conditional_edges("route_input", decide_next_node,
                                        {
                                            "generate_story_output": "generate_story_output",
                                            "generate_joke_output": "generate_joke_output",
                                            "generate_poem_output": "generate_poem_output",
                                        })
    routing_graph.add_edge("generate_story_output", END)
    routing_graph.add_edge("generate_joke_output", END)
    routing_graph.add_edge("generate_poem_output", END)
    
    # Compile the workflow
    routing_workflow = routing_graph.compile()
    
    # Invoke the workflow with a sample input
    workflow_state = routing_workflow.invoke({"user_input": "Write me a joke about cats"})
    print("Final Output:")
    print(workflow_state.get("final_output", "No output generated"))

if __name__ == "__main__":
    run_routing_workflow()
