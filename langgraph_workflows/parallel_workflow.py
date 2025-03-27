from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph_workflows.llm_setup import get_llm_instance

class ParallelWorkflowState(TypedDict):
    topic: str # The subject for which the outputs are generated
    joke: str # Output from the joke LLM call
    story: str # Output from the story LLM call
    poem: str # Output from the poem LLM call
    combined_output: str # The aggregated result

llm = get_llm_instance()

def generate_joke_output(state: ParallelWorkflowState):
    """LLM call to generate a joke about the topic."""
    response_message = llm.invoke(f"Write a joke about {state['topic']}")
    return {"joke": response_message.content}

def generate_story_output(state: ParallelWorkflowState):
    """LLM call to generate a short story about the topic."""
    response_message = llm.invoke(f"Write a story about {state['topic']}")
    return {"story": response_message.content}

def generate_poem_output(state: ParallelWorkflowState):
    """LLM call to generate a poem about the topic."""
    response_message = llm.invoke(f"Write a poem about {state['topic']}")
    return {"poem": response_message.content}

def aggregate_outputs(state: ParallelWorkflowState):
    """Combine the joke, story, and poem into a single formatted output."""
    aggregated_text = (f"Here is a combined output for the topic '{state['topic']}':\n\n"
                       f"--- STORY ---\n{state['story']}\n\n"
                       f"--- JOKE ---\n{state['joke']}\n\n"
                       f"--- POEM ---\n{state['poem']}"
                      )
  
    return {"combined_output": aggregated_text}

def run_parallel_workflow():
    """Build, compile, and run the parallelization workflow."""
    parallel_graph = StateGraph(ParallelWorkflowState)
    
    parallel_graph.add_node("generate_joke_output", generate_joke_output)
    parallel_graph.add_node("generate_story_output", generate_story_output)
    parallel_graph.add_node("generate_poem_output", generate_poem_output)
    parallel_graph.add_node("aggregate_outputs", aggregate_outputs)
    
    # Add edges: All three LLM calls start in parallel, then converge to the aggregator
    parallel_graph.add_edge(START, "generate_joke_output")
    parallel_graph.add_edge(START, "generate_story_output")
    parallel_graph.add_edge(START, "generate_poem_output")
    parallel_graph.add_edge("generate_joke_output", "aggregate_outputs")
    parallel_graph.add_edge("generate_story_output", "aggregate_outputs")
    parallel_graph.add_edge("generate_poem_output", "aggregate_outputs")
    parallel_graph.add_edge("aggregate_outputs", END)
    
    parallel_workflow = parallel_graph.compile()
    
    workflow_state = parallel_workflow.invoke({"topic": "cats"})
    print("Combined Output:")
    print(workflow_state.get("combined_output", "No output generated"))

if __name__ == "__main__":
    run_parallel_workflow()
