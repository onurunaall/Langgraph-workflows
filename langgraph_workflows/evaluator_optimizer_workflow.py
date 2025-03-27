"""
Evaluator-Optimizer Workflow Example

This workflow demonstrates iterative refinement.
One LLM call generates a joke and another evaluates it.
If the joke is not funny, the workflow loops back to generate a new joke incorporating feedback.
"""

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph_workflows.llm_setup import get_llm_instance
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

# Define the state schema for evaluator-optimizer workflow
class EvaluatorWorkflowState(TypedDict):
    report_topic: str  # For this example, using "report_topic" as the subject for a joke
    generated_joke: str
    evaluation_feedback: str
    joke_quality: str  # Expected to be "funny" or "not funny"

# Define a Pydantic model for feedback structured output
class JokeFeedback(BaseModel):
    grade: str = Field(..., description="Grade of the joke: funny or not funny")
    feedback: str = Field(..., description="Feedback on how to improve the joke")

# Get the LLM instance
llm = get_llm_instance()
# Augment the LLM with structured output for evaluation using the JokeFeedback schema
evaluator_llm = llm.with_structured_output(JokeFeedback)

def generate_joke_node(state: EvaluatorWorkflowState):
    """Generate a joke based on the topic and optional feedback."""
    if state.get("evaluation_feedback"):
        response = llm.invoke(f"Write a joke about {state['report_topic']} considering this feedback: {state['evaluation_feedback']}")
    else:
        response = llm.invoke(f"Write a joke about {state['report_topic']}")
    return {"generated_joke": response.content}

def evaluate_joke_node(state: EvaluatorWorkflowState):
    """Evaluate the generated joke and provide feedback."""
    evaluation = evaluator_llm.invoke(f"Evaluate the following joke and provide a grade (funny or not funny) and feedback: {state['generated_joke']}")
    return {"joke_quality": evaluation.grade, "evaluation_feedback": evaluation.feedback}

def route_joke_decision(state: EvaluatorWorkflowState):
    """Route based on the evaluation: if the joke is funny, end; if not, loop back."""
    if state["joke_quality"] == "funny":
        return "Accepted"
    elif state["joke_quality"] == "not funny":
        return "LoopBackForImprovement"

def run_evaluator_optimizer_workflow():
    """Build, compile, and run the evaluator-optimizer workflow."""
    evaluator_graph = StateGraph(EvaluatorWorkflowState)
    
    # Add nodes for generating and evaluating the joke
    evaluator_graph.add_node("generate_joke_node", generate_joke_node)
    evaluator_graph.add_node("evaluate_joke_node", evaluate_joke_node)
    
    # Set up edges: start with joke generation, then evaluation, then conditionally loop
    evaluator_graph.add_edge(START, "generate_joke_node")
    evaluator_graph.add_edge("generate_joke_node", "evaluate_joke_node")
    evaluator_graph.add_conditional_edges(
        "evaluate_joke_node",
        route_joke_decision,
        {"Accepted": END, "LoopBackForImprovement": "generate_joke_node"}
    )
    
    # Compile the workflow
    evaluator_optimizer_workflow = evaluator_graph.compile()
    
    # Invoke the workflow with an initial topic (using "Cats" for the joke)
    initial_state = {"report_topic": "Cats", "generated_joke": "", "evaluation_feedback": "", "joke_quality": ""}
    final_state = evaluator_optimizer_workflow.invoke(initial_state)
    print("Final Joke (accepted):")
    print(final_state.get("generated_joke", "No joke generated."))

if __name__ == "__main__":
    run_evaluator_optimizer_workflow()
