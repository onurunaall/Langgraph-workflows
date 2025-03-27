"""
Orchestrator-Worker Workflow Example

This workflow demonstrates an orchestrator delegating subtasks to worker nodes.
The orchestrator generates a plan (sections of a report) and assigns each section to a worker.
Then, the outputs are aggregated into a final report.

Note: This example uses the Send API to simulate parallel worker delegation.
"""

from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph_workflows.llm_setup import get_llm_instance

# Define a Pydantic model for each report section
class ReportSection(BaseModel):
    section_name: str = Field(..., description="Name of the report section")
    section_description: str = Field(..., description="Description of what this section should cover")

# Define a Pydantic model for the overall plan
class ReportPlan(BaseModel):
    sections: List[ReportSection] = Field(..., description="List of planned sections for the report")

# Define the state schema for the orchestrator-worker workflow
class OrchestratorWorkflowState(TypedDict):
    report_topic: str                      # The overall topic of the report
    planned_sections: List[ReportSection]  # Sections planned by the orchestrator
    worker_outputs: List[str]              # Collected outputs from workers
    final_report: str                      # The aggregated final report

# Define the worker state schema
class WorkerState(TypedDict):
    assigned_section: ReportSection        # The section assigned to this worker
    worker_outputs: List[str]              # The worker's output list (aggregated using operator.add)

# Get the LLM instance
llm = get_llm_instance()
# Augment the LLM with structured output for planning using the ReportPlan schema
report_planner_llm = llm.with_structured_output(ReportPlan)

def orchestrator_node(state: OrchestratorWorkflowState):
    """Orchestrator node: generate a plan for the report based on the topic."""
    plan = report_planner_llm.invoke([
        SystemMessage(content="Generate a plan for a report."),
        HumanMessage(content=f"The report topic is: {state['report_topic']}")
    ])
    return {"planned_sections": plan.sections}

def worker_node(state: WorkerState):
    """Worker node: generate a report section based on the assigned section details."""
    response = llm.invoke([
        SystemMessage(content="Write a report section in markdown format without preamble."),
        HumanMessage(content=f"Section Name: {state['assigned_section'].section_name}\n"
                             f"Section Description: {state['assigned_section'].section_description}")
    ])
    # Return the worker's output wrapped in a list for aggregation
    return {"worker_outputs": [response.content]}

def synthesizer_node(state: OrchestratorWorkflowState):
    """Synthesize the final report from all worker outputs."""
    aggregated_sections = "\n\n---\n\n".join(state["worker_outputs"])
    final_report_text = f"Final Report on {state['report_topic']}:\n\n{aggregated_sections}"
    return {"final_report": final_report_text}

def assign_workers(state: OrchestratorWorkflowState):
    """Assign each planned section to a worker using the Send API."""
    # For each section, create a Send instruction for the worker node
    send_instructions = [Send("worker_node", {"assigned_section": section}) for section in state["planned_sections"]]
    return send_instructions

def run_orchestrator_worker_workflow():
    """Build, compile, and run the orchestrator-worker workflow."""
    # Create a new state graph for the orchestrator-worker workflow
    ow_graph = StateGraph(OrchestratorWorkflowState)
    
    # Add nodes: orchestrator, worker node (for each section), and synthesizer
    ow_graph.add_node("orchestrator_node", orchestrator_node)
    ow_graph.add_node("worker_node", worker_node)
    ow_graph.add_node("synthesizer_node", synthesizer_node)
    
    # Set up edges:
    # Start with the orchestrator, then conditionally assign workers based on the plan,
    # then aggregate worker outputs using the synthesizer.
    ow_graph.add_edge(START, "orchestrator_node")
    ow_graph.add_conditional_edges("orchestrator_node", assign_workers, ["worker_node"])
    ow_graph.add_edge("worker_node", "synthesizer_node")
    ow_graph.add_edge("synthesizer_node", END)
    
    # Compile the workflow
    orchestrator_worker_workflow = ow_graph.compile()
    
    # Invoke the workflow with an initial report topic
    initial_state = {"report_topic": "Scaling Laws in LLMs"}
    workflow_result = orchestrator_worker_workflow.invoke(initial_state)
    print("Final Report:")
    print(workflow_result.get("final_report", "No report generated."))

if __name__ == "__main__":
    run_orchestrator_worker_workflow()
