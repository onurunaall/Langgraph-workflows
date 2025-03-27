"""
Basic tests for the LangGraph workflows.
These tests ensure that each workflow can be invoked without errors.
"""

import pytest
from langgraph_workflows.prompt_chain_workflow import run_prompt_chain_workflow
from langgraph_workflows.parallel_workflow import run_parallel_workflow
from langgraph_workflows.routing_workflow import run_routing_workflow
from langgraph_workflows.orchestrator_worker_workflow import run_orchestrator_worker_workflow
from langgraph_workflows.evaluator_optimizer_workflow import run_evaluator_optimizer_workflow
from langgraph_workflows.agent_workflow import run_agent_workflow

def test_prompt_chain_workflow(capsys):
    # Run the prompt chaining workflow and capture output
    run_prompt_chain_workflow()
    captured = capsys.readouterr().out
    assert "joke" in captured.lower()

def test_parallel_workflow(capsys):
    run_parallel_workflow()
    captured = capsys.readouterr().out
    assert "combined output" in captured.lower()

def test_routing_workflow(capsys):
    run_routing_workflow()
    captured = capsys.readouterr().out
    assert "final output" in captured.lower()

def test_orchestrator_worker_workflow(capsys):
    run_orchestrator_worker_workflow()
    captured = capsys.readouterr().out
    assert "final report" in captured.lower()

def test_evaluator_optimizer_workflow(capsys):
    run_evaluator_optimizer_workflow()
    captured = capsys.readouterr().out
    assert "final joke" in captured.lower()

def test_agent_workflow(capsys):
    run_agent_workflow()
    captured = capsys.readouterr().out
    assert "add" in captured.lower()
