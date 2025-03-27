# Langgraph-workflows

# LangGraph Workflows & Agents

This project demonstrates common patterns for agentic systems using LangGraph. It includes examples of:

- **Prompt Chaining:** Decomposing a task into a sequence of LLM calls.
- **Parallelization:** Running multiple LLM calls simultaneously and aggregating outputs.
- **Routing:** Directing an input to specialized subtasks.
- **Orchestrator-Worker:** Dynamically breaking down tasks and delegating them to worker nodes.
- **Evaluator-Optimizer:** Iteratively refining output with evaluation and feedback.
- **Agent Loop:** A simple agent that performs arithmetic using tool calls.

Each example is implemented as a separate workflow in the `langgraph_workflows/` package. The code is written with descriptive comments and clear variable names to help junior developers learn.

- This repo contains my implementations of the core LangGraph workflow examples (like agents, prompt chaining, routing, etc.) as shown in their official tutorials which can be found at https://langchain-ai.github.io/langgraph/tutorials/workflows/#agent.
