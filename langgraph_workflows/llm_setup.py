import os
import getpass
from langchain_anthropic import ChatAnthropic

def prompt_for_env_variable(variable_name: str) -> None:
    if not os.environ.get(variable_name):
        os.environ[variable_name] = getpass.getpass(f"Please enter your {variable_name}: ")

prompt_for_env_variable("ANTHROPIC_API_KEY")

def get_llm_instance():
    return ChatAnthropic(model="claude-3-5-sonnet-latest")
