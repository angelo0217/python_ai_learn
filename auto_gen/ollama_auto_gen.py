import asyncio
import nest_asyncio
from typing import Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_ext.models.openai import OpenAIChatCompletionClient
from rich.console import Console
from rich.panel import Panel

from core.config import settings
from core.logger import logger
from core.exceptions import APIError

nest_asyncio.apply()
console = Console()

class OllamaAgentManager:
    \"\"\"
    Manages the creation and execution of agents using Ollama.
    \"\"\"
    
    def __init__(self, model_name: str = "llama3.1:latest", base_url: str = "http://localhost:11434/v1"):
        self.model_name = model_name
        self.base_url = base_url

    def get_model_client(self) -> OpenAIChatCompletionClient:
        \"\"\"
        Creates and returns an OpenAIChatCompletionClient configured for Ollama.
        \"\"\"
        try:
            logger.info(f"Initializing Ollama client with model: {self.model_name}")
            return OpenAIChatCompletionClient(
                model=self.model_name,
                api_key="ollama",
                base_url=self.base_url,
                model_capabilities={
                    "json_output": False,
                    "vision": False,
                    "function_calling": True,
                },
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {str(e)}")
            raise APIError(f"Ollama client initialization failed: {str(e)}")

    async def run_simple_chat(self, task: str, agent_names: list[str], max_messages: int = 11) -> None:
        \"\"\"
        Runs a round-robin chat between a set of agents.
        \"\"\"
        try:
            model_client = self.get_model_client()
            agents = [AssistantAgent(name=name, model_client=model_client) for name in agent_names]
            
            termination = MaxMessageTermination(max_messages)
            team = RoundRobinGroupChat([*agents], termination_condition=termination)

            logger.info(f"Starting Ollama group chat for task: {task}")
            stream = team.run_stream(task=task)

            async for message in stream:
                if hasattr(message, "content"):
                    source = getattr(message, "source", "Unknown")
                    console.print(
                        Panel(
                            message.content,
                            title=f"[bold blue]{source}[/bold blue]",
                            expand=False
                        )
                    )
        except Exception as e:
            logger.exception(f"Error during Ollama chat execution: {str(e)}")
            raise APIError(f"Ollama chat failed: {str(e)}")

async def main() -> None:
    manager = OllamaAgentManager()
    await manager.run_simple_chat(
        task="Count from 1 to 10, respond one at a time.",
        agent_names=["Assistant1", "Assistant2"]
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("User interrupted the process.")
    except Exception as e:
        logger.error(f"Unexpected error in main: {str(e)}")
