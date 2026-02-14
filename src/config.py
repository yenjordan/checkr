from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA

load_dotenv(dotenv_path=".env")

class Settings(BaseSettings):
    #TODO: update kb name
    kb_id: str = ""
    llm_model: str = "nvidia/llama-3.3-nemotron-super-49b-v1"
    llm_reasoning_model: str = "nvidia/llama-3.1-nemotron-ultra-253b-v1"

settings = Settings()

# Super 49B — fast, used for planner, extractors, coding agent
llm = ChatNVIDIA(model=settings.llm_model)

# Ultra 253B — reasoning-heavy, used for math agent & replanner
llm_reasoning = ChatNVIDIA(model=settings.llm_reasoning_model)

class PaperCheckerConfig(BaseModel):
    """base config for paper checker agent lolz we can change name later"""
    agent_name: str = "Paper Checker"
    default_prompt: str = "No prompt found in input, please guide user to provide a research publication to check."

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "PaperCheckerConfig":
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        return cls()