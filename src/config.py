from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic

load_dotenv()

class Settings(BaseSettings):
    #TODO: update kb name
    kb_id: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

settings = Settings()

llm = ChatAnthropic(model=settings.llm_model)

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