from typing import Optional
import os
import json

from pydantic_settings import BaseSettings
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_core.runnables import RunnableConfig
from langchain_google_vertexai import ChatVertexAI
from google.oauth2 import service_account

load_dotenv(dotenv_path=".env")


def _get_vertex_credentials():
    """Build Vertex AI credentials from the service account JSON env var."""
    sa_json = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "")
    if sa_json:
        return service_account.Credentials.from_service_account_info(
            json.loads(sa_json)
        )
    return None


_vertex_creds = _get_vertex_credentials()
_vertex_project = os.environ.get("GOOGLE_CLOUD_PROJECT_ID", "")
_vertex_location = os.environ.get("VERTEX_AI_LOCATION", "us-central1")


class Settings(BaseSettings):
    kb_id: str = ""
    llm_model: str = "gemini-2.0-flash"
    llm_reasoning_model: str = "gemini-2.5-pro"

settings = Settings()

# Gemini 2.0 Flash — fast, used for planner, extractors, coding agent
llm = ChatVertexAI(
    model=settings.llm_model,
    project=_vertex_project,
    location=_vertex_location,
    credentials=_vertex_creds,
    max_output_tokens=4096,
)

# Gemini 2.5 Pro — reasoning-heavy, used for math agent & replanner
llm_reasoning = ChatVertexAI(
    model=settings.llm_reasoning_model,
    project=_vertex_project,
    location=_vertex_location,
    credentials=_vertex_creds,
    max_output_tokens=8192,
)

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