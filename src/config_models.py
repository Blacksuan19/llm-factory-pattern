from enum import Enum
from token import OP
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Provider(str, Enum):
    """Enumeration for supported LLM providers."""

    BEDROCK = "bedrock"
    OPENAI = "openai"


class ModelConfig(BaseModel):
    """Pydantic model for a single LLM configuration."""

    name: str
    provider: Union[Provider, str]  # Can be either enum or string for custom providers
    model_id: str
    region_name: Optional[str] = None
    api_key_secret_name: Optional[str] = None
    api_key_env_var: Optional[str] = "OPENAI_API_KEY"
    input_token_cost: float = Field(
        0.0, ge=0.0, alias="input_token_cost_usd_per_million"
    )
    output_token_cost: float = Field(
        0.0, ge=0.0, alias="output_token_cost_usd_per_million"
    )
    max_tokens: int = Field(1024, ge=1)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    description: Optional[str] = None


class AllModelsConfig(BaseModel):
    """Pydantic model for the entire collection of LLM configurations."""

    models: Dict[str, ModelConfig] = Field(default_factory=dict)


class EnvSettings(BaseSettings):
    SSM_PROVIDER_PATH_PARAMETER: Optional[str] = "/LLM_CONFIG/PROVIDER_MODULES_S3_PATH"
    SSM_MODELS_PATH_PARAMETER: Optional[str] = "/LLM_CONFIG/MODELS_CONFIG_S3_PATH"
    OPENAI_API_KEY: Optional[str] = None


env = EnvSettings()
