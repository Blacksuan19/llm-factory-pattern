import os

import boto3
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from config_models import ModelConfig

from .base_model import BaseLlmModel


class LangChainOpenAIModel(BaseLlmModel):
    """Concrete implementation for OpenAI chat models using LangChain's ChatOpenAI."""

    def __init__(self, name: str, config: ModelConfig):
        """Initializes a LangChainOpenAIModel instance."""
        super().__init__(name, config)

    def _initialize_llm(self) -> BaseChatModel:
        """Initializes the underlying LangChain ChatOpenAI LLM."""
        api_key = None
        if self.config.api_key_secret_name:
            try:
                secrets_client = boto3.client("secretsmanager")
                response = secrets_client.get_secret_value(
                    SecretId=self.config.api_key_secret_name
                )
                api_key = response["SecretString"]
                print(f"Fetched API key for {self.name} from AWS Secrets Manager.")
            except Exception as e:
                print(
                    f"Warning: Could not fetch API key from Secrets Manager: {e}. Falling back to env var."
                )

        if not api_key:
            api_key = os.environ.get(self.config.api_key_env_var)
            if not api_key:
                raise ValueError(
                    f"API key for {self.name} not found. Set '{self.config.api_key_env_var}' or configure 'api_key_secret_name'."
                )

        return ChatOpenAI(
            model=self.config.model_id,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            api_key=api_key,
        )
