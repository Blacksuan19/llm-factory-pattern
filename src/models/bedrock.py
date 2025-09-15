from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseChatModel

from config_models import ModelConfig
from models.base_model import BaseLlmModel


class LangChainBedrockModel(BaseLlmModel):
    """Concrete implementation for AWS Bedrock models using LangChain's ChatBedrock."""

    def __init__(self, name: str, config: ModelConfig):
        """Initializes a LangChainBedrockModel instance."""
        super().__init__(name, config)

    def _initialize_llm(self) -> BaseChatModel:
        """Initializes the underlying LangChain ChatBedrock LLM."""
        return ChatBedrock(
            model_id=self.config.model_id,
            region_name=self.config.region_name,
            temperature=self.config.temperature,
            model_kwargs={"max_tokens": self.config.max_tokens},
        )
