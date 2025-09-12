from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from config_models import ModelConfig


class BaseLlmModel(ABC):
    """Abstract Base Class for all language models, wrapping LangChain's BaseChatModel.

    This class provides a common interface for interacting with LLMs, including application-specific
    functionality like cost tracking, while also allowing direct access to the underlying
    LangChain `BaseChatModel` for advanced usage.

    Attributes:
        name (str): The human-readable name of the model.
        config (ModelConfig): The Pydantic configuration object for this model.
    """

    def __init__(self, name: str, config: ModelConfig):
        """Initializes the BaseModel with a name and its configuration.

        Args:
            name (str): The human-readable name of the model.
            config (ModelConfig): The Pydantic configuration object for this model.
        """
        self.name = name
        self.config = config
        self._llm: BaseChatModel = self._initialize_llm()

    @abstractmethod
    def _initialize_llm(self) -> BaseChatModel:
        """Initializes the underlying LangChain LLM based on the configuration.

        This method must be implemented by concrete model classes.

        Returns:
            BaseChatModel: An initialized LangChain chat model instance.
        """
        pass

    @property
    def llm(self) -> BaseChatModel:
        """Provides direct access to the underlying LangChain BaseChatModel instance.

        This allows users to leverage full LangChain capabilities directly.

        Returns:
            BaseChatModel: The wrapped LangChain BaseChatModel instance.
        """
        return self._llm

    def invoke(self, prompt: str, **kwargs) -> str:
        """Generates a response for a given prompt using the wrapped LangChain LLM.

        Args:
            prompt (str): The input prompt for the LLM.
            **kwargs: Additional keyword arguments to pass to the LLM's invoke method.

        Returns:
            str: The generated response from the LLM.
        """
        print(f"[{self.name}] Invoking LangChain LLM...")
        try:
            # LangChain's invoke method takes a string or a list of messages
            # For simplicity, we'll use string input here.
            return self._llm.invoke(prompt, **kwargs)
        except Exception as e:
            return f"Error calling {self.name} via LangChain: {e}"

    def calculate_cost(self, token_count: int, is_input: bool) -> float:
        """Calculates the cost for a given token count and direction (input/output).

        Args:
            token_count (int): The number of tokens to calculate the cost for.
            is_input (bool): True to get input token cost, False for output token cost.

        Returns:
            float: The calculated cost in USD.
        """
        token_price = (
            self.config.input_token_cost if is_input else self.config.output_token_cost
        )
        token_cost = (token_count / 1_000_000) * token_price
        return token_cost

    def __repr__(self):
        """Returns a string representation of the model."""
        return f"{self.__class__.__name__}(name='{self.name}', llm_type='{self._llm.__class__.__name__}')"
