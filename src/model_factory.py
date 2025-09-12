import importlib.util
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Dict, Type

import boto3
from omegaconf import DictConfig, OmegaConf
from pydantic import ValidationError
from s3path import S3Path

from config_models import AllModelsConfig, ModelConfig, Provider, env
from exceptions import ModelConfigurationError, ModelNotFoundError
from models.base_model import BaseLlmModel
from models.bedrock import LangChainBedrockModel
from models.openai import LangChainOpenAIModel


class ModelFactory:
    """A singleton factory class for dynamically creating and managing language model instances."""

    _instance = None
    # Provider type (str) to model class mapping
    _model_type_registry: Dict[str, Type[BaseLlmModel]] = {}
    # Custom provider modules loaded from S3
    _custom_provider_modules = {}

    def __new__(cls, local_path: str):
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
            # Register built-in providers
            if not cls._model_type_registry:
                cls._model_type_registry = {
                    Provider.BEDROCK.value: LangChainBedrockModel,
                    Provider.OPENAI.value: LangChainOpenAIModel,
                }
            cls._instance._all_models_config = None
            # Load custom provider modules from S3
            cls._instance._load_custom_providers()
            cls._instance.load_configurations(local_path)

        return cls._instance

    def _fetch_ssm_parameter(self, parameter_name: str, required: bool = False) -> str:
        """Fetch a parameter value from SSM Parameter Store

        Args:
            parameter_name: The name of the SSM parameter
            required: If True, raises an exception when parameter cannot be retrieved

        Returns:
            The parameter value or empty string if not found and not required
        """
        try:
            ssm_client = boto3.client("ssm")
            response = ssm_client.get_parameter(
                Name=parameter_name,
                WithDecryption=True,
            )
            return response.get("Parameter", {}).get("Value", "")
        except Exception as e:
            error_msg = f"Failed to load parameter '{parameter_name}' from SSM: {e}"
            if required:
                raise ModelConfigurationError(error_msg)
            print(f"Warning: {error_msg}")
            return ""

    def _load_custom_providers(self):
        """Load custom provider modules from S3"""
        # Get SSM parameter name from environment variable
        param_name = env.SSM_PROVIDER_PATH_PARAMETER
        s3_provider_path = self._fetch_ssm_parameter(param_name)
        if not s3_provider_path:
            print("No S3 provider path configured, skipping custom provider loading")
            return

        # Create a temporary directory to store downloaded provider modules
        with tempfile.TemporaryDirectory() as temp_dir:
            s3_dir = S3Path(s3_provider_path)
            if not s3_dir.is_dir():
                print(f"S3 provider path '{s3_provider_path}' is not a directory.")
                return

            # Download all Python files from S3
            for py_file in s3_dir.glob("*.py"):
                provider_name = py_file.stem
                local_file_path = Path(temp_dir) / f"{provider_name}.py"
                local_file_path.write_text(py_file.read_text())
                self._load_provider_module(provider_name, local_file_path)

    def _load_provider_module(self, provider_name: str, file_path: str):
        """Dynamically load a provider module from a file path"""
        # Create a module spec and load the module
        spec = importlib.util.spec_from_file_location(
            f"custom_provider_{provider_name}", file_path
        )
        if not spec or not spec.loader:
            print(f"Failed to create module spec for {provider_name}")
            return

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the provider class in the module (should implement BaseLlmModel)
        for attr_name in dir(module):
            candidate_class = getattr(module, attr_name)
            is_valid_provider_class = (
                isinstance(candidate_class, type)
                and candidate_class is not BaseLlmModel
                and issubclass(candidate_class, BaseLlmModel)
            )
            if is_valid_provider_class:
                provider_key = provider_name.lower()
                self._model_type_registry[provider_key] = candidate_class
                self._custom_provider_modules[provider_key] = module
                print(f"Successfully registered custom provider: {provider_key}")
                break
        else:
            print(f"No valid provider class found in module {provider_name}")

    def _load_and_validate_configs(self, raw_configs: DictConfig) -> AllModelsConfig:
        """Loads raw OmegaConf DictConfig and validates it with Pydantic."""
        try:
            dict_configs = OmegaConf.to_container(raw_configs, resolve=True)
            return AllModelsConfig.model_validate(dict_configs)
        except ValidationError as e:
            raise ModelConfigurationError(f"Configuration validation failed: {e}")
        except Exception as e:
            raise ModelConfigurationError(f"Error processing configurations: {e}")

    def load_configurations(self, source_path: str):
        """Loads model configurations from various sources and validates them."""
        print(f"Loading configurations from: {source_path}")
        local_config = self._load_local_config(Path(source_path))

        # Get SSM parameter name from environment variable
        param_name = env.SSM_MODELS_PATH_PARAMETER
        remote_config = self._load_s3_config(
            self._fetch_ssm_parameter(param_name, required=True)
        )
        raw_configs = OmegaConf.merge(local_config, remote_config)

        if not raw_configs:
            raise ModelConfigurationError(
                f"No 'models' section found in configurations from {source_path}"
            )

        self._all_models_config = self._load_and_validate_configs(raw_configs)
        model_count = len(self._all_models_config.models)
        print(f"Successfully loaded {model_count} model configurations.")

    def _load_yaml_configs_from_path(self, path: Path, path_desc: str) -> DictConfig:
        """Loads configurations from YAML files in a directory (local or S3)."""
        all_raw_configs = OmegaConf.create({"models": {}})
        try:
            if not path.is_dir():
                raise ModelConfigurationError(
                    f"{path_desc} '{path}' is not a directory."
                )

            for yaml_file in path.glob("*.yaml"):
                model_raw_config = OmegaConf.create(yaml_file.read_text())
                model_name_key = yaml_file.stem
                all_raw_configs.models[model_name_key] = model_raw_config

            return all_raw_configs
        except Exception as e:
            raise ModelConfigurationError(
                f"An unexpected error occurred loading yaml config from {path_desc} {path}: {e}"
            )

    def _load_local_config(self, config_root: Path) -> DictConfig:
        """Loads configurations from a local YAML file using OmegaConf."""
        return self._load_yaml_configs_from_path(config_root, "local directory")

    def _load_s3_config(self, s3_path_prefix: str) -> DictConfig:
        """Loads configurations from multiple YAML files within an S3 directory."""
        if not s3_path_prefix:
            print("No S3 models path configured, skipping S3 model loading")
            return OmegaConf.create({"models": {}})

        s3_dir = S3Path(s3_path_prefix)
        return self._load_yaml_configs_from_path(s3_dir, "S3 directory")

    def get_model_instance(self, model_name: str) -> BaseLlmModel:
        """Retrieves an instantiated model based on its configuration key."""
        if not self._all_models_config:
            raise ModelConfigurationError(
                "Model configurations not loaded into the factory."
            )

        model_config: ModelConfig = self._all_models_config.models.get(model_name)
        if not model_config:
            raise ModelNotFoundError(f"Config for '{model_name}' not found.")

        # Get provider value (either from enum or string for custom providers)
        provider_key = (
            model_config.provider.value
            if isinstance(model_config.provider, Provider)
            else model_config.provider
        )
        model_class = ModelFactory._model_type_registry.get(provider_key)

        if not model_class:
            raise ModelConfigurationError(
                f"No model class registered for provider '{provider_key}'."
            )

        return model_class(name=model_config.name, config=model_config)


from functools import lru_cache


def _get_llm_cached(model_name_key: str, local_path: str) -> BaseLlmModel:
    factory = ModelFactory(local_path)
    return factory.get_model_instance(model_name_key)


@lru_cache(maxsize=128)
def _get_llm_lru(model_name_key: str, local_path: str) -> BaseLlmModel:
    return _get_llm_cached(model_name_key, local_path)


def get_llm(
    model_name_key: str, local_path: str, force_reload: bool = False
) -> BaseLlmModel:
    """
    Create and retrieve an instantiated LLM.

    Args:
        model_name_key (str): The key identifying the model.
        local_path (Path): The local path to a folder containing a yaml file for each model.
        force_reload (bool): Whether to force reload the the model factory content from S3.

    Returns:
        BaseLlmModel: The instantiated LLM model.
    """
    if force_reload:
        _get_llm_lru.cache_clear()
    return _get_llm_lru(model_name_key, str(local_path))
