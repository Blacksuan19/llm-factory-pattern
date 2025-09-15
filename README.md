# LLM Factory Pattern

A flexible, extensible factory pattern implementation for dynamically loading
LLM providers and models.

## Features

- Singleton factory pattern with force-reload capability
- Dynamic loading of custom provider modules from S3
- Configurable model settings via YAML files (local and S3)
- Seamless caching of model instances for improved performance
- Support for existing providers (OpenAI, Bedrock) and easy extension

## Installation

### Using pip

```bash
pip install git+https://github.com/Blacksuan19/llm-factory-pattern.git
```

### Using uv (recommended)

```bash
uv pip install git+https://github.com/Blacksuan19/llm-factory-pattern.git
```

### Development Installation

```bash
# Clone the repository
git clone https://github.com/Blacksuan19/llm-factory-pattern.git
cd llm-factory-pattern

# Install in development mode
uv pip install -e ".[dev]"
```

## Usage

### Basic Usage

```python
from llm_factory import get_llm

# Get a model by name
model = get_llm("gpt_4o")
print(f"Loaded model: {model.name}")

# Use the model
response = model.invoke("Hello, world!")
print(response)

# Force reload to get a fresh instance
model_reloaded = get_llm("gpt_4o", force_reload=True)

# create a langchain chain

prompt_template = ChatPromptTemplate.from_template(
    "What is the capital of {country}?"
)
chain = prompt_template | model_reloaded.llm

response = chain.invoke({"country": "France"})

```

### Using Built-in Model Configurations

The package includes pre-defined model configurations in the `model_config`, the
argument to `get_llm` is optional and defaults to the value of
`get_default_config_dir()` directory:

```python
from llm_factory import get_llm, model_config

# Get the path to the built-in model configurations
config_dir = model_config.get_default_config_dir()
print(f"Built-in configurations: {model_config.get_default_configs()}")

# Use a built-in model configuration
model = get_llm("gpt_4o", config_dir)
```

### Configuration

- Copy `.env.example` to `.env` and fill the parameters. For defaults, check the
  `EnvSettings` class in `src/config_models.py`.

Environment variables of interest:

```env
# SSM parameters that hold your S3 directories
SSM_PROVIDER_PATH_PARAMETER=/LLM_CONFIG/PROVIDER_MODULES_S3_PATH
SSM_MODELS_PATH_PARAMETER=/LLM_CONFIG/MODELS_CONFIG_S3_PATH

# OpenAI (if using OpenAI provider)
OPENAI_API_KEY=
OPENAI_BASE_URL=
```

### Creating Custom Providers (from S3)

You can add new providers without changing this codebase by placing Python
modules in S3. Each module should define a class that subclasses `BaseLlmModel`.
The file name (without `.py`) becomes the provider key you can reference in your
YAML model configs.

Setup overview:

- Create an SSM Parameter (default name `/LLM_CONFIG/PROVIDER_MODULES_S3_PATH`)
  whose value is an S3 directory path containing your provider modules, e.g.
  `s3://my-bucket/llm/providers/`.
- Ensure your environment has access to SSM and S3 (AWS credentials/roles).
- On first use, or when `force_reload=True` is passed to `get_llm`, provider
  modules are fetched from S3, loaded dynamically, and registered for use.

S3 layout example:

```text
s3://my-bucket/
        ├── models
        │   └── mistral_7b_instruct.yaml
        └── providers
            └── huggingface.py

```

Example custom provider:

```python
# providers/huggingface.py
from models.base_model import BaseLlmModel
from langchain_huggingface import HuggingFaceEndpoint

class HuggingFaceProvider(BaseLlmModel):
    def __init__(self, name, config: ModelConfig):
        super().__init__(name, config)

    def _initialize_llm(self):
        return HuggingFaceEndpoint(
            endpoint_url=self.config.model_id,
            model_kwargs={
                "max_length": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
        )
```

YAML model config example referencing a custom provider by file stem:

```yaml
# models/mistral_7b_instruct.yaml
name: Mistral-7B-Instruct
provider: huggingface # matches huggingface.py
model_id: mistralai/Mistral-7B-Instruct-v0.1
input_token_cost_usd_per_million: 5.0
output_token_cost_usd_per_million: 2.0
max_tokens: 2048
temperature: 0.5
```

Now you can load the above provider dynamically (after uploading the file to the
configured S3 path):

```python
from llm_factory import get_llm

# Force reload to trigger fetching providers and configs from S3
# model name matches the YAML file name without .yaml
model = get_llm("mistral_7b_instruct", force_reload=True)
print(f"Loaded model: {model.name}")

# Use the model
response = model.invoke("Hello, world!")
print(response)
```

## Model Configuration

Models are configured using YAML files in the local path and S3. Example:

```yaml
name: GPT-4o
provider: openai
model_id: gpt-4o
input_token_cost_usd_per_million: 5.0
output_token_cost_usd_per_million: 15.0
max_tokens: 4096
temperature: 0.7
```

### Environment and Parameters

Two optional SSM parameters control remote loading:

- `SSM_PROVIDER_PATH_PARAMETER` (default
  `/LLM_CONFIG/PROVIDER_MODULES_S3_PATH`): S3 directory containing provider
  Python modules.
- `SSM_MODELS_PATH_PARAMETER` (default `/LLM_CONFIG/MODELS_CONFIG_S3_PATH`): S3
  directory containing YAML model config files.

You can override these via environment variables matching their names, or by
setting different values for the parameters themselves.

## License

MIT
