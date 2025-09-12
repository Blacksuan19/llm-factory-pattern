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

- copy `.env.example` to `.env` and fill the paramters, for defaults check the
  `EnvSetting` class in `config_models.py`

### Creating Custom Providers

1. Create a Python file that defines a class inheriting from `BaseLlmModel`
2. Upload the file to S3 in the path specified by the SSM parameter
3. The file name (without .py) will be used as the provider key

Example custom provider:

```python
# cluade_sonnet_4.py
from llm_factory.models import BaseLlmModel

class CustomLlmModel(BaseLlmModel):
    def __init__(self, name, config):
        super().__init__(name, config)
        # Custom initialization

    def generate_response(self, prompt, **kwargs):
        # Implement response generation
        return "Custom response"
```

now you can load the above model dynamically

```python
from llm_factory import get_llm

# force reload to trigger fetching from S3
model = get_llm("cluade_sonnet_4", force_reload=True)
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

## License

MIT
