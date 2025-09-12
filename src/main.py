from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

from model_config import get_default_config_dir
from model_factory import ModelNotFoundError, get_llm


def run_demo(model_repository_path: str):
    """Demonstrates accessing LLMs using the get_llm function."""

    print(f"\n--- Demonstrating from {model_repository_path} ---")
    # The first call to get_llm will trigger configuration loading and instantiation
    claude_model = get_llm("claude_sonnet_3_7", model_repository_path)
    print(f"Retrieved model: {claude_model.name} ({claude_model.__class__.__name__})")
    print(f"Claude sonnet Input Cost: ${claude_model.config.input_token_cost}/M tokens")

    # Demonstrate direct LangChain usage via the .llm property
    print("\n--- Demonstrating direct LangChain usage for Claude ---")
    prompt_template = ChatPromptTemplate.from_messages(
        [("human", "Tell me a short, funny story about a programmer and a duck.")]
    )
    chain = prompt_template | claude_model.llm
    # Uncomment the following lines to attempt actual API calls (requires valid AWS credentials)
    # try:
    #     response = chain.invoke({})
    #     print(f"Claude (LangChain Chain) Response: {response.content[:100]}...")
    # except Exception as e:
    #     print(f"Error generating Claude response via chain: {e}")

    # Requesting the same model again with the same source should return the cached instance
    print("\n--- Requesting Claude again with same source to demonstrate caching ---")
    claude_model_cached = get_llm("claude_sonnet_3_7", model_repository_path)
    print(
        f"Retrieved model (cached): {claude_model_cached.name} ({claude_model_cached.__class__.__name__})"
    )
    print(f"Is it the same instance? {claude_model is claude_model_cached}")

    # Get an OpenAI GPT-4o model from the same source
    gpt_model = get_llm("gpt_4o", model_repository_path)
    print(f"Retrieved model: {gpt_model.name} ({gpt_model.__class__.__name__})")
    print(f"GPT-4o Output Cost: ${gpt_model.config.output_token_cost}/M tokens")

    # Demonstrate direct LangChain usage for GPT-4o
    print("\n--- Demonstrating direct LangChain usage for GPT-4o ---")
    prompt_template_gpt = ChatPromptTemplate.from_template(
        "What is the capital of {country}?"
    )
    chain_gpt = prompt_template_gpt | gpt_model.llm
    # Uncomment the following lines to attempt actual API calls (requires valid OPENAI_API_KEY)
    try:
        response_gpt = chain_gpt.invoke({"country": "France"})
        print(f"GPT-4o (LangChain Chain) Response: {response_gpt.content}")
    except Exception as e:
        print(f"Error generating GPT-4o response via chain: {e}")

    # Get a Bedrock Llama model from the same source
    llama_model = get_llm("llama_3_8b_instruct", model_repository_path)
    print(f"Retrieved model: {llama_model.name} ({llama_model.__class__.__name__})")
    print(f"Llama Input Cost: ${llama_model.config.input_token_cost}/M tokens")
    # Uncomment the following lines to attempt actual API calls (requires valid AWS credentials)
    # try:
    #     response = llama_model.generate_response("Write a haiku about Python.")
    #     print(f"Llama Response: {response[:50]}...")
    # except Exception as e:
    #     print(f"Error generating Llama response: {e}")

    # Try to get a non-existent model
    try:
        get_llm("non_existent_model", model_repository_path)
    except ModelNotFoundError as e:
        print(f"Caught expected error: {e}")


if __name__ == "__main__":

    load_dotenv()
    run_demo(get_default_config_dir())
