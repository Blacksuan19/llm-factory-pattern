#!/usr/bin/env python3
"""
Setup script for the llm_factory package.
"""

import os
import shutil

from setuptools import find_packages, setup

# Create a model_config directory in src if it doesn't exist
if not os.path.exists("src/model_config"):
    os.makedirs("src/model_config", exist_ok=True)

# Copy model config files to the src/model_config directory
if os.path.exists("model_config"):
    for file in os.listdir("model_config"):
        if file.endswith(".yaml"):
            shutil.copy(
                os.path.join("model_config", file),
                os.path.join("src/model_config", file),
            )

setup(
    name="llm_factory",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"llm_factory": "src"},
    install_requires=[
        "pydantic",
        "langchain",
        "boto3",
        "langchain_aws",
        "langchain_openai",
        "s3path",
        "omegaconf",
        "pydantic_settings",
        "python-dotenv",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "": ["*.yaml"],
        "llm_factory": ["model_config/*.yaml"],
    },
)
