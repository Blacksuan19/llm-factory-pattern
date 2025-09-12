class ModelNotFoundError(Exception):
    """Raised when a requested model is not found in the configuration."""

    pass


class ModelConfigurationError(Exception):
    """Raised when there's an issue with model configuration."""

    pass
