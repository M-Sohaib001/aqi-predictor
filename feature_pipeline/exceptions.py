"""Custom exception hierarchy so callers (and CI) can distinguish
'an upstream API failed' from 'the feature store failed' from 'a bug'."""


class AQIPipelineError(Exception):
    """Base exception for all feature pipeline errors."""


class DataFetchError(AQIPipelineError):
    """Raised when an upstream API call fails or returns an unexpected shape."""


class FeatureStoreError(AQIPipelineError):
    """Raised when reading from or writing to the feature store fails."""