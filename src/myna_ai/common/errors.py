"""Explicit failures. Callers must not treat these as successful fallbacks."""


class MynaAIError(Exception):
    """Base error for this package."""


class ConfigError(MynaAIError):
    """A configuration file is missing or invalid."""


class DatasetLicenceError(MynaAIError):
    """A dataset licence does not allow the requested use."""


class DatasetValidationError(MynaAIError):
    """A training record failed validation."""


class GpuRequiredError(MynaAIError):
    """The requested training run cannot proceed on the current device."""


class ExportError(MynaAIError):
    """An artifact could not be exported."""


class InferenceError(MynaAIError):
    """Local inference failed."""


class CancelledError(InferenceError):
    """Inference was cancelled."""


class InferenceTimeoutError(InferenceError):
    """Inference exceeded its time limit."""


class AuthorisationError(MynaAIError):
    """A state-changing tool call was not authorised."""


class UnsupportedArchitectureError(MynaAIError):
    """The model architecture does not support the requested operation."""
