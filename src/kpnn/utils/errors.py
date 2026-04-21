class KPNNError(Exception):
    """Base exception for all KPNN-specific errors."""


class InputValidationError(KPNNError):
    """Raised when public API inputs are invalid."""


class GraphValidationError(KPNNError):
    """Raised when the graph is biologically or structurally invalid."""


class CompilationError(KPNNError):
    """Raised when graph compilation fails."""


class InterpretationError(KPNNError):
    """Raised when model interpretation fails."""
