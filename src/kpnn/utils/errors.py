"""
Package-specific exception hierarchy.

Why this file exists
--------------------
This file defines the error types that give the package a consistent and
recognizable failure surface. Keeping these exceptions in one place makes
error handling easier to understand, reuse, and extend across the codebase.

Role in the package
-------------------
This is an internal error-definition module. It defines the exception
classes used to signal package-specific problems in validation,
compilation, and interpretation. It should contain the exception
hierarchy itself, not error-producing logic or higher-level API
behavior.
"""

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
