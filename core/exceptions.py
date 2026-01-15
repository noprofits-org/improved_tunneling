"""Custom exception hierarchy for improved_tunnel."""


class ImprovedTunnelError(Exception):
    """Base exception for all improved_tunnel errors."""
    pass


class ConfigurationError(ImprovedTunnelError):
    """Raised when configuration is invalid or inconsistent."""
    pass


class ConvergenceError(ImprovedTunnelError):
    """Raised when an iterative procedure fails to converge."""

    def __init__(self, message: str, iterations: int = None,
                 final_value: float = None, threshold: float = None):
        super().__init__(message)
        self.iterations = iterations
        self.final_value = final_value
        self.threshold = threshold


class QChemError(ImprovedTunnelError):
    """Raised when a quantum chemistry calculation fails."""

    def __init__(self, message: str, method: str = None,
                 basis: str = None, error_type: str = None):
        super().__init__(message)
        self.method = method
        self.basis = basis
        self.error_type = error_type


class IntegrationError(ImprovedTunnelError):
    """Raised when numerical integration fails or doesn't converge."""

    def __init__(self, message: str, estimated_error: float = None,
                 tolerance: float = None, subdivisions: int = None):
        super().__init__(message)
        self.estimated_error = estimated_error
        self.tolerance = tolerance
        self.subdivisions = subdivisions


class GeometryError(ImprovedTunnelError):
    """Raised when molecular geometry operations fail."""
    pass


class TurningPointError(ImprovedTunnelError):
    """Raised when classical turning points cannot be found."""

    def __init__(self, message: str, energy: float = None,
                 barrier_height: float = None):
        super().__init__(message)
        self.energy = energy
        self.barrier_height = barrier_height


class FileIOError(ImprovedTunnelError):
    """Raised when file reading or writing fails."""

    def __init__(self, message: str, filepath: str = None):
        super().__init__(message)
        self.filepath = filepath


class WorkflowError(ImprovedTunnelError):
    """Raised when workflow execution fails."""

    def __init__(self, message: str, step: str = None):
        super().__init__(message)
        self.step = step
