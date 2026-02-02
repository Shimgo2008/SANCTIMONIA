class SolverError(Exception):
    """Base class for exceptions in the solver module."""
    pass


class NotSquareMatrixError(SolverError):
    """Exception raised for errors in the input when the matrix is not square."""
    def __init__(self, message="The provided matrix is not square."):
        self.message = message
        super().__init__(self.message)


class NotSymmetricMatrixError(SolverError):
    """Exception raised for errors in the input when the matrix is not symmetric."""
    def __init__(self, message="The provided matrix is not symmetric."):
        self.message = message
        super().__init__(self.message)


class ConvergenceError(SolverError):
    """Exception raised when the solver fails to converge."""
    def __init__(self, message="Solver failed to converge", iterations=None, error=None):
        self.iterations = iterations
        self.error = error
        msg = message
        if iterations is not None and error is not None:
            msg = f"{message} (iterations: {iterations}, error: {error})"
        super().__init__(msg)


class DecompositionError(SolverError):
    """Exception raised when the matrix decomposition fails (e.g. zero pivot)."""
    pass
