class SolverError(Exception):
    """Base class for exceptions in the solver module."""
    pass


class isNotSquareMatrixError(SolverError):
    """Exception raised for errors in the input when the matrix is not square."""
    def __init__(self, message="The provided matrix is not square."):
        self.message = message
        super().__init__(self.message)


class isNotSymmetricMatrixError(SolverError):
    """Exception raised for errors in the input when the matrix is not symmetric."""
    def __init__(self, message="The provided matrix is not symmetric."):
        self.message = message
        super().__init__(self.message)
