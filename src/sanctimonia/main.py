import numpy as np
from typing import Optional

from scipy import sparse

from . import core
from .cogs.solver import Solver, CGSolver, BiCGStabSolver, LSCGSolver
from .cogs.preprocessor import Preprocessor, ILUPreprocessor
from .types.typing import Matrix, Vector


def solve(
    A: Matrix,
    b: Matrix | Vector,
    method: Solver,
    preprocessor: Optional[Preprocessor] = None,
    tol: float = 1e-6,
    x0: Optional[Matrix | Vector] = None,
    M: Optional[Matrix] = None
) -> Matrix | Vector:
    """
    Solves the linear system Ax = b using the specified method and preprocessor.
    """

    # --- 1. Handle Multiple Right-Hand Sides (Matrix b) ---
    is_multi_rhs = (b.ndim == 2 and b.shape[1] > 1)  # (N, K) with K > 1

    if is_multi_rhs:
        if preprocessor is not None and not isinstance(preprocessor, ILUPreprocessor):
            A, b, x0_pre, M = preprocessor.preprocess(A, b)
            if x0 is None:
                x0 = x0_pre

        # Step 1.2: Solve for each column
        n_cols = b.shape[1]
        x_sols = []

        for i in range(n_cols):
            b_col = b[:, i]

            # Extract corresponding initial guess column if available
            x0_col = None
            if x0 is not None:
                if x0.ndim == 2 and x0.shape[1] == n_cols:
                    x0_col = np.ascontiguousarray(x0[:, i])
                elif x0.ndim == 1 and x0.size == b_col.size:
                    x0_col = x0
                else:
                    # Fallback
                    x0_col = None

            use_preproc = preprocessor if isinstance(preprocessor, ILUPreprocessor) else None

            sol_i = solve(A, b_col, method=method, preprocessor=use_preproc, tol=tol, x0=x0_col, M=M)
            x_sols.append(sol_i)

        return np.column_stack(x_sols)

    # --- 2. Single Right-Hand Side (Vector b) ---
    # Treat (N, 1) matrix as vector
    if b.ndim > 1:
        b = b.flatten()

    if x0 is not None and x0.ndim > 1:
        x0 = x0.flatten()

    # Special handling for ILUPreprocessor (Requires Sparse A and specific C++ bindings)
    if isinstance(preprocessor, ILUPreprocessor):
        A_input = sparse.csc_matrix(A) if not sparse.issparse(A) else A
        match method:
            case CGSolver():
                ilu_solver = core.ILUCGSolverCore(method.num_threads, method.device, method.default_tol, method.max_iterations)
                return ilu_solver.solve_sparse(A_input, b, x0, tol)
            case LSCGSolver():
                ilu_solver = core.ILULSCGSolverCore(method.num_threads, method.device, method.default_tol, method.max_iterations)
                return ilu_solver.solve_sparse(A_input, b, x0, tol)
            case BiCGStabSolver():
                ilu_solver = core.ILUBiCGStabSolverCore(method.num_threads, method.device, method.default_tol, method.max_iterations)
                return ilu_solver.solve_sparse(A_input, b, x0, tol)
            case _:
                # Fallback if solver doesn't support ILU directly in core
                pass

    # Generic Preprocessor Handling (e.g. NNPreprocessor)
    if preprocessor is not None:
        A, b, x0_pre, M = preprocessor.preprocess(A, b)
        if x0 is None:
            x0 = x0_pre

        if x0 is not None and x0.ndim > 1:
            x0 = x0.flatten()

    return method.solve(A, b, x0=x0, M=M, tol=tol)
