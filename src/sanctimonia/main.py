from typing import Optional

import scipy as sp

from . import core
from .cogs.solver import Solver, CGSolver, BiCGStabSolver, LSCGSolver
from .cogs.preprocessor import Preprocessor, ILUPreprocessor
from .types.typing import Matrix, Vector


def solve(
    A: Matrix,
    b: Vector,
    method: Solver,
    preprocessor: Optional[Preprocessor] = None,
    tol: float = 1e-6,
) -> Vector:

    if b.ndim > 1:
        b = b.flatten()

    if isinstance(preprocessor, ILUPreprocessor):
        # ここだけは A を疎行列に強制する
        A_input = sp.csc_matrix(A) if not sp.issparse(A) else A
        match method:
            case CGSolver(): return core.solve_cg_ilu(A_input, b, None, tol)
            case LSCGSolver(): return core.solve_lscg_ilu(A_input, b, None, tol)
            case BiCGStabSolver(): return core.solve_bicgstab_ilu(A_input, b, None, tol)
            case _: raise ValueError("ILUPreprocessor is not compatible with the selected solver.")

    print("Using general solve path.")

    x0, M = None, None
    if preprocessor is not None:
        A, b, x0, M = preprocessor.preprocess(A, b)

    return method.solve(A, b, x0=x0, M=M, tol=tol)
