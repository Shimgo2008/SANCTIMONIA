from typing import Optional

from numpy import ndarray
from .cogs import Solver, Preprocessor


def solve(
    A: ndarray,
    b: ndarray,
    method: Solver,
    preprocessor: Optional[Preprocessor] = None,
) -> ndarray:
    """Solve the linear system Ax = b using the specified solver and preprocessor.
    Args:
        A (ndarray): Coefficient matrix.
        b (ndarray): Right-hand side vector.
        method (Solver): Solver to use.
        preprocessor (Optional[Preprocessor]): Preprocessor to use. Defaults to None.
    Returns:
        ndarray: Solution vector x.
    """
    return solve_inverse_matrix_operator(A, b, method, preprocessor)


def solve_inverse_matrix_operator(
    A: ndarray,
    b: ndarray,
    method: Solver,
    preprocessor: Optional[Preprocessor] = None,
) -> ndarray:
    # デフォルトの初期値と前処理
    x0 = None
    M = None

    if preprocessor is not None:
        # Preprocessorは (A, b) を見て、計算を助けるヒントを生成する
        # NNならx0を、ILUならMを、あるいはスケーリング済みのA', b'を返す
        A, b, x0, M = preprocessor.preprocess(A, b)

    # Solverは受け取った武器をフル活用して解く
    return method.solve(A, b, x0=x0, M=M)
