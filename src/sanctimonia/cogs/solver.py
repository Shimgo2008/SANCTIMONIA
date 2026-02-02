from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sanctimonia.types import Matrix, Vector, SquareMatrix, SymmetricMatrix
import sanctimonia.core as core


class Solver(ABC):
    @abstractmethod
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> np.ndarray:
        """
        A: 係数行列
        b: 右辺ベクトル
        x0: 初期値推定（NNやILUが作った良さげな値）
        M: 前処理行列（ILU分解済みの行列など）
        tol: 許容誤差
        """
        ...


class CGSolver(Solver):
    def solve(self, A: SymmetricMatrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        return core.solve_cg(A, b, x0, tol)


class BiCGStabSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        return core.solve_bicgstab(A, b, x0, tol)


class LSCGSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        return core.solve_lscg(A, b, x0, tol)


class CholeskySolver(Solver):
    def solve(self, A: SymmetricMatrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        raise NotImplementedError()


class FullPivLUSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Matrix:
        return core.solve_full_piv_lu(A, b)


class PartialPivLUSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Matrix:
        raise NotImplementedError()
