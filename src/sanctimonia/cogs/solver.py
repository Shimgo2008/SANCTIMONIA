from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sanctimonia.types import Matrix, Vector, SymmetricMatrix
import sanctimonia.core as core


class Solver(ABC):
    def __init__(self, num_threads: int = 0, device: str = "cpu", tol: float = 1e-6, max_iterations: int = 0) -> None:
        self.num_threads = num_threads
        self.device = device
        self.default_tol = tol
        self.max_iterations = max_iterations

    @abstractmethod
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> np.ndarray:
        """
        A: 係数行列
        b: 右辺ベクトル
        x0: 初期値推定(NNやILUが作った良さげな値)
        M: 前処理行列(ILU分解済みの行列など)
        tol: 許容誤差
        """
        ...


class CGSolver(Solver):
    def __init__(self, num_threads: int = 0, device: str = "cpu", tol: float = 1e-6, max_iterations: int = 0) -> None:
        super().__init__(num_threads=num_threads, device=device, tol=tol, max_iterations=max_iterations)
        self._core_solver = core.CGSolverCore(num_threads=num_threads, device=device, tol=tol, max_iterations=max_iterations)

    def solve(self, A: SymmetricMatrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        effective_tol = tol if tol is not None else self.default_tol
        return self._core_solver.solve(A, b, x0, effective_tol)


class BiCGStabSolver(Solver):
    def __init__(self, num_threads: int = 0, device: str = "cpu", tol: float = 1e-6, max_iterations: int = 0) -> None:
        super().__init__(num_threads=num_threads, device=device, tol=tol, max_iterations=max_iterations)
        self._core_solver = core.BiCGStabSolverCore(num_threads=num_threads, device=device, tol=tol, max_iterations=max_iterations)

    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        effective_tol = tol if tol is not None else self.default_tol
        return self._core_solver.solve(A, b, x0, effective_tol)


class LSCGSolver(Solver):
    def __init__(self, num_threads: int = 0, device: str = "cpu", tol: float = 1e-6, max_iterations: int = 0) -> None:
        super().__init__(num_threads=num_threads, device=device, tol=tol, max_iterations=max_iterations)
        self._core_solver = core.LSCGSolverCore(num_threads=num_threads, device=device, tol=tol, max_iterations=max_iterations)

    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        effective_tol = tol if tol is not None else self.default_tol
        return self._core_solver.solve(A, b, x0, effective_tol)


class CholeskySolver(Solver):
    def solve(self, A: SymmetricMatrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Vector:
        raise NotImplementedError()


class FullPivLUSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Matrix:
        return core.solve_full_piv_lu(A, b)


class PartialPivLUSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None, tol: float = 1e-6) -> Matrix:
        raise NotImplementedError()
