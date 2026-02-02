from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from sanctimonia.types import Matrix, Vector, SquareMatrix, SymmetricMatrix


class Solver(ABC):
    @abstractmethod
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None) -> Matrix:
        """
        A: 係数行列
        b: 右辺ベクトル
        x0: 初期値推定（NNやILUが作った良さげな値）
        M: 前処理行列（ILU分解済みの行列など）
        """
        ...


class CGSolver(Solver):
    def solve(self, A: SymmetricMatrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None) -> SquareMatrix:
        return NotImplementedError


class BiCGStabSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None) -> Matrix:
        # ここにBiCGStab法の実装を追加
        return NotImplementedError


class GMRESSolver(Solver):
    def solve(self, A: Matrix, b: Vector, x0: Optional[np.ndarray] = None, M: Optional[np.ndarray] = None) -> Matrix:
        # ここにGMRES法の実装を追加
        return NotImplementedError
