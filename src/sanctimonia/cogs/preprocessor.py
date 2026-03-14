from pathlib import Path
import scipy.sparse as sp
import numpy as np

from abc import ABC, abstractmethod
from typing import Annotated, Literal

from sanctimonia.types import Matrix, Vector
from sanctimonia.core import NNPreprocessor

A = Annotated[Matrix, "A"]
b = Annotated[Vector, "b"]
x0 = Annotated[Vector, "x0"]
X = Annotated[Matrix, "X"]
M = Annotated[Matrix, "M"]


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, A, b) -> tuple[A, b, x0, M]:
        pass


class AbstructNNPreprocessor(Preprocessor):
    def __init__(self, model_path: str | Path, /, device: Literal["cpu", "cuda", "auto"]) -> None:
        self.engine = NNPreprocessor(str(model_path), device)

    def _preprocess_matrix(self, A: Matrix, B: Matrix) -> tuple[sp.csc_matrix, np.ndarray]:
        # Convert A to sparse complex matrix
        if not sp.issparse(A):
            A_sparse = sp.csc_matrix(A, dtype=np.complex128)
        else:
            A_sparse = A.tocsc().astype(np.complex128)

        # Convert B to dense complex matrix
        # if B is vector, convert to 2D matrix with shape (N, 1)
        if B.ndim == 1:
            B = B[:, np.newaxis]

        if sp.issparse(B):
            B_converted = B.todense().astype(np.complex128)
        else:
            B_converted = B.astype(np.complex128)

        return A_sparse, B_converted

    @abstractmethod
    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, x0, M]:
        raise NotImplementedError()


class ILUPreprocessor(Preprocessor):
    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, x0, M]:
        return


class JacobiPreprocessor(Preprocessor):
    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, x0, M]:
        raise NotImplementedError()




class LowFrequencyNNPreprocessor(AbstructNNPreprocessor):
    """# 

    Parameters
    ----------
    AbstructNNPreprocessor : _type_
        _description_
    """
    def __init__(self, model_path: str | Path, device: Literal["cpu", "cuda", "auto"] = "auto") -> None:
        super().__init__(model_path, device)

    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, X, M]:
        A_input, b_input = self._preprocess_matrix(A, b)

        print(f"Preprocessor input shape: {A_input.shape}, {b_input.shape}")

        # Check for potential CUDA configuration issues based on edge count
        num_edges = A_input.nnz
        num_systems = b_input.shape[1]

        total_work = num_edges * num_systems
        if total_work > 30_000_000:
            msg = (
                f"GNN Preprocessor Error: Input matrix is too dense or has too many RHS systems.\n"
                f"  Edges (nnz): {num_edges}\n"
                f"  RHS Systems: {num_systems}\n"
                f"  Total projected work: {total_work} (Limit: ~30,000,000)\n"
                "To fix this:\n"
                "  1. Use a sparser matrix (reduce 'nnz').\n"
                "  2. Process RHS columns in smaller batches (reduce 'num_systems')."
            )
            # Use RuntimeError to stop execution before CUDA crash
            raise RuntimeError(msg)

        # C++ predict now accepts Matrix and returns Matrix [N, S]
        x0 = self.engine.predict(A_input, b_input)

        # If the original problem was real, x0 might need to be real.
        if not np.iscomplexobj(A) and not np.iscomplexobj(b):
            x0 = np.ascontiguousarray(x0.real, dtype=np.float64)
        else:
            x0 = np.ascontiguousarray(x0, dtype=np.complex128)

        # Flatten x0 if b was 1D and x0 is (N, 1)
        if b.ndim == 1 and x0.ndim == 2 and x0.shape[1] == 1:
            x0 = x0.flatten()

        if x0.shape != b.shape:
            print(f"Warning: x0 shape {x0.shape} does not match b shape {b.shape}")

        return A, b, x0, None


class SubspaceNNPreprocessor(AbstructNNPreprocessor):
    """
    GNN が生成した基底ベクトル群から部分空間 S=span(V) を作り、
    x0 = V (V^* A V)^(-1) V^* b を構成する preprocessor。

    Notes
    -----
    - ここでの `engine.predict(A, B_seed)` の出力を V とみなす。
    - 数値安定性のため V は QR で列直交化する。
    - E = V^* A V が特異に近い場合は正則化 + 疑似逆行列へフォールバックする。
    """

    def __init__(
        self,
        model_path: str | Path,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        num_basis_vectors: int = 8,
        seed_mode: Literal["canonical", "random"] = "canonical",
        regularization: float = 1e-8,
        random_seed: int = 0,
    ) -> None:
        super().__init__(model_path, device)
        if num_basis_vectors <= 0:
            raise ValueError("num_basis_vectors must be >= 1")
        if regularization < 0:
            raise ValueError("regularization must be >= 0")

        self.num_basis_vectors = num_basis_vectors
        self.seed_mode = seed_mode
        self.regularization = regularization
        self.random_seed = random_seed

        self.last_basis: np.ndarray | None = None
        self.last_coarse_operator: np.ndarray | None = None

    def _build_seed_rhs(self, num_nodes: int, dtype: np.dtype) -> np.ndarray:
        m = min(self.num_basis_vectors, num_nodes)

        if self.seed_mode == "canonical":
            seed_rhs = np.zeros((num_nodes, m), dtype=dtype)
            seed_rhs[np.arange(m), np.arange(m)] = 1.0
            return seed_rhs

        rng = np.random.default_rng(self.random_seed)
        if np.issubdtype(dtype, np.complexfloating):
            return (
                rng.standard_normal((num_nodes, m))
                + 1j * rng.standard_normal((num_nodes, m))
            ).astype(dtype)
        return rng.standard_normal((num_nodes, m)).astype(dtype)

    def _orthonormalize_columns(self, V: np.ndarray) -> np.ndarray:
        # economy QR: V = QR, Q columns are orthonormal basis for span(V)
        Q, _ = np.linalg.qr(V, mode="reduced")
        return Q

    def _project_initial_guess(self, A_sparse: sp.csc_matrix, V: np.ndarray, b_matrix: np.ndarray) -> np.ndarray:
        AV = A_sparse @ V
        coarse = V.conj().T @ AV

        if self.regularization > 0:
            coarse = coarse + self.regularization * np.eye(coarse.shape[0], dtype=coarse.dtype)

        rhs_coarse = V.conj().T @ b_matrix

        try:
            coeff = np.linalg.solve(coarse, rhs_coarse)
        except np.linalg.LinAlgError:
            coeff = np.linalg.pinv(coarse) @ rhs_coarse

        self.last_coarse_operator = coarse
        return V @ coeff

    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, X, M]:
        A_input, b_input = self._preprocess_matrix(A, b)

        num_nodes = A_input.shape[0]
        seed_rhs = self._build_seed_rhs(num_nodes, b_input.dtype)

        V_pred = self.engine.predict(A_input, seed_rhs)
        V_pred = np.ascontiguousarray(V_pred)
        if V_pred.ndim == 1:
            V_pred = V_pred[:, np.newaxis]

        V = self._orthonormalize_columns(V_pred)
        self.last_basis = V

        x0 = self._project_initial_guess(A_input, V, b_input)

        if not np.iscomplexobj(A) and not np.iscomplexobj(b):
            x0 = np.ascontiguousarray(x0.real, dtype=np.float64)
        else:
            x0 = np.ascontiguousarray(x0, dtype=np.complex128)

        if b.ndim == 1 and x0.ndim == 2 and x0.shape[1] == 1:
            x0 = x0.flatten()

        if x0.shape != b.shape:
            print(f"Warning: x0 shape {x0.shape} does not match b shape {b.shape}")

        return A, b, x0, None
