from pathlib import Path
import scipy.sparse as sp
import numpy as np

from abc import ABC, abstractmethod
from typing import Annotated

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


class ILUPreprocessor(Preprocessor):
    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, x0, M]:
        return


class JacobiPreprocessor(Preprocessor):
    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, x0, M]:
        raise NotImplementedError()


class LowFrequencyNNPreprocessor(Preprocessor):
    def __init__(self, model_path: str | Path, /) -> None:
        self.engine = NNPreprocessor(str(model_path))

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

    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, X, M]:
        A_input, b_input = self._preprocess_matrix(A, b)

        # C++ predict now accepts Matrix and returns Matrix [N, S]
        x0 = self.engine.predict(A_input, b_input)

        # If the original problem was real, x0 might need to be real.
        if not np.iscomplexobj(A) and not np.iscomplexobj(b):
            x0 = np.ascontiguousarray(x0.real, dtype=np.float64)
        else:
            x0 = np.ascontiguousarray(x0, dtype=np.complex128)

        if x0.shape != b.shape:
            print(f"Warning: x0 shape {x0.shape} does not match b shape {b.shape}")

        print(f"Preprocessor output x0 shape: {x0.shape}, dtype: {x0.dtype}")
        return A, b, x0, None
