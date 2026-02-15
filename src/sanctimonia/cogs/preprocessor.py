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

    def preprocess(self, A: Matrix, b: Vector) -> tuple[A, b, x0, M]:
        # Convert to types expected by C++ extension (complex sparse matrix and complex vector)
        if not sp.issparse(A):
            A_input = sp.csc_matrix(A, dtype=np.complex128)
        else:
            A_input = A.tocsc().astype(np.complex128)
            
        b_input = np.asarray(b, dtype=np.complex128).flatten()
        
        x0 = self.engine.predict(A_input, b_input)
        
        # If the original problem was real, x0 might need to be real (or cast A, b to complex).
        # However, we don't know if the user intends to use real or complex solver here easily without inspecting A.
        # But if A is real, x0 (complex) will cause type mismatch in strict bindings.
        # Let's try to be smart: if A is real, take real part of x0? 
        # Or should we return complex x0 and let the user handle it?
        # The user's error shows mismatch.
        # Let's check if A is real.
        if not np.iscomplexobj(A) and not np.iscomplexobj(b):
            x0 = np.ascontiguousarray(x0.real, dtype=np.float64)
        else:
            x0 = np.ascontiguousarray(x0, dtype=np.complex128)
            
        return A, b, x0, None
