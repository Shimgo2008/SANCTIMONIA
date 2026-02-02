from abc import ABC, abstractmethod
from typing import Annotated

from sanctimonia.types import Matrix, Vector

A = Annotated[Matrix, "A"]
b = Annotated[Vector, "b"]
x0 = Annotated[Vector, "x0"]
M = Annotated[Matrix, "M"]


class Preprocessor(ABC):
    @abstractmethod
    def preprocess(self, A, b) -> tuple[A, b, x0, M]:
        pass


class NNPreprocessor(Preprocessor):
    def preprocess(self, A, b) -> tuple[A, b, x0, M]:
        raise NotImplementedError()


class ILUPreprocessor(Preprocessor):
    def preprocess(self, A, b) -> tuple[A, b, x0, M]:
        return


class JacobiPreprocessor(Preprocessor):
    def preprocess(self, A, b) -> tuple[A, b, x0, M]:
        raise NotImplementedError()


class LowFrequencyNNPreprocessor(NNPreprocessor):
    def preprocess(self, A, b) -> tuple[A, b, x0, M]:
        raise NotImplementedError()
