import numpy as np
from typing import Annotated

# 型ヒントのためのエイリアス（実行時はただのnp.ndarray）
Matrix = Annotated[np.ndarray, "2D Array"]
Vector = Annotated[np.ndarray, "1D Array"]

SquareMatrix = Annotated[Matrix, "Square 2D Array"]
SymmetricMatrix = Annotated[Matrix, "Symmetric Square 2D Array"]

Tensor = Annotated[np.ndarray, "N-D Array"]
SquareTensor = Annotated[Tensor, "Square N-D Array"]
SymmetricTensor = Annotated[Tensor, "Symmetric Square N-D Array"]
