from typing import Annotated, overload

import numpy
from numpy.typing import NDArray
import scipy.sparse


@overload
def solve_cg(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_cg(A: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_bicgstab(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_bicgstab(A: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_lscg(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_lscg(A: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_cg_ilu(A: scipy.sparse.csc_matrix[float], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_cg_ilu(A: scipy.sparse.csc_matrix[complex], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_bicgstab_ilu(A: scipy.sparse.csc_matrix[float], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_bicgstab_ilu(A: scipy.sparse.csc_matrix[complex], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_lscg_ilu(A: scipy.sparse.csc_matrix[float], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_lscg_ilu(A: scipy.sparse.csc_matrix[complex], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)], x0: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')] | None = None, tol: float = 1e-06) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_full_piv_lu(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_full_piv_lu(A: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...

@overload
def solve_partial_piv_lu(A: Annotated[NDArray[numpy.float64], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.float64], dict(shape=(None,), writable=False)]) -> Annotated[NDArray[numpy.float64], dict(shape=(None,), order='C')]: ...

@overload
def solve_partial_piv_lu(A: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), writable=False)], b: Annotated[NDArray[numpy.complex128], dict(shape=(None,), writable=False)]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C')]: ...
