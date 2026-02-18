
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sanctimonia import core
from sanctimonia.cogs.preprocessor import LowFrequencyNNPreprocessor


def create_heavy_test_case(n):
    """
    Creates a 1D Poisson matrix (Tridiagonal) as a dense matrix.
    Diag = 4, Off-diag = -1
    """
    # diagonals
    main_diag = np.ones(n) * 4.0
    off_diag = np.ones(n - 1) * -1.0

    diagonals = [main_diag, off_diag, off_diag]
    offsets = [0, 1, -1]

    A = sp.diags(diagonals, offsets, shape=(n, n), format="csc")
    # Convert to dense array because sanctimonia bindings (except ILU special path) appear to require numpy.ndarray
    return A.toarray()

def create_ill_conditioned_case(n):
    """
    Creates an ill-conditioned matrix (Hilbert matrix approximation).
    """
    # 1. Very small eigenvalues (Hilbert matrix approximation)
    # A_ij = 1 / (i + j + 1)
    indices = np.arange(n)
    A = 1.0 / (indices[:, None] + indices[None, :] + 1.0)

    # 2. Add small value to diagonal to maintain minimal positive definiteness
    # The smaller this value, the worse the condition number
    A += np.eye(n) * 1e-12
    return A


def create_sparse_ill_conditioned_case(n):
    """
    疎行列(ポアソン)をベースに、対角成分を極端に小さくして悪条件化させたもの。
    反復法が低周波成分の誤差を消しにくくなる設定。
    """
    # 1D Laplacian (Poisson) の作成
    main_diag = np.ones(n) * 2.0
    off_diag = np.ones(n - 1) * -1.0

    # ここで悪条件化: 特異行列に近づけるために全体を微小な値でシフト、
    # または特定の対角成分だけを非常に小さくする
    epsilon = 1e-10
    main_diag -= (2.0 - epsilon)  # 固有値の最小値を epsilon 付近まで下げる

    diagonals = [main_diag, off_diag, off_diag]
    offsets = [0, 1, -1]

    A_sparse = sp.diags(diagonals, offsets, shape=(n, n), format="csc")

    # 現状の sanctimonia の制約(dense受け取り)に合わせて変換
    return A_sparse.toarray()


if __name__ == "__main__":

    print("[ITER] Iteration Count, Relative Residual, Elapsed Time (s)")

    model_path = Path(__file__).parent.parent / "models" / "cg_initializer.onnx"
    print(f"Loading model from: {model_path}")
    preprocessor = LowFrequencyNNPreprocessor(model_path, device="cpu")

    N = 30000

    # Keep this example challenging on purpose
    A = create_sparse_ill_conditioned_case(N)
    b = np.ascontiguousarray(np.identity(A.shape[0])[:, 0])
    A = np.ascontiguousarray(A)

    cg_solver = core.CGSolverCore(num_threads=0, device="cpu", tol=1e-3, max_iterations=20000)

    # print config info
    print(f"Matrix size: {A.shape} x {A.shape}")
    print(f"Preprocessor: {preprocessor.__class__.__name__}")
    print(f"Solver: {cg_solver.__class__.__name__}")



    print("="* 10, "Running only CG", "="* 10)
    try:
        cg_solver.solve(A, b, tol=1e-3)
    except RuntimeError as e:
        print(f"CG did not converge ({e}). Trying NN initial guess...")

    print("\n" + "="* 10, "Running CG with NN Preprocessor", "="* 10)
    A_prep, b_prep, X_prep, M_prep = preprocessor.preprocess(A, b)
    b_prep = np.ascontiguousarray(b_prep).reshape(-1)
    x0 = np.ascontiguousarray(X_prep).reshape(-1)

    try:
        cg_solver.solve(np.ascontiguousarray(A_prep), b_prep, x0=x0, tol=1e-3)
    except RuntimeError as e2:
        print(f"CG+NN did not converge ({e2}). Trying ILU preconditioned CG...")
        try:
            A_sparse = sp.csc_matrix(A)
            core.solve_cg_ilu(A_sparse, b, tol=1e-3)
        except RuntimeError as e3:
            print(f"CG+ILU did not converge ({e3}). Falling back to direct LU...")
            core.solve_full_piv_lu(A, b)

    else:
        print("CG converged.")
