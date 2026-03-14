import random
import sys
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sanctimonia import core
from sanctimonia.cogs.preprocessor import LowFrequencyNNPreprocessor, SubspaceNNPreprocessor
try:
    import pyamg  # type: ignore[import-not-found]
except ImportError:
    pyamg = None

sys.stdout.flush()

def create_oracle_V(A: np.ndarray, num_basis_vectors: int) -> np.ndarray:
    """
    A から num_basis_vectors 個の固有ベクトルを抽出して V を作る。
    これが「オラクル基底」として機能することを期待する。
    """
    n = A.shape[0]
    m = min(num_basis_vectors, max(1, n - 2))  # 固有ベクトルは最大 n-2 個までしか取れない

    # A を疎行列として扱うために CSC 形式に変換
    A_sparse = sp.csc_matrix(A)

    # 固有値問題を解いて最も小さい m 個の固有ベクトルを取得
    # A が対称正定値であれば eigsh を使うと効率的
    v0 = np.linspace(1.0, 2.0, n, dtype=np.float64)
    v0 /= np.linalg.norm(v0)
    _, vecs = sp.linalg.eigsh(A_sparse, k=m, which="SA", v0=v0)

    # 固有ベクトルの符号は任意なので、列ごとに符号を揃えて実験再現性を上げる
    for col in range(vecs.shape[1]):
        pivot_idx = int(np.argmax(np.abs(vecs[:, col])))
        if vecs[pivot_idx, col] < 0:
            vecs[:, col] *= -1.0

    # vecs は n x m の行列で、列が固有ベクトルになっている
    return vecs

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
    return A

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

def create_heterogeneous_ill_conditioned_case(n: int, epsilon=1e-12):
    main_diag = np.full(n, 2.0 + epsilon)
    # 真ん中のセクションだけ極端に結合を弱くする（断熱材のようなイメージ）
    off_diag = np.full(n - 1, -1.0)
    off_diag[n // 2] = -1e-4  # ここでドメインが分断されかけ、低周波が停滞する

    A = sp.diags([main_diag, off_diag, off_diag], [0, 1, -1], format="csc")
    return A


def create_sparse_ill_conditioned_case(n: int, main_diag: float, off_diag: float):
    """
    疎行列(1D Poisson/Tri-diagonal)をベースに、
    SPD を保ったまま最小固有値を小さくして悪条件化させる。

    CG は SPD 前提なので、対角を 2|off_diag| 未満に落とす設定は避ける。
    """
    # Toeplitz tridiagonal の SPD 条件: main_diag > 2 * |off_diag|
    # 境界付近(main_diag ~= 2|off|)に置くと SPD のまま悪条件化できる。
    epsilon = float(os.environ.get("SPD_EPSILON", "1e-6"))
    print(epsilon)
    off_abs = abs(float(off_diag))
    main_diag_spd = max(float(main_diag), 2.0 * off_abs + epsilon)

    main_diag_arr = np.full(n, main_diag_spd, dtype=np.float64)
    off_diag_arr = np.full(n - 1, float(off_diag), dtype=np.float64)

    diagonals = [main_diag_arr, off_diag_arr, off_diag_arr]
    offsets = [0, 1, -1]

    A_sparse = sp.diags(diagonals, offsets, shape=(n, n), format="csc")

    return sp.csc_matrix(A_sparse)


def create_sparse_2d_poisson_case(n_side: int, epsilon=1e-16):
    """
    n_side: 一辺の格子数。行列サイズ N は n_side^2 になる。
    例: n_side=90 なら N=8100
    """
    # 1Dの基本構造 (2+eps, -1, -1)
    d = np.full(n_side, 2.0 + epsilon/2) # 2Dなので1D成分を調整
    off = np.full(n_side - 1, -1.0)
    T = sp.diags([d, off, off], [0, 1, -1])

    # 2Dポアソン: A = T ⊗ I + I ⊗ T
    I = sp.eye(n_side)
    A = sp.kron(T, I) + sp.kron(I, T)

    return sp.csc_matrix(A)


def create_hilbert_matrix(n: int):
    """
    n次のヒルベルト行列を生成する。
    極めて悪条件(ill-conditioned)な正定値対称行列。
    """
    # i, j のインデックスグリッドを作成 (0-indexed)
    i, j = np.meshgrid(np.arange(1, n + 1), np.arange(1, n + 1))

    # ヒルベルト行列の定義: H[i,j] = 1 / (i + j - 1)
    # 0-indexedの場合は 1 / (i + j + 1) と書かれることも多いですが、
    # 本質的な性質（条件数の爆発）は同じです。
    A_hilbert = 1.0 / (i + j - 1.0)

    return A_hilbert


def relative_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    r = b - A @ x
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        return float(np.linalg.norm(r))
    return float(np.linalg.norm(r) / b_norm)


def print_summary(results: list[dict], seed: int | None = None) -> None:
    print("\n" + "=" * 10, "Summary", "=" * 10)
    if seed is not None:
        print(f"Seed: {seed}")
    print(
        f"{'Mode':<28} {'Iter':>8} {'Prep(s)':>10} {'Solve(s)':>10} {'Total(s)':>10} {'RelRes':>12}"
    )
    for row in results:
        iter_str = "-" if row["iterations"] is None else str(row["iterations"])
        print(
            f"{row['name']:<28} {iter_str:>8} {row['prep_time']:>10.4f} "
            f"{row['solve_time']:>10.4f} {row['total_time']:>10.4f} {row['rel_res']:>12.6e}"
        )


if __name__ == "__main__":

    print("[ITER] Iteration Count, Relative Residual, Elapsed Time (s)")

    seed = int(os.environ.get("SEED", "91079"))  # 検証に用いた乱数シード。再現性のため固定値を設定
    np.random.seed(seed)
    random.seed(seed)

    model_variant = os.environ.get("MODEL_VARIANT", "subspace").lower()
    preprocessor_type = os.environ.get("PREPROCESSOR", "subspace").lower()
    model_path = Path(__file__).parent.parent / "models" / model_variant / "cg_initializer.onnx"
    print(f"Loading model from: {model_path}")
    if preprocessor_type == "subspace":
        num_basis_vectors = int(os.environ.get("NUM_BASIS_VECTORS", "8"))
        preprocessor = SubspaceNNPreprocessor(
            model_path,
            device="cpu",
            num_basis_vectors=num_basis_vectors,
        )
    else:
        preprocessor = LowFrequencyNNPreprocessor(model_path, device="cpu")

    N = 4096*2

    n_side = int(np.sqrt(N))
    N_total = n_side**2

    # A の生成 (2D Poisson)
    A = create_sparse_2d_poisson_case(n_side)

    # b の生成 (2D平面に広がる滑らかな sin 波)
    # grid_1d = np.sin(np.linspace(0, np.pi, n_side))
    # grid_2d = np.outer(grid_1d, grid_1d)
    # b = np.ascontiguousarray(grid_2d.flatten(), dtype=np.float64)

    # b の生成)
    b = np.ascontiguousarray(np.identity(A.shape[0])[:, 0])
    is_sparse = sp.issparse(A)
    print(f"is A sparse? {sp.issparse(A)}")
    A_csr = A if sp.issparse(A) else sp.csr_matrix(A)

    cg_solver = core.CGSolverCore(num_threads=1, device="cpu", tol=1e-3, max_iterations=20000)
    tol = 1e-11
    max_iterations = 20000

    # print config info
    print(f"Matrix size: {A.shape} x {A.shape}")
    print(f"Preprocessor: {preprocessor.__class__.__name__}")
    print(f"Solver: {cg_solver.__class__.__name__}")
    print("Comparison modes: CG / CG+NN x0 / CG+Oracle x0 / PyAMG solve / CG+PyAMG preconditioner")

    results: list[dict] = []


    print("="* 10, "Running only CG", "="* 10)
    try:
        t0 = perf_counter()
        x_cg = cg_solver.solve_sparse(A_csr, b, tol=tol) if is_sparse else cg_solver.solve(A, b, tol=tol)
        solve_time = perf_counter() - t0
        results.append(
            {
                "name": "CG (x0=0)",
                "iterations": None,
                "prep_time": 0.0,
                "solve_time": solve_time,
                "total_time": solve_time,
                "rel_res": relative_residual(A, b, x_cg),
            }
        )
    except RuntimeError as e:
        print(f"CG did not converge ({e}). Trying NN initial guess...")


    print("\n" + "="* 10, "Running CG with NN Preprocessor", "="* 10)
    prep_t0 = perf_counter()
    A_prep, b_prep, X_prep, M_prep = preprocessor.preprocess(A, b) 
    prep_time = perf_counter() - prep_t0

    b_prep = np.ascontiguousarray(b_prep).reshape(-1)
    x0 = np.ascontiguousarray(X_prep).reshape(-1)

    try:
        t0 = perf_counter()
        x_cg_nn = cg_solver.solve_sparse(A_prep, b_prep, x0=x0, tol=tol)
        solve_time = perf_counter() - t0
        
        results.append({
            "name": "CG + NN initial guess",
            "iterations": None,
            "prep_time": prep_time,
            "solve_time": solve_time,
            "total_time": prep_time + solve_time,
            "rel_res": relative_residual(A_prep, b_prep, x_cg_nn),
        })
    except RuntimeError as e2:
        print(f"CG+NN did not converge ({e2}). Trying ILU preconditioned CG...")
        try:
            # A = sp.csc_matrix(A)
            core.solve_cg_ilu(A, b, tol=tol)
        except RuntimeError as e3:
            print(f"CG+ILU did not converge ({e3}). Falling back to direct LU...")
            core.solve_full_piv_lu(A, b)

    print("\n" + "="* 10, "Testing Oracle V", "="* 10)

    start = perf_counter()
    V = create_oracle_V(A, num_basis_vectors=8)
    x0_oracle = preprocessor._project_initial_guess(sp.csc_matrix(A), V, b)

    oracle_prep_time = perf_counter() - start
    print(f"Oracle V shape: {V.shape[1]} basis vectors")
    print(f"Time to compute oracle V: {oracle_prep_time} seconds")

    try:
        t0 = perf_counter()
        x_cg_oracle = cg_solver.solve_sparse(A, b, x0=x0_oracle, tol=tol) if sp.issparse(A) else cg_solver.solve(np.ascontiguousarray(A), b, x0=x0_oracle, tol=tol)
        solve_time = perf_counter() - t0
        results.append(
            {
                "name": "CG + Oracle initial guess",
                "iterations": None,
                "prep_time": oracle_prep_time,
                "solve_time": solve_time,
                "total_time": oracle_prep_time + solve_time,
                "rel_res": relative_residual(A, b, x_cg_oracle),
            }
        )
    except RuntimeError as e4:
        print(f"CG with oracle V did not converge ({e4}). Trying ILU preconditioned CG...")
        try:
            core.solve_cg_ilu(A, b, tol=tol)
        except RuntimeError as e5:
            print(f"CG+ILU did not converge ({e5}). Falling back to direct LU...")
            core.solve_full_piv_lu(A, b)
    except Exception as e:
        print(f"Unexpected error with oracle V: {e}. Skipping oracle test.")

    if pyamg is None:
        print("\n" + "="* 10, "PyAMG", "="* 10)
        print("pyamg is not installed. Run with: uv run --with pyamg example/test_ident.py")
    else:
        print("\n" + "="* 10, "PyAMG Solve (standalone)", "="* 10)
        pyamg_setup_t0 = perf_counter()
        ml = pyamg.ruge_stuben_solver(A_csr)
        pyamg_setup_time = perf_counter() - pyamg_setup_t0
        pyamg_solve_t0 = perf_counter()
        x_pyamg = ml.solve(b, tol=tol)
        pyamg_solve_time = perf_counter() - pyamg_solve_t0
        results.append(
            {
                "name": "PyAMG solve",
                "iterations": None,
                "prep_time": pyamg_setup_time,
                "solve_time": pyamg_solve_time,
                "total_time": pyamg_setup_time + pyamg_solve_time,
                "rel_res": relative_residual(A, b, np.asarray(x_pyamg).reshape(-1)),
            }
        )

        print("\n" + "="* 10, "CG + PyAMG Preconditioner", "="* 10)
        m_prec = ml.aspreconditioner()
        cg_iters = [0]

        def _count_iters(_xk: np.ndarray) -> None:
            cg_iters[0] += 1

        pyamg_cg_t0 = perf_counter()
        x_pyamg_cg, info = spla.cg(
            A,
            b,
            M=m_prec,
            rtol=tol,
            maxiter=max_iterations,
            callback=_count_iters,
        )
        pyamg_cg_time = perf_counter() - pyamg_cg_t0

        if info > 0:
            print(f"SciPy CG reached max iterations without convergence: info={info}")
        elif info < 0:
            print(f"SciPy CG failed with illegal input/breakdown: info={info}")

        results.append(
            {
                "name": "CG + PyAMG preconditioner",
                "iterations": cg_iters[0],
                "prep_time": pyamg_setup_time,
                "solve_time": pyamg_cg_time,
                "total_time": pyamg_setup_time + pyamg_cg_time,
                "rel_res": relative_residual(A, b, np.asarray(x_pyamg_cg).reshape(-1)),
            }
        )

    print_summary(results, seed=seed)
    print("Finish!")