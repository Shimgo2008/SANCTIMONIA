import time
import numpy as np
import scipy.sparse as sp
from pathlib import Path

# sanctimonia imports
from sanctimonia.main import solve
from sanctimonia.cogs.solver import CGSolver, BiCGStabSolver, FullPivLUSolver
from sanctimonia.cogs.preprocessor import ILUPreprocessor, LowFrequencyNNPreprocessor


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


def measure_solver(name, A, b, method, preprocessor=None, tol=1e-6):
    """
    Solves Ax=b using the specified method and preprocessor, measuring time and residual.
    """
    print(f"Running {name:<20} ... ", end="", flush=True)

    # Ensure b is flattened for solver
    b_vec = b.flatten()

    start_time = time.time()
    try:
        x_sol = solve(A, b_vec, method=method, preprocessor=preprocessor, tol=tol)
        elapsed = time.time() - start_time

        # Calculate Relative Residual
        # residual = ||Ax - b|| / ||b||
        res_norm = np.linalg.norm(A @ x_sol - b_vec)
        b_norm = np.linalg.norm(b_vec)
        rel_res = res_norm / (b_norm + 1e-12)  # Avoid div by zero

        print(f"Done. | Time: {elapsed:.6f}s | RelRes: {rel_res:.2e}")
        return {"name": name, "time": elapsed, "residual": rel_res, "status": "Success"}
    except Exception as e:
        print(f"Failed. | Error: {e}")
        return {"name": name, "time": None, "residual": None, "status": f"Failed: {e}"}


def create_sparse_ill_conditioned_case(n):
    """
    疎行列（ポアソン）をベースに、対角成分を極端に小さくして悪条件化させたもの。
    反復法が低周波成分の誤差を消しにくくなる設定。
    """
    # 1D Laplacian (Poisson) の作成
    main_diag = np.ones(n) * 2.0
    off_diag = np.ones(n - 1) * -1.0

    # ここで悪条件化: 特異行列に近づけるために全体を微小な値でシフト、
    # または特定の対角成分だけを非常に小さくする
    epsilon = 1e-10 
    main_diag -= (2.0 - epsilon) # 固有値の最小値を epsilon 付近まで下げる

    diagonals = [main_diag, off_diag, off_diag]
    offsets = [0, 1, -1]

    A_sparse = sp.diags(diagonals, offsets, shape=(n, n), format="csc")

    # 現状の sanctimonia の制約（dense受け取り）に合わせて変換
    return A_sparse.toarray()


def run_benchmark(case_name, A_factory, N, TOL, model_path):
    print(f"\n=== Benchmark: {case_name} (N={N}, tol={TOL}) ===")

    # 1. Prepare Problem
    A = A_factory(N)

    # Create a random solution and compute b
    np.random.seed(42)
    x_true = np.random.rand(N)
    b = A @ x_true

    print(f"Matrix shape: {A.shape}, Non-zeros: {np.count_nonzero(A)}")
    cond_num_check = False  # Very slow for large N, enable if needed for debugging
    if cond_num_check and N <= 500:
        print(f"Condition Number: {np.linalg.cond(A):.2e}")

    results = []

    # (A) BiCGStab (No Preconditioner)
    results.append(
        measure_solver("BiCGStab (None)", A, b, BiCGStabSolver(), None, tol=TOL)
    )

    # (B) CG (ILU Preconditioner)
    results.append(
        measure_solver("CG (ILU)", A, b, CGSolver(), ILUPreprocessor(), tol=TOL)
    )

    # (C) Full LU (Direct Solver)
    results.append(
        measure_solver("Full Piv LU", A, b, FullPivLUSolver(), None, tol=TOL)
    )

    # (D) CG (NN Preconditioner)
    if model_path:
        try:
            nn_preprocessor = LowFrequencyNNPreprocessor(model_path)
            results.append(
                measure_solver(
                    "CG (NN Precond)", A, b, CGSolver(), nn_preprocessor, tol=TOL
                )
            )
        except Exception as e:
            # print(f"Skipping NN Solver due to init error: {e}")
            results.append(
                {
                    "name": "CG (NN Precond)",
                    "time": None,
                    "residual": None,
                    "status": f"Init Failed: {e}",
                }
            )
    else:
        print("Skipping NN Solver (Model not found)")

    # Summary Table for this case
    print(f"\n--- Summary: {case_name} ---")
    print(f"{'Solver':<20} | {'Time (s)':<10} | {'RelRes':<10}")
    print("-" * 46)
    for res in results:
        if res["status"] == "Success":
            print(f"{res['name']:<20} | {res['time']:.6f}   | {res['residual']:.2e}")
        else:
            print(f"{res['name']:<20} | FAILED     | -")


def main():
    # Settings
    N = 2000
    TOL = 1e-8

    # Check Model Path
    model_path = Path("models/cg_initializer.onnx")
    if not model_path.exists():
        if Path("exports/cg_initializer.onnx").exists():
            model_path = Path("exports/cg_initializer.onnx")
        else:
            model_path = None

    # # Run Benchmark for Heavy Case (SPD Poisson)
    # run_benchmark("Heavy (SPD Poisson)", create_heavy_test_case, N, TOL, model_path)

    # # Run Benchmark for Ill-Conditioned Case (Hilbert)
    # run_benchmark("Ill-Conditioned", create_ill_conditioned_case, N, TOL, model_path)

    # Run Benchmark for Sparse Ill-Conditioned Case (Poisson + Bad Diagonal)
    run_benchmark("Sparse Ill-Conditioned", create_sparse_ill_conditioned_case, N, TOL, model_path)


if __name__ == "__main__":
    main()
