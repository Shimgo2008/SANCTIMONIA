# /// script
# dependencies = [
#     "numpy",
#     "scipy",
#     "pyamg"
# ]
# [tools.uv.sources]
# sanctimonia = { path = "src/sanctimonia" }
# ///


import gc
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pyamg
import scipy.sparse as sp

# sanctimonia imports
from sanctimonia.main import solve
from sanctimonia.types import Matrix
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


def measure_solver_identity(name, A, N, method, preprocessor=None, tol=1e-6, sample_count=5):
    """
    Estimates or Measures the time to solve AX=I.
    Since solve() now supports matrix b (multiple RHS), we can pass the Identity matrix directly.
    However, for standard iterativ solvers (without batch preprocessor), computing all N columns might be slow.
    If 'preprocessor' is NNPreprocessor, we SHOULD pass full Identity to leverage batch inference.
    """
    print(f"Running {name:<20} (Identity) ... ", end="", flush=True)

    # Use identity matrix as B
    I = np.identity(N)

    is_nn = preprocessor is not None and isinstance(preprocessor, LowFrequencyNNPreprocessor)

    if is_nn:
        # Full solve
        try:
            start_time = time.time()
            X_sol = solve(A, I, method=method, preprocessor=preprocessor, tol=tol)
            elapsed = time.time() - start_time

            indices = np.random.choice(N, size=min(N, 10), replace=False)
            total_res = 0.0
            for col_idx in indices:
                b_vec = I[:, col_idx]
                x_vec = X_sol[:, col_idx]
                res_norm = np.linalg.norm(A @ x_vec - b_vec)
                total_res += res_norm  # b_norm is 1.0

            avg_res = total_res / len(indices)

            print(f"Done. | Time: {elapsed:.4f}s | Avg RelRes: {avg_res:.2e}")
            return {
                "name": f"{name} (Identity)",
                "time": elapsed,
                "residual": avg_res,
                "status": "Success"
            }
        except Exception as e:
            print(f"Failed. | Error: {e}")
            return {
                "name": f"{name} (Identity)",
                "time": None,
                "residual": None,
                "status": f"Failed: {e}"
            }

    else:
        total_time = 0.0
        total_residual = 0.0
        success_count = 0

        is_direct = isinstance(method, FullPivLUSolver)
        n_samples = 1 if is_direct else sample_count

        try:
            for i in range(n_samples):
                b_vec = I[:, i]
                start_time = time.time()
                x_sol = solve(A, b_vec, method=method, preprocessor=preprocessor, tol=tol)
                elapsed = time.time() - start_time
                total_time += elapsed

                res_norm = np.linalg.norm(A @ x_sol - b_vec)
                total_residual += res_norm
                success_count += 1

            avg_time = total_time / success_count
            avg_res = total_residual / success_count
            estimated_total_time = avg_time * N

            print(f"Done. | Est. Time: {estimated_total_time:.4f}s (Approx) | Avg RelRes: {avg_res:.2e}")

            return {
                "name": f"{name} (Identity)",
                "time": estimated_total_time,
                "residual": avg_res,
                "status": "Success (Est)"
            }
        except Exception as e:
            print(f"Failed. | Error: {e}")
            return {
                "name": f"{name} (Identity)",
                "time": None,
                "residual": None,
                "status": f"Failed: {e}"
            }


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
    main_diag -= (2.0 - epsilon)  # 固有値の最小値を epsilon 付近まで下げる

    diagonals = [main_diag, off_diag, off_diag]
    offsets = [0, 1, -1]

    A_sparse = sp.diags(diagonals, offsets, shape=(n, n), format="csc")

    # 現状の sanctimonia の制約（dense受け取り）に合わせて変換
    return A_sparse.toarray()


def run_benchmark(case_name, A_factory, N, TOL, model_path):
    print(f"\n=== Benchmark: {case_name} (N={N}, tol={TOL}) ===")

    # 1. Prepare Problem
    A = A_factory(N)

    # Create a random solution and compute b for single vector test
    np.random.seed(42)
    x_true = np.random.rand(N)
    b = A @ x_true

    print(f"Matrix shape: {A.shape}, Non-zeros: {np.count_nonzero(A)}")
    cond_num_check = False  # Very slow for large N, enable if needed for debugging
    if cond_num_check and N <= 500:
        print(f"Condition Number: {np.linalg.cond(A):.2e}")

    results = []

    # --- Single Vector Benchmarks ---
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
        # print("Skipping NN Solver (Model not found)")
        pass

    # --- Identity Matrix (Multiple RHS) Benchmarks ---
    print("\n--- Identity Matrix Benchmark (Estimating for N columns) ---")

    # (A-Identity) BiCGStab
    results.append(
        measure_solver_identity("BiCGStab (None)", A, N, BiCGStabSolver(), None, tol=TOL)
    )

    gc.collect()

    # (B-Identity) CG (ILU)
    results.append(
        measure_solver_identity("CG (ILU)", A, N, CGSolver(), ILUPreprocessor(), tol=TOL)
    )

    gc.collect()

    # (C-Identity) Full Piv LU
    # For Direct Solver, we just multiply single solve time by N (linear scaling assumption)
    results.append(
        measure_solver_identity("Full Piv LU", A, N, FullPivLUSolver(), None, tol=TOL)
    )

    gc.collect()

    # (D-Identity) CG (NN Preconditioner)
    if model_path:
        try:
            nn_preprocessor = LowFrequencyNNPreprocessor(model_path)
            results.append(
                measure_solver_identity(
                    "CG (NN Precond)", A, N, CGSolver(), nn_preprocessor, tol=TOL
                )
            )
        except Exception as e:
            results.append(
                {
                    "name": "CG (NN Precond) (Identity)",
                    "time": None,
                    "residual": None,
                    "status": f"Init Failed: {e}",
                }
            )

    # Summary Table for this case
    print(f"\n--- Summary: {case_name} ---")
    print(f"{'Solver':<30} | {'Time (s)':<10} | {'RelRes':<10}")
    print("-" * 56)
    for res in results:
        if res["status"].startswith("Success"):
            print(f"{res['name']:<30} | {res['time']:.6f}   | {res['residual']:.2e}")
        else:
            print(f"{res['name']:<30} | FAILED     | -")


def main():
    # Settings
    N = 1000
    TOL = 1e-8

    # Check Model Path
    model_path = Path("models/cg_initializer.onnx")
    if not model_path.exists():
        if Path("exports/cg_initializer.onnx").exists():
            model_path = Path("exports/cg_initializer.onnx")
        else:
            model_path = None

    # # Run Benchmark for Heavy Case (SPD Poisson)
    run_benchmark("Heavy (SPD Poisson)", create_heavy_test_case, N, TOL, model_path)

    gc.collect()

    # # Run Benchmark for Ill-Conditioned Case (Hilbert)
    run_benchmark("Ill-Conditioned", create_ill_conditioned_case, N, TOL, model_path)

    gc.collect()

    # Run Benchmark for Sparse Ill-Conditioned Case (Poisson + Bad Diagonal)
    run_benchmark("Sparse Ill-Conditioned", create_sparse_ill_conditioned_case, N, TOL, model_path)


def test_pyamg():
    N = 1000
    A = create_sparse_ill_conditioned_case(N)
    b = np.random.rand(N)
    A = sp.csc_matrix(A)  # Convert to sparse format for pyamg

    start_time = time.time()
    ml = pyamg.ruge_stuben_solver(A)
    x = ml.solve(b, tol=1e-8)
    elapsed = time.time() - start_time
    res_norm = np.linalg.norm(A @ x - b)
    print(f"Time = {elapsed:.6f}s, RelRes = {res_norm / np.linalg.norm(b):.2e}")


if __name__ == "__main__":
    main()
    # test_pyamg()
