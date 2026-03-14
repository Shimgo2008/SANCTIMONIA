import sys
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.linalg as la
import matplotlib.pyplot as plt
from pathlib import Path
import time

# PyAMG is optional but recommended for full comparison
try:
    import pyamg
    HAS_PYAMG = True
except ImportError:
    HAS_PYAMG = False
    print("Warning: PyAMG not installed. AMG solver will be skipped. (pip install pyamg)")

# Add parent directory to path to import sanctimonia
sys.path.append(str(Path(__file__).parent.parent))

from sanctimonia.cogs.preprocessor import LowFrequencyNNPreprocessor
import sanctimonia.core as core
from sanctimonia.cogs.solver import CGSolver, BiCGStabSolver

# ==========================================
# 1. Problem Generation
# ==========================================

def create_poisson_2d(n):
    """
    Generate 2D Poisson matrix on n x n grid.
    Total size N = n*n.
    Standard 5-point stencil: 4 on diag, -1 on neighbors.
    Condition number ~ N (since N = n^2). Actually cond ~ (n)^2.
    """
    N = n * n
    main_diag = np.full(N, 4.0)
    off_diag = np.full(N - 1, -1.0)
    
    # Off-diagonal blocks for 2D connectivity (distance n)
    off_diag_n = np.full(N - n, -1.0)
    
    # Correct the immediate off-diagonal to not wrap around rows
    # i.e., node (i, n-1) should not connect to (i+1, 0)
    for i in range(1, n):
        off_diag[i*n - 1] = 0.0
        
    diagonals = [main_diag, off_diag, off_diag, off_diag_n, off_diag_n]
    offsets = [0, 1, -1, n, -n]
    
    return sp.diags(diagonals, offsets, shape=(N, N), format="csc")

def create_poisson_2d_ill_shifted(n):
    """
    2D Poisson on n x n grid, shifted to be ill-conditioned.
    Smallest eigenvalue of 2D Laplacian on unit square is 2*pi^2 (approx).
    For discrete, lambda_min = 4 - 2*cos(pi/(n+1)) - 2*cos(pi/(n+1))
                             = 4 * sin^2(pi/(2(n+1))) * 2 
    """
    N = n * n
    # Theoretical min eigenvalue for 2D Laplacian (4, -1)
    lambda_min_poisson = 8.0 * (np.sin(np.pi / (2 * (n + 1))) ** 2)
    
    epsilon = 1e-6 # Stricter shift to ensure SPD but VERY ill
    shift = lambda_min_poisson - epsilon
    
    # Base 2D Poisson
    A = create_poisson_2d(n)
    # Subtract shift from diagonal
    A.setdiag(A.diagonal() - shift)
    
    return A

def create_poisson_2d_extreme_user(n):
    """
    """
    N = n * n


    main_diag = np.ones(N) * 2.0 
    off_diag = np.ones(N - 1) * -1.0
    
    epsilon = 1e-10
    main_diag -= (2.0 - epsilon)
    
    diagonals = [main_diag, off_diag, off_diag]
    offsets = [0, 1, -1]
    
    A_sparse = sp.diags(diagonals, offsets, shape=(N, N), format="csc")
    return np.ascontiguousarray(A_sparse.toarray(), dtype=np.float64)


# Registry of problem generators
# Key: Name used in reports
# Value: Function(n) -> sparse_matrix (where N = n*n)
PROBLEM_GENERATORS = {
    'Well-Conditioned (2D Poisson)': create_poisson_2d,
    'Ill-Conditioned (2D Shifted)': create_poisson_2d_ill_shifted,
    'Extreme Ill-Conditioned (2D User)': create_poisson_2d_extreme_user
}

def compute_exact_eigen(A_dense):
    # print("   Computing Exact Eigendecomposition...")
    return la.eigh(A_dense)

# ==========================================
# 2. Solvers & Helpers
# ==========================================

def compute_a_norm(e, A):
    """Computes ||e||_A = sqrt(e^T A e). Handles negative A-norms (indefinite) by returning NaN or Warning."""
    val = np.dot(e, A @ e)
    if val < 0:
        # For indefinite matrices, A-norm is not defined.
        # This happens in the 'Extreme Ill-Conditioned' user case where the matrix is actually Indefinite.
        return np.nan 
    return np.sqrt(val)

def captured_cg_solve(A, b, x0=None, M=None, max_iter=None, tol=1e-8, name="CG", solver_impl=None):
    """
    Wrapper to solve using either SciPy's CG or Sanctimonia's CGSolver.
    Since Sanctimonia (Eigen) doesn't return iteration history, we can't plot the curve.
    "Obediently" using Sanctimonia means we use the library as provided.
    However, plotting a straight line is misleading.
    We will just return [init, final] for Sanctimonia.
    """
    N = len(b)
    if max_iter is None: max_iter = N
    
    start_time = time.time()
    residuals = []
    
    # Pre-calculate initial residual
    if x0 is None:
        r0 = b
    else:
        # A might be sparse
        if sp.issparse(A):
            r0 = b - A @ x0
        else:
            r0 = b - A.dot(x0)
    
    init_res = np.linalg.norm(r0)
    residuals.append(init_res)
    
    x = None
    
    if solver_impl == 'sanctimonia':
        # Use Sanctimonia CGSolver
        # We need to instantiate the solver class
        # Assuming sanctimonia.cogs.solver.CGSolver exists and wraps core
        try:
             # Import locally to avoid circular dependency issues if any
             from sanctimonia.cogs.solver import CGSolver
             
             # Sanctimonia CGSolver takes (num_threads, device, tol, max_iterations)
             s_solver = CGSolver(max_iterations=max_iter, tol=tol)
             
             # Solve!
             # Note: Sanctimonia bindings for dense solver expect dense arrays (numpy)
             if sp.issparse(A):
                 A_in = A.toarray()
             else:
                 A_in = A
                 
             # x0 handling
             x0_in = x0 if x0 is not None else np.zeros(N)
             
             # The error suggests implicit conversion failure or argument mismatch.
             # Ensure inputs are contiguous and correct type.
             A_in = np.ascontiguousarray(A_in, dtype=np.float64)
             b = np.ascontiguousarray(b, dtype=np.float64)
             x0_in = np.ascontiguousarray(x0_in, dtype=np.float64)

             # Call solve. The wrapper class CGSolver.solve wraps core.solve.
             # CGSolver.solve(A, b, x0, M, tol)
             # Core solve(A, b, x0, tol)
             # We invoke the wrapper here. Wrapper definition:
             # def solve(self, A, b, x0=None, M=None, tol=1e-6)
             
             # The error suggests we are passing arguments that don't match the signature.
             # In python, keyword arguments are often used, but here 'M' is positional in definition 
             # if we don't name x0. But we name x0.
             
             # Let's call explicitly with keyword args matching the CGSolver.solve signature
             # solve(A, b, x0=None, M=None, tol=None)
             x = s_solver.solve(A_in, b, x0=x0_in, tol=tol)
             
        except Exception as e:
             # If invoking via CGSolver.solve fails, try invoking the core solver directly to debug.
             # The core solver signature is slightly different: solve(A, b, x0, tol)
             # (no M argument)
             try:
                 print(f"Wrapper failed ({e}), trying core direct...")
                 core_solver = core.CGSolverCore(max_iterations=max_iter, tol=tol)
                 # Core expects exactly (A, b, x0, tol)
                 x = core_solver.solve(A_in, b, x0_in, tol)
             except Exception as e2:
                 print(f"Sanctimonia Core Solver also Failed: {e2}")
                 x = x0 if x0 is not None else np.zeros(N)
             
    else:
        # Standard SciPy
        def callback(xk):
            r = b - A @ xk
            residuals.append(np.linalg.norm(r))

        kwargs = {
            'maxiter': max_iter,
            'M': M,
            'x0': x0,
            'callback': callback
        }
        
        # Check tolerance arg
        import inspect
        sig = inspect.signature(spla.cg)
        if 'rtol' in sig.parameters:
            kwargs['rtol'] = tol
        else:
            kwargs['tol'] = tol
            
        x, info = spla.cg(A, b, **kwargs)

    if x is None:
        x = x0 if x0 is not None else np.zeros(N)

    # Compute Final Residual
    if sp.issparse(A):
        final_r = b - A @ x
    else:
        final_r = b - A.dot(x)
        
    final_res_norm = np.linalg.norm(final_r)
    
    # For Sanctimonia, we append final to make a list of 2 points [start, end]
    # This will plot as a line, which is better than nothing but obviously not the full story.
    if solver_impl == 'sanctimonia':
         residuals.append(final_res_norm)
    elif not residuals: # SciPy converged instantly (0 iters) or failed
         residuals.append(final_res_norm)
         
    duration = time.time() - start_time
    return x, residuals, duration



# ==========================================
# 3. Main Benchmark Logic
# ==========================================

def run_benchmark(N, model_path, export_dir):
    print(f"=== Starting Comprehensive Benchmark (N_total={N} -> ~{int(np.sqrt(N))}x{int(np.sqrt(N))}) ===\n")
    
    # Adjust N to be square number for 2D grid
    n = int(np.sqrt(N))
    N = n * n
    print(f"Using {n}x{n} grid (N={N})")
    
    # Pre-load GNN Model if available
    preprocessor = None
    if model_path.exists():
        print(f"Loading GNN Model from {model_path}...")
        try:
            preprocessor = LowFrequencyNNPreprocessor(model_path, device="cpu")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print(f"Warning: Model not found at {model_path}")

    for case_name, generator in PROBLEM_GENERATORS.items():
        print(f"\n--- Running Case: {case_name.upper()} ---")
        
        # Generator for 2D takes grid size 'n' NOT total 'N'
        # The generator functions are defined as expecting 'n' now.
        # But wait, original code passed N. Let's fix generator calls.
        # create_poisson_2d takes n (grid side).
        
        # 1. Setup Data
        A_sparse = generator(n)
        
        # Ensure sparse for ILU/AMG, dense for Sanctimonia
        if not sp.issparse(A_sparse):
            A_dense = A_sparse
            A_sparse = sp.csc_matrix(A_dense)
        else:
            A_dense = A_sparse.toarray()
            A_sparse = A_sparse.tocsc()
        
        # Check actual sizes match
        assert A_sparse.shape[0] == N, f"Size mismatch: {A_sparse.shape[0]} != {N}"
        
        # Eigen decomposition for analysis
        eigvals, eigvecs = compute_exact_eigen(A_dense)
        
        # Condition number: max(|lambda|) / min(|lambda|)
        abs_eigvals = np.abs(eigvals)
        cond_num = abs_eigvals.max() / (abs_eigvals.min() + 1e-16)
        
        print(f"   Condition Number: {cond_num:.2e}")
        print(f"   Eigenvalues Range: [{eigvals[0]:.2e}, {eigvals[-1]:.2e}]")
        
        if eigvals[0] < 0:
            print(f"   Note: Matrix is Indefinite (Min Eigenvalue < 0)")
        
        # RHS: Random to excite all modes
        np.random.seed(42)
        b = np.random.randn(N)
        b = b / np.linalg.norm(b) # Normalize RHS
        
        # Exact solution for error analysis
        x_true = la.solve(A_dense, b)

        # 2. Setup Solvers
        solvers = []
        
        # Solver 1: Standard CG (None, Zero guess)
        solvers.append({
            'name': 'Standard CG',
            'M': None,
            'x0': np.zeros(N),
            'color': 'gray',
            'style': '--'
        })

        # Solver 2: AI-CG (GNN Initial Guess)
        if preprocessor:
            print("   Running GNN Inference...")
            start_nn = time.time()
            # GNN preprocess
            try:
                # Assuming signature preprocess(A, b) -> (A, b, x, M)
                _, _, x_pred, _ = preprocessor.preprocess(A_dense, b)
                x0_nn = np.ascontiguousarray(-x_pred, dtype=np.float64).reshape(-1)
                nn_time = time.time() - start_nn
                
                solvers.append({
                    'name': 'AI-CG (GNN + Sanctimonia)',
                    'M': None,
                    'x0': x0_nn,
                    'color': 'green',
                    'style': '-',
                    'overhead': nn_time,
                    'solver_impl': 'sanctimonia'
                })
            except Exception as e:
                print(f"   GNN Failed: {e}")

        # Solver 3: AMG-CG
        if HAS_PYAMG:
            try:
                A_csr = A_sparse.tocsr()
                ml = pyamg.smoothed_aggregation_solver(A_csr)
                M_amg = ml.aspreconditioner(cycle='V')
                solvers.append({
                    'name': 'AMG-CG',
                    'M': M_amg,
                    'x0': np.zeros(N),
                    'color': 'purple',
                    'style': '-'
                })
            except Exception as e:
                print(f"   AMG Setup Failed: {e}")

        # Solver 4: Sanctimonia Native CG (Eigen C++)
        # We force this one to run to compare performance/convergence
        # Note: It will likely just show Start -> End line due to missing history
        try:
             solvers.append({
                'name': 'Sanctimonia-CG',
                'M': None,
                'x0': np.zeros(N),
                'color': 'red',
                'style': '--',
                'solver_impl': 'sanctimonia'
            })
        except Exception as e:
            print(f"   Sanctimonia Setup Failed: {e}")

        # 3. Run All Solvers
        results = []
        # CG theoretically finishes in N steps (exact arithmetic), but practically needs N or slightly more.
        # User reports convergence exactly at N-th step or around it.
        # We set max_steps slightly larger than N to capture this behavior.
        max_steps = N + 20 
        
        for s in solvers:
            print(f"   Running {s['name']} (max_iter={max_steps})...")
            M_op = s.get('M')
            x0_vec = s.get('x0')
            solver_impl = s.get('solver_impl', 'scipy')
            
            x_sol, res_hist, duration = captured_cg_solve(
                A_dense, b, x0=x0_vec, M=M_op, 
                max_iter=max_steps, name=s['name'],
                solver_impl=solver_impl
            )
            
            # Analyze final error
            e_final = x_true - x_sol
            
            final_res = res_hist[-1]
            iters = len(res_hist) - 1
            l2_err = np.linalg.norm(e_final)
            anorm_err = compute_a_norm(e_final, A_dense)
            
            print(f"      -> Iterations: {iters:3d} | Residual: {final_res:.2e} | L2 Error: {l2_err:.2e} | A-Norm Error: {anorm_err:.2e} | Time: {duration*1000:.1f}ms")
            
            # Additional detail for last few iterations (User specifically mentioned "sudden drop at N")
            if iters >= 5:
                start_idx = max(0, iters - 5)
                tail_res = res_hist[start_idx:]
                tail_str = ", ".join([f"{r:.2e}" for r in tail_res])
                print(f"         Last 5 Residuals: [{tail_str}]")
            
            # Additional Analysis: Spectral Snapshot at Iter K=10 (to see filtering effect)
            # We re-run briefly or just use final if converged fast.
            # To show spectral filtering, let's look at error after exactly 10 iterations
            # (or fewer if converged)
            snapshot_iter = min(10, len(res_hist)-1)
            x_snap, _, _ = captured_cg_solve(A_dense, b, x0=x0_vec, M=M_op, max_iter=snapshot_iter)
            e_snap = x_true - x_snap
            coeffs_snap = np.abs(eigvecs.T @ e_snap)
            
            # A-norm of initial error
            e_init = x_true - x0_vec
            anorm_init = compute_a_norm(e_init, A_dense)
            
            s['results'] = {
                'x_final': x_sol,
                'residuals': res_hist,
                'duration': duration + s.get('overhead', 0),
                'coeffs_snap': coeffs_snap,
                'snapshot_iter': snapshot_iter,
                'anorm_init': anorm_init
            }
        
        # 4. Visualization
        # Use simple filenames for compatibility
        safe_case_name = case_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        
        plot_results(results_map=solvers, 
                     eigvals=eigvals, 
                     eigvecs=eigvecs, 
                     case_name=case_name, 
                     export_dir=export_dir,
                     filename_suffix=safe_case_name,
                     N=N,
                     x_true=x_true, # For Zero-error reference
                     b=b
                     )

def plot_results(results_map, eigvals, eigvecs, case_name, export_dir, filename_suffix, N, x_true, b):
    print(f"   Plotting results for {case_name}...")
    
    # Setup Figure Grid
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3)
    
    ax_conv = fig.add_subplot(gs[0, 0])      # Convergence
    ax_spec = fig.add_subplot(gs[0, 1])      # Spectral Error (Absolute)
    ax_att  = fig.add_subplot(gs[0, 2])      # Attenuation Factor
    ax_anorm = fig.add_subplot(gs[1, 0])     # Initial A-Norm Comparison
    ax_time = fig.add_subplot(gs[1, 1])      # Wall-time
    
    # Reference Error (Zero Guess) Analysis
    e_zero = x_true - np.zeros_like(b)
    coeffs_zero = np.abs(eigvecs.T @ e_zero)
    
    # 1. Convergence Plot
    for s in results_map:
        res = s['results']['residuals']
        label = s['name']
        # Normalize relative to initial residual of Standard CG (Zero guess)
        # to make comparisons fair
        # norm_factor = res[0] 
        # Actually better to plot raw or relative to self. Let's use relative to ||b||
        norm_b = np.linalg.norm(b)
        ax_conv.semilogy(np.array(res)/norm_b, label=label, color=s['color'], linestyle=s['style'], alpha=0.8)

    ax_conv.set_title(f"Convergence History (RelRes)")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("||r_k|| / ||b||")
    ax_conv.grid(True, alpha=0.3)
    ax_conv.legend(fontsize='small')
    
    # 2. Spectral Error (Snapshot)
    # Plot Zero Reference
    ax_spec.semilogy(coeffs_zero, color='black', alpha=0.2, linestyle=':', label='Initial Error (Zero)')
    
    for s in results_map:
        coeffs = s['results']['coeffs_snap']
        k = s['results']['snapshot_iter']
        ax_spec.semilogy(coeffs, color=s['color'], alpha=0.7, label=f"{s['name']} (Iter {k})")
        
        # 3. Attenuation (Ratio vs Zero Initial)
        # Avoid div by zero
        ratio = coeffs / (coeffs_zero + 1e-16)
        ax_att.semilogy(ratio, color=s['color'], alpha=0.7, label=s['name'])

    ax_spec.set_title(f"Error Spectrum Snapshot")
    ax_spec.set_xlabel("Eigenmode (Low -> High)")
    ax_spec.set_ylabel("Error Coeff Magnitude")
    ax_spec.legend(fontsize='small', loc='lower left')
    ax_spec.grid(True, alpha=0.3)
    
    ax_att.set_title(f"Attenuation Factor vs Zero Init")
    ax_att.set_xlabel("Eigenmode (Low -> High)")
    ax_att.set_ylabel("Ratio (Current / Initial_Zero)")
    ax_att.axhline(1.0, color='red', linestyle='--', alpha=0.3)
    ax_att.axhline(0.1, color='gray', linestyle=':', alpha=0.3)
    ax_att.grid(True, alpha=0.3)

    # 4. A-Norm Initial Improvement
    names = [s['name'] for s in results_map]
    anorms = [s['results']['anorm_init'] for s in results_map]
    
    # Normalize by Standard CG (Zero)
    base_anorm = anorms[0]
    bars = ax_anorm.bar(names, anorms, color=[s['color'] for s in results_map], alpha=0.7)
    ax_anorm.set_title("Initial Error Energy (A-Norm)")
    ax_anorm.set_ylabel("||e_0||_A")
    ax_anorm.bar_label(bars, fmt='%.1e', padding=3)
    ax_anorm.tick_params(axis='x', rotation=45, labelsize=8)

    # 5. Time
    times = [s['results']['duration'] * 1000 for s in results_map] # ms
    bars_t = ax_time.bar(names, times, color=[s['color'] for s in results_map], alpha=0.7)
    ax_time.set_title("Total Solver Time (incl. Overhead)")
    ax_time.set_ylabel("Time (ms)")
    ax_time.bar_label(bars_t, fmt='%.1f', padding=3)
    ax_time.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Save
    plt.tight_layout()
    filename = export_dir / f"comprehensive_{filename_suffix}.png"
    plt.savefig(filename, dpi=150)
    print(f"   Saved plot to {filename}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Solver Benchmark")
    parser.add_argument("--N", type=int, default=256, help="Matrix size")
    args = parser.parse_args()
    
    # Paths
    project_root = Path(__file__).parent.parent
    model_path = project_root / "models" / "highmodel" / "cg_initializer.onnx"
    export_dir = Path(__file__).parent / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    run_benchmark(args.N, model_path, export_dir)

if __name__ == "__main__":
    main()
