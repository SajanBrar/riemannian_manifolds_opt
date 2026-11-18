import numpy as np
import matplotlib.pyplot as plt

from assignment1_helper_forclass import solve_with_metric, METRICS

SEED = 0                                    # I will ask you to run your code with a seed of my choice
rng = np.random.default_rng(SEED)           # use this generator for *all* randomness below

LABELS = {"canonical": "Approach 1 (canonical)",
          "preconditioned": "Approach 2 (preconditioned, Eq. (8))",
          "right_invariant": "Approach 3 (right-invariant, Eq. (9))"}


def generate_data(m, n, p, rng, noise_std=1.0):
    """Return Y = A B' + noise  with A (m x p), B (n x p) ~ N(0,1) entries, and the noise-free A B'."""
    A, B = rng.standard_normal((m, p)), rng.standard_normal((n, p))
    Y_true = A @ B.T
    Y = Y_true
    # Y = Y_true + noise_std * rng.standard_normal((m, n))
    return Y, Y_true


def random_init(m, n, r, rng):
    """Return a random initial point x0 = (U0, V0)."""
    U0, V0 = rng.standard_normal((m, r)), rng.standard_normal((n, r))  
    return U0, V0


def generate_omega(m, n, r, os, rng):
    """Bonus. Return an m x n 0/1 mask with k = os*(m+n-r)*r ones, at least one per row and column."""
    k = os * (m + n - r) * r
    if k < max(m, n) or k > m * n:
        raise ValueError("Invalid number of observed entries.")
    while True:
        Omega = np.zeros((m, n))
        indices = rng.choice(m * n, size=k, replace=False)
        Omega.flat[indices] = 1
        if (np.all(Omega.sum(axis=1) >= 1) and np.all(Omega.sum(axis=0) >= 1)):
            return Omega
    # Omega = None  
    # return Omega


if __name__ == "__main__":
    m, n, r, p = 100, 500, 10, 10      # keep the default run small; change these freely for your experiments
    Y, Y_true = generate_data(m, n, p, rng)
    U0, V0 = random_init(m, n, r, rng)

    Y_norm = np.linalg.norm(Y_true, "fro")
    if Y_norm == 0:
        raise ValueError("Ground-truth matrix has zero norm.")
    
    os = 6
    Omega = generate_omega(m,n,r,os,rng)

    runs = {}
    relative_errors = {}
    for metric in METRICS:
        xopt, fval, infos, result = solve_with_metric(
            Y, U0, V0, metric=metric, options={"max_iterations": 200}, Omega=Omega,
            run_gradient_check=False)        # set to False once you are convinced your gradient is right
        # (in the gradient-check plot, ignore a slight zigzag near the bottom-left, around 1e-8: numerical noise)
        runs[metric] = infos
        print(f"{metric:16s} final_cost={fval:.6e}, iterations={len(infos['iter'])}, time={infos['time'][-1]:.3f}s, stopping={result.stopping_criterion}")
        points = result.log["iterations"]["point"]
        relative_errors[metric] = np.array([np.linalg.norm(U_k @ V_k.T - Y_true, "fro") / Y_norm for U_k, V_k in points])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1 = axes[0]
    for metric, infos in runs.items():
        ax1.semilogy(infos["iter"],infos["cost"],linewidth=2,label=LABELS[metric])
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel(r"Observed objective " r"$\frac{1}{2}\|\Omega\odot(UV^\top-Y)\|_F^2$")
    ax1.set_title("Objective on observed entries")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=8)
    ax2 = axes[1]
    for metric, errors in relative_errors.items():
        iterations = runs[metric]["iter"]
        ax2.semilogy(iterations,errors,linewidth=2,label=LABELS[metric])
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel(r"Relative error " r"$\frac{\|UV^\top-Y_{\mathrm{true}}\|_F}" r"{\|Y_{\mathrm{true}}\|_F}$")
    ax2.set_title("Relative error on the full matrix")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend(fontsize=8)
    fig.suptitle(f"Matrix completion (m={m}, n={n}, r={r}, " f"os={os}, seed={SEED})", fontsize=13)
    plt.tight_layout()
    plt.show()
