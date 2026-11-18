"""
assignment1_main_template.py
============================

Skeleton of the main script for Coding Assignment 1.  Fill in the TODOs and
submit your version as  main.py  (and a separate  main_bonus.py  for the bonus
question).  Running

    python main.py

must first generate a small synthetic dataset (say m = 100, n = 500, r = p = 5)
and then produce (at least) one figure comparing the reconstruction error of the
three approaches across iterations, for the same Y and the same initial point.
Feel free to restructure this file; only `assignment1_helper_forclass.py` has a
fixed interface.
"""

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
    # Y = Y_true
    Y = Y_true + noise_std * rng.standard_normal((m, n))
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
    m, n, r, p = 100, 1000, 10, 10      # keep the default run small; change these freely for your experiments
    Y, Y_true = generate_data(m, n, p, rng)
    U0, V0 = random_init(m, n, r, rng)

    runs = {}
    for metric in METRICS:
        xopt, fval, infos, result = solve_with_metric(
            Y, U0, V0, metric=metric, options={"max_iterations": 200},
            run_gradient_check=True)        # set to False once you are convinced your gradient is right
        # (in the gradient-check plot, ignore a slight zigzag near the bottom-left, around 1e-8: numerical noise)
        runs[metric] = infos
        print(f"{metric:16s} final_cost={fval:.6e}, iterations={len(infos['iter'])}, time={infos['time'][-1]:.3f}s, stopping={result.stopping_criterion}")

    plt.figure(figsize=(7, 4.5))
    for metric, infos in runs.items():
        plt.semilogy(infos["iter"], infos["cost"], label=LABELS[metric])
    plt.xlabel("iteration"); plt.ylabel("reconstruction error  0.5 ||U V' - Y||^2")
    plt.title(f"m={m}, n={n}, r={r}, p={p}, seed={SEED}")
    plt.grid(True, which="both", alpha=.3); plt.legend(); plt.tight_layout()
    plt.show()

    C_VALUES = [0.01, 1, 100]
    scaling_results = {}
    for metric in METRICS:
        for c in C_VALUES:
            U_init = c * U0
            V_init = V0 / c
            xopt, fval, infos, result = solve_with_metric(Y, U_init, V_init, metric=metric, options={"max_iterations": 200}, run_gradient_check=False)
            scaling_results[(metric, c)] = infos
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, metric in zip(
        axes, ("canonical", "preconditioned", "right_invariant")):
        for c in C_VALUES:
            infos = scaling_results[(metric, c)]
            ax.semilogy(infos["iter"], infos["cost"], linewidth=2, label=f"c={c:g}")
        ax.set_title(LABELS[metric])
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Objective value")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"Effect of rescaling ($cU_0, V_0/c$), (m={m}, n={n}, r={r}, p={p}, seed={SEED})")
    fig.tight_layout()
    plt.show()