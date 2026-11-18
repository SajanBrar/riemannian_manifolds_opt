import numpy as np
import matplotlib.pyplot as plt
from assignment1_helper_forclass import solve_with_metric, METRICS
from main import generate_data, random_init, LABELS

SEED = 0
m, n, p = 100, 500, 10
R_VALUES = [2, 5, 10, 20, 50]
N_RUNS = 5
MAX_ITERATIONS = 200
NOISE_STD = 1.0

def run_rank_experiment():
    rng = np.random.default_rng(SEED)
    costs = {
        metric: np.zeros((N_RUNS, len(R_VALUES)))
        for metric in METRICS
    }
    for trial in range(N_RUNS):
        Y, Y_true = generate_data(m, n, p, rng, noise_std=NOISE_STD)
        for j, r in enumerate(R_VALUES):
            U0, V0 = random_init(m, n, r, rng)
            for metric in METRICS:
                xopt, fval, infos, result = solve_with_metric(Y, U0, V0, metric=metric, options={"max_iterations": MAX_ITERATIONS}, run_gradient_check=False)
                costs[metric][trial, j] = fval

    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        mean_cost = np.mean(costs[metric], axis=0)
        std_cost = np.std(costs[metric], axis=0, ddof=1)
        ax.errorbar(R_VALUES, mean_cost, yerr=std_cost, marker="o", linewidth=2, capsize=4, label=LABELS[metric],)
    ax.set_xlabel("Reconstruction rank r")
    ax.set_ylabel(r"Mean final objective  $\frac{1}{2}\|UV^\top-Y\|_F^2$")
    ax.set_title(f"Final reconstruction error vs rank "f"(m={m}, n={n}, p={p}, runs={N_RUNS}, seed={SEED})")
    ax.set_xticks(R_VALUES)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()
    for metric in METRICS:
        mean_cost = np.mean(costs[metric], axis=0)
        std_cost = np.std(costs[metric], axis=0, ddof=1)
        print(f"\n{LABELS[metric]}")
        for r, mean, std in zip(R_VALUES, mean_cost, std_cost):
            print(f"r={r:2d}: mean={mean:.6e}, std={std:.6e}")
if __name__ == "__main__":
    run_rank_experiment()