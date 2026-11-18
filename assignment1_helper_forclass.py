"""
assignment1_helper_forclass.py
==============================

Helper file for Coding Assignment 1.

This file sets
up the product manifold  M = R^{m x r} x R^{n x r}, endows it with one of three
Riemannian metrics (Approaches 1, 2 and 3 of the assignment), and calls
pymanopt's Riemannian steepest descent solver.  Your main script should import
and call `solve_with_metric` for each of the three metrics.

You have to fill in the parts marked  # TODO.  Everything else is provided.

A note on how pymanopt turns a manifold into an optimization problem
--------------------------------------------------------------------
A pymanopt manifold is a Python class.  The optimization algorithms only talk
to the manifold through a handful of methods, exactly the "key optimization
ingredients" listed in the assignment:

    inner_product(x, xi, eta)                  <xi, eta>_x
    norm(x, xi)                                sqrt(<xi, xi>_x)
    projection(x, u) / to_tangent_space(x, u)  projection of an ambient vector u onto T_x M
    retraction(x, xi)                          R_x(xi)
    euclidean_to_riemannian_gradient(x, G)     maps the Euclidean gradient G of the cost
                                               to the Riemannian gradient grad f(x)

The optimizer never looks at the Euclidean gradient directly: whenever it needs
grad f(x) it calls  euclidean_to_riemannian_gradient(x, euclidean_gradient(x)).
So the *metric* enters through `inner_product` / `norm` (used by the line search
and the stopping criterion) and through `euclidean_to_riemannian_gradient`.

Points on the product manifold are pairs (U, V); tangent vectors are pairs
(xi_U, xi_V).  Both are indexed like Python sequences:  U, V = point.
"""

import numpy as np
import pymanopt
from pymanopt.manifolds import Euclidean, Product
from pymanopt.manifolds.product import _ProductTangentVector
from pymanopt.optimizers import SteepestDescent

METRICS = ("canonical", "preconditioned", "right_invariant")


class ProductManifoldWithMetric(Product):
    """The product manifold R^{m x r} x R^{n x r} with a user-chosen metric.

    metric = "canonical"       (Approach 1)
        <xi, eta>_x = Tr(xi_U' eta_U) + Tr(xi_V' eta_V)
    metric = "preconditioned"  (Approach 2, Eq. (8) of Mishra, Apuroop & Sepulchre)
        <xi, eta>_x = Tr((V'V) xi_U' eta_U) + Tr((U'U) xi_V' eta_V)
    metric = "right_invariant" (Approach 3, Eq. (9) of Mishra, Apuroop & Sepulchre)
        <xi, eta>_x = Tr((U'U)^{-1} xi_U' eta_U) + Tr((V'V)^{-1} xi_V' eta_V)

    The class inherits from pymanopt's `Product` of two `Euclidean` manifolds.
    `projection`, `to_tangent_space` and `retraction` are therefore inherited
    from the Euclidean factors. Think about why that is fine for all three
    metrics (this is one of the questions in the assignment).
    """

    def __init__(self, m, n, r, metric="canonical"):
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        super().__init__([Euclidean(m, r), Euclidean(n, r)])
        self.metric = metric

    # ------------------------------------------------------------------ metric
    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        """Return the scalar <tangent_vector_a, tangent_vector_b>_point."""
        U, V = point
        xi_U, xi_V = tangent_vector_a
        eta_U, eta_V = tangent_vector_b
        if self.metric == "canonical":
            # Approach 1
            inner = np.trace(xi_U.T @ eta_U) + np.trace(xi_V.T @ eta_V)
                        
        elif self.metric == "preconditioned":
            # Approach 2
            inner = np.trace((V.T @ V) @ xi_U.T @ eta_U) + np.trace((U.T @ U) @ xi_V.T @ eta_V)

        elif self.metric == "right_invariant":
            # Approach 3
            inner = np.trace(np.linalg.solve(U.T @ U, xi_U.T @ eta_U)) + np.trace(np.linalg.solve(V.T @ V, xi_V.T @ eta_V))

        if inner is None:
            raise NotImplementedError(f"inner_product for metric {self.metric!r} is not implemented yet")
        return inner

    def norm(self, point, tangent_vector):
        # The norm is defined by the inner product, nothing to do here.
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    # ------------------------------------------------------- Riemannian gradient
    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """Map the Euclidean gradient (G_U, G_V) of the cost to the Riemannian gradient.

        Recall the defining property of the Riemannian gradient:
            D f(x)[xi] = <grad f(x), xi>_x   for all xi in T_x M.
        The left-hand side does not depend on the metric, the right-hand side does.
        """
        U, V = point
        G_U, G_V = euclidean_gradient
        if self.metric == "canonical":
            # Approach 1
            rgrad_U, rgrad_V = G_U, G_V 

        elif self.metric == "preconditioned":
            # Approach 2
            rgrad_U, rgrad_V = np.linalg.solve(V.T @ V, G_U.T).T, np.linalg.solve(U.T @ U, G_V.T).T

        elif self.metric == "right_invariant":
            # Approach 3
            rgrad_U, rgrad_V = G_U @ U.T @ U, G_V @ V.T @ V  

        if rgrad_U is None or rgrad_V is None:
            raise NotImplementedError(f"euclidean_to_riemannian_gradient for metric {self.metric!r} is not implemented yet")
        # pymanopt expects tangent vectors of a product manifold in this container
        return _ProductTangentVector([rgrad_U, rgrad_V])


def gradient_check(problem, point=None, direction=None, plot=True, seed=None):
    """Numerical check of the Riemannian gradient (pymanopt's own `check_gradient`
    does not work with recent numpy versions).

    Compares f(R_x(h d)) with the first-order model  f(x) + h <grad f(x), d>_x
    for h in [1e-8, 1].  If the Riemannian gradient (and hence your metric) is
    correct, the error behaves like h^2: the log-log plot has slope 2 over a few
    orders of magnitude.  A slope of 1 means the gradient is wrong.
    You may ignore a slight zigzag, if any, at the lower-left end of the curve
    (errors around 1e-8 and below): there the model error is of the order of the
    floating-point round-off in evaluating the cost, i.e. numerical noise.  The
    slope is fitted only on the middle part of the curve.
    Returns the fitted slope.  The check is run at a random point and along a
    random direction unless `point` / `direction` are given; `seed` makes the
    random choice reproducible.
    """
    M = problem.manifold
    if seed is not None:
        np.random.seed(seed)      # pymanopt's random_point / random_tangent_vector use numpy's global generator
    if point is None:
        point = M.random_point()
    if direction is None:
        direction = M.random_tangent_vector(point)
    f0 = problem.cost(point)
    grad = problem.riemannian_gradient(point)
    df0 = M.inner_product(point, grad, direction)
    h = np.logspace(-8, 0, 51)
    err = np.array([abs(problem.cost(M.retraction(point, hk * direction)) - (f0 + hk * df0))
                    for hk in h])
    # fit a line on the middle part of the curve, where neither round-off nor
    # higher-order terms dominate
    sel = slice(15, 36)
    slope = np.polyfit(np.log10(h[sel]), np.log10(err[sel] + 1e-300), 1)[0]
    print(f"gradient_check: fitted slope = {slope:.3f} (should be close to 2)")
    if plot:
        import matplotlib.pyplot as plt
        # below ~10 units in the last place of f(x) the error is pure round-off noise: hide those points
        floor = 10 * np.finfo(float).eps * abs(f0)
        keep = err > floor
        plt.figure()
        plt.loglog(h[keep], err[keep], label="model error")
        plt.loglog(h[keep], err[sel][0] * (h[keep] / h[sel][0]) ** 2, "k--", label="slope-2 reference")
        plt.xlabel("h"); plt.ylabel(r"$|f(R_x(hd)) - f(x) - h\langle \mathrm{grad} f(x), d\rangle_x|$")
        plt.title(f"Gradient check, metric = {M.metric!r} (slope {slope:.2f})"); plt.legend()
    return slope


def solve_with_metric(Y, U0, V0, metric="canonical", options=None, Omega=None,
                      run_gradient_check=False):
    """Riemannian steepest descent for  min_{U,V} 0.5 * || (U V' - Y)_Omega ||_F^2.

    This function must be called by your main script, once per metric.

    Inputs
        Y       : m x n matrix (given as input)
        U0      : m x r initialization of U
        V0      : n x r initialization of V
        metric  : "canonical" | "preconditioned" | "right_invariant"
        options : dict of hyper-parameters of the steepest descent algorithm,
                  e.g. {"max_iterations": 200}. Other useful keys: "min_gradient_norm",
                  "min_step_size", "verbosity" (2 prints one line per iteration).
        Omega   : None for the matrix factorisation problem (all entries of Y known), or an
                  m x n 0/1 mask for the matrix completion problem of the bonus part.
        run_gradient_check : if True, run `gradient_check` at (U0, V0) before optimizing.

    Outputs
        xopt   : dict with xopt["U"] (m x r) and xopt["V"] (n x r)
        fval   : objective value at xopt
        infos  : dict of numpy arrays with keys "iter", "cost", "gradnorm", "time",
                 one entry per iteration. Of particular
                 interest are "iter" and "cost".
        result : the raw pymanopt OptimizerResult (contains e.g. result.stopping_criterion
                 and the iterates in result.log["iterations"]["point"])
    """
    m, n = Y.shape
    r = U0.shape[1]
    mask = None if Omega is None else np.asarray(Omega, dtype=float)

    manifold = ProductManifoldWithMetric(m, n, r, metric=metric)

    # The decorator tells pymanopt which manifold the function lives on; for a
    # product manifold the components of the point are passed as separate
    # arguments.  Use plain numpy inside.
    @pymanopt.function.numpy(manifold)
    def cost(U, V):
        """Objective value 0.5 * || (U V' - Y)_Omega ||_F^2  (mask is None => all entries)."""
        error = U @ V.T - Y
        if mask is not None:
            error = mask * error 
        objective_value = 0.5 * np.sum(error**2)

        if objective_value is None:
            raise NotImplementedError("cost is not implemented yet")
        return objective_value

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(U, V):
        """Euclidean gradient of the cost, returned as the pair (dcost/dU, dcost/dV).

        This is the Euclidean (i.e. metric-independent) gradient. It is converted
        to the Riemannian gradient by `euclidean_to_riemannian_gradient` above.
        """
        error = U @ V.T - Y
        if mask is not None:
            error = mask * error
        grad_U, grad_V = error @ V, error.T @ U

        if grad_U is None or grad_V is None:
            raise NotImplementedError("euclidean_gradient is not implemented yet")
        return grad_U, grad_V

    problem = pymanopt.Problem(manifold, cost, euclidean_gradient=euclidean_gradient)

    if run_gradient_check:
        gradient_check(problem, point=(U0, V0))

    opts = dict(max_iterations=200, min_gradient_norm=1e-10, min_step_size=1e-14,
                verbosity=0, log_verbosity=2)
    if options:
        opts.update(options)
    optimizer = SteepestDescent(**opts)     # backtracking (Armijo) line search by default
    result = optimizer.run(problem, initial_point=(U0.copy(), V0.copy()))

    xopt = {"U": np.asarray(result.point[0]), "V": np.asarray(result.point[1])}
    log = result.log["iterations"]
    infos = {k: np.asarray(log[src]) for k, src in
             (("iter", "iteration"), ("cost", "cost"), ("gradnorm", "gradient_norm"), ("time", "time"))}
    return xopt, result.cost, infos, result
