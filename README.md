# Riemannian Optimization for Low-Rank Matrix Factorization

This project studies how the choice of Riemannian geometry affects the numerical optimization of low-rank matrix factorization. It implements three Riemannian metrics on the same product space and compares the resulting steepest-descent methods.

The implementation uses **Python, NumPy, Matplotlib, and Pymanopt**.

## 1. Problem formulation

Let $Y\in\mathbb{R}^{m\times n}$ be a matrix to be approximated by a rank-$r$ factorization. We represent the approximation as

$$
X=UV^\top,\qquad U\in\mathbb{R}^{m\times r},\quad V\in\mathbb{R}^{n\times r}.
$$

For fully observed data, the optimization problem is

$$
\boxed{\min_{U\in\mathbb{R}^{m\times r},\,V\in\mathbb{R}^{n\times r}}
 f(U,V)=\frac{1}{2}\lVert UV^\top-Y\rVert_F^2.}
$$

The variables lie on the product space

$$
\mathcal M=\mathbb{R}^{m\times r}\times\mathbb{R}^{n\times r}.
$$

A point on this space is a pair $(U,V)$, and a tangent vector is a pair $(\xi_U,\xi_V)$ of matrices with the same respective dimensions. Since the factors are unconstrained Euclidean matrices, the tangent space at every point is the full product space. The factorization is non-unique: for any invertible $M\in\mathbb{R}^{r\times r}$,

$$
UV^\top=(UM^{-1})(VM^\top)^\top.
$$

Thus, different pairs of factors can represent exactly the same matrix. The experiments investigate how different geometries handle this freedom and how their optimization trajectories compare.

Define the residual $E=UV^\top-Y$. The Euclidean gradients, common to all three methods, are
$$
\nabla_U f=EV,\qquad \nabla_V f=E^\top U.
$$

The Riemannian gradient depends on the metric used to define the inner product on each tangent space.

## 2. The three Riemannian metrics

All three methods optimize the same objective, use the same initial factors for a given comparison, and use Pymanopt's Riemannian steepest-descent optimizer with backtracking line search. They differ in the inner product and in the conversion from the Euclidean gradient to the Riemannian gradient.

Let $\xi=(\xi_U,\xi_V)$ and $\eta=(\eta_U,\eta_V)$ be tangent vectors at $(U,V)$.

### 2.1 Canonical metric

The canonical metric is the ordinary Euclidean inner product on the product space:

$$
\langle\xi,\eta\rangle_{\mathrm{can}}
={tr}(\xi_U^\top\eta_U)
+{tr}(\xi_V^\top\eta_V).
$$

Its Riemannian gradient is therefore the Euclidean gradient:

$$
{grad}_{\mathrm{can}} f
=(EV, E^\top U).
$$

This gives the baseline geometry. It is simple, but it does not account for the non-uniqueness and rescaling freedom of the factors.

### 2.2 Preconditioned metric

The preconditioned metric weights each factor's tangent directions using the Gram matrix of the other factor:

\begin{align*}
\langle\xi,\eta\rangle_{\mathrm{pre}}
={}&{tr}\!\left((V^\top V)\xi_U^\top\eta_U\right)\\
&+{tr}\!\left((U^\top U)\xi_V^\top\eta_V\right).
\end{align*}

The corresponding Riemannian gradient is

$$
{grad}_{\mathrm{pre}} f
=\left(EV(V^\top V)^{-1},\;
E^\top U(U^\top U)^{-1}\right).
$$

This geometry is related to preconditioning and to the geometry proposed by Mishra, Apuroop, and Sepulchre. It compensates for the scale and conditioning of the two factors through their Gram matrices.

### 2.3 Right-invariant metric

The right-invariant metric is

\begin{align*}
\langle\xi,\eta\rangle_{\mathrm{ri}}
={}&{tr}\!\left((U^\top U)^{-1}\xi_U^\top\eta_U\right)\\
&+{tr}\!\left((V^\top V)^{-1}\xi_V^\top\eta_V\right).
\end{align*}

Its Riemannian gradient is

$$
{grad}_{\mathrm{ri}} f
=\left(EV(U^\top U),\;
E^\top U(V^\top V)\right).
$$

Unlike the preconditioned metric, this metric is motivated by the geometry of the factor space rather than by the objective's cross-factor structure.

### 2.4 Metric comparison

| Metric | Inner-product weighting | Riemannian gradient |
|---|---|---|
| Canonical | Identity weighting for both factors | \((EV,\ E^\top U)\) |
| Preconditioned | $V^\top V$ for $U$; $U^\top U$ for $V$ | $(EV(V^\top V)^{-1},\ E^\top U(U^\top U)^{-1})$ |
| Right-invariant | $(U^\top U)^{-1}$ for $U$; $(V^\top V)^{-1}$ for $V$ | $(EV(U^\top U),\ E^\top U(V^\top V))$ |


## 3. Experimental cases

### 3.1 Synthetic low-rank data: exact and noisy observations

The synthetic ground-truth matrix is generated from independent standard Gaussian factors:

$$
A\in\mathbb{R}^{m\times p},\qquad
B\in\mathbb{R}^{n\times p},\qquad
A_{ij},B_{ij}\sim\mathcal N(0,1),
$$

with

$$
Y_{\mathrm{true}}=AB^\top.
$$

The observed matrix can be either noise-free,

$$
Y=Y_{\mathrm{true}},
$$

or noisy,

$$
Y=Y_{\mathrm{true}}+\varepsilon,
\qquad \varepsilon_{ij}\sim\mathcal N(0,\sigma^2).
$$

### 3.2 Gradient validation

For each metric, the implementation checks the first-order approximation

$$
f(R_x(h\eta))\approx f(x)+h\langle\operatorname{grad}f(x),\eta\rangle_x,
$$

where $R_x$ is the retraction and $h$ is a small step. For a correct Riemannian gradient, the model error should scale quadratically with the step size, producing a slope close to 2 in a log-log plot over the range where truncation error dominates floating-point round-off. The report includes this check for all three metrics.

The check must be interpreted separately for each metric because both the inner product and the Riemannian gradient conversion change with the metric. Very small step sizes can be dominated by floating-point round-off.

### 3.3 Equivalent factor rescaling

Because the product \(UV^\top\) is unchanged by

\[
(U,V)\longmapsto(UM^{-1},VM^\top),
\]

the two factor pairs represent the same matrix. A scalar instance is

\[
(U,V)\longmapsto(cU,V/c).
\]

The report tests \(c\in\{0.01,1,100\}\) and compares the objective trajectories from these equivalent initial representations.

- **Canonical metric:** it is not invariant under this reparameterization. Consequently, equivalent initial factorizations can produce different optimization trajectories and convergence behavior.
- **Preconditioned metric:** it is invariant under the coupled transformation above. Its trajectories are therefore expected to be unchanged in exact arithmetic when the starting factors are transformed together.
- **Right-invariant metric:** it is also invariant under the coupled transformation. The report additionally establishes invariance under independent invertible changes of basis in the two factors, a broader invariance of the metric itself.

In the reported rescaling experiment, the canonical curves separate, while the preconditioned and right-invariant curves are approximately invariant. A small deviation between right-invariant curves is visible around iteration 60; the report notes that this may be due to numerical effects.

### 3.4 Varying the reconstruction rank

The rank experiment fixes the generative rank at \(p=10\) and tests reconstruction ranks

$$
r\in\{2,5,10,20,50\}.
$$

For each rank, the code performs five independent runs and plots the mean final objective with standard-deviation error bars for each metric. The reported curves decrease as the fitted rank grows. They are similar through the correctly specified rank $r=p=10$, then diverge for larger ranks. Beyond the signal rank, additional factor dimensions can fit components of the noise, and the three geometries can behave differently during optimization.

### 3.5 Partially observed matrices (matrix completion)

The matrix-completion experiment assumes that only entries indexed by an observation set \(\Omega\) are available. A binary mask is defined by

$$
\Omega_{ij}=\begin{cases}
1,&\text{if entry }(i,j)\text{ is observed},\\
0,&\text{otherwise}.
\end{cases}
$$

The objective becomes

$$
\boxed{
\min_{U,V}\frac12\left\lVert
\Omega\odot(UV^\top-Y)
\right\rVert_F^2,
}
$$

where $\odot$ denotes elementwise multiplication. Only observed entries contribute to the optimization objective. The three Riemannian metrics and their gradient mappings are otherwise unchanged; the residual used by the objective and Euclidean gradient is masked before the gradient is formed.

The synthetic mask contains

$$
k= s(m+n-r)r
$$

observed entries, where \(s\) is an oversampling parameter. The generator rejects masks that leave any row or column entirely unobserved. The uploaded `main_bonus.py` uses \(s=6\) and generates a noise-free synthetic target. It plots both (i) the objective on observed entries and (ii) the relative reconstruction error against the complete synthetic ground-truth matrix:

$$
\frac{\lVert UV^\top-Y_{\mathrm{true}}\rVert_F}
{\lVert Y_{\mathrm{true}}\rVert_F}.$$

The second quantity is available here because the experiment generated the complete ground-truth matrix. In a real matrix-completion application, unobserved ground-truth entries are not available, so this full-matrix error cannot normally be computed directly.


### Script descriptions

| File | Purpose | Output |
|---|---|---|
| `assignment1_helper_forclass.py` | Defines the product manifold, the three inner products, metric-specific Riemannian gradients, gradient checking, and the shared Pymanopt solver. | Solver results, iteration logs, and optional gradient-check plots. |
| `main.py` | Generates a fully observed synthetic matrix, runs all three metrics from a common initialization, and performs the coupled and same-direction factor-rescaling comparisons. | Printed optimization diagnostics and interactive Matplotlib figures. |
| `rank_exp.py` | Repeats the noisy low-rank experiment for several fitted ranks and multiple random trials. | Mean final objective, standard deviations, and an error-bar plot. |
| `main_bonus.py` | Generates a partially observed synthetic matrix and runs all three metrics with an observation mask. | Printed optimization diagnostics and plots of observed objective and full-matrix relative error. |
| `requirements.txt` | Lists the Python package dependencies. | Used to install the environment. |

## 9. References

- Mishra, Apuroop, and Sepulchre, *A Riemannian Perspective on Low-Rank Optimization*, especially the preconditioned and right-invariant geometries discussed in the paper: [arXiv:1211.1550](https://arxiv.org/abs/1211.1550).
- Pymanopt documentation: [https://pymanopt.org](https://pymanopt.org).
