"""
Unified simulation script for self-attention dynamics on S^{d-1}.

PART A (symmetric baseline, Kuehn & Yoon arXiv:2604.26085, Eq. 2.2):
    xdot_i = (I - x_i x_i^T) sum_j alpha_ij(X) V x_j,
    alpha_ij(X) = softmax_j( beta <x_i, V x_j> ),   V = V^T.

PART B (asymmetric extension, this project):
    Score matrix M and value matrix V are decoupled:
        xdot_i = (I - x_i x_i^T) sum_j softmax_j(beta <x_i, M x_j>) V x_j
    with M(eps) = V + eps * A,  A = -A^T,  V = V^T fixed.
    At eps = 0, M = V and Part B reduces EXACTLY to Part A (checked below).

Run stages in order; each stage prints a self-check before moving to plots.
"""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=8, suppress=True)

SEED = 0
rng = np.random.default_rng(SEED)


# ======================================================================
#  Core utilities (validated, from the latest symmetric baseline script)
# ======================================================================

def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Normalize every row of X; reject zero or non-finite rows."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if not np.all(np.isfinite(X)) or not np.all(np.isfinite(norms)):
        raise ValueError("X must contain only finite values")
    if np.any(norms <= 0.0):
        raise ValueError("rows must be nonzero before normalization")
    return X / norms


def validate_symmetric_matrix(V: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Validate and return a finite real symmetric square matrix."""
    V = np.asarray(V, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1]:
        raise ValueError("V must be a square matrix")
    if not np.all(np.isfinite(V)):
        raise ValueError("V must contain only finite values")
    if not np.allclose(V, V.T, rtol=0.0, atol=atol):
        raise ValueError("this routine requires V = V.T")
    return V


def validate_square_matrix(M: np.ndarray) -> np.ndarray:
    """Validate and return a finite real square matrix (no symmetry required)."""
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("M must be a square matrix")
    if not np.all(np.isfinite(M)):
        raise ValueError("M must contain only finite values")
    return M


# ======================================================================
#  PART A: symmetric field (M = V = V^T) -- unchanged, validated baseline
# ======================================================================

def symmetric_field(X: np.ndarray, V: np.ndarray, beta: float) -> np.ndarray:
    """Symmetric self-attention vector field: F_i = (I - x_i x_i^T) sum_j alpha_ij V x_j."""
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    V = validate_symmetric_matrix(V)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != V.shape[0]:
        raise ValueError("X must have shape (n, d), with d = V.shape[0]")

    U = normalize_rows(X)
    scores = beta * (U @ V @ U.T)
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    attended = weights @ (U @ V)
    return attended - np.sum(U * attended, axis=1, keepdims=True) * U


# ======================================================================
#  PART B: general field with decoupled score matrix M and value matrix V
#  Reduces EXACTLY to symmetric_field when M = V.
# ======================================================================

def general_field(X: np.ndarray, M: np.ndarray, V: np.ndarray, beta: float) -> np.ndarray:
    """General self-attention vector field with decoupled score (M) / value (V) matrices."""
    if not np.isfinite(beta):
        raise ValueError("beta must be finite")
    M = validate_square_matrix(M)
    V = validate_square_matrix(V)
    if M.shape != V.shape:
        raise ValueError("M and V must have the same shape")
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != V.shape[0]:
        raise ValueError("X must have shape (n, d), with d = V.shape[0]")

    U = normalize_rows(X)
    scores = beta * (U @ M @ U.T)             # S_ij = beta <x_i, M x_j>
    scores -= np.max(scores, axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=1, keepdims=True)
    attended = weights @ (U @ V)               # sum_j K_ij V x_j
    return attended - np.sum(U * attended, axis=1, keepdims=True) * U


def simulate_rk4(
    X0: np.ndarray,
    M: np.ndarray,
    V: np.ndarray,
    *,
    beta: float = 1.0,
    T: float = 10.0,
    dt: float = 1e-2,
    every: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate xdot_i = general_field(...) with classical RK4 + row-renormalization."""
    if not np.isfinite(T) or not np.isfinite(dt):
        raise ValueError("T and dt must be finite")
    if T < 0.0 or dt <= 0.0 or every < 1:
        raise ValueError("require T >= 0, dt > 0, and every >= 1")

    steps = int(round(T / dt))
    tolerance = 64.0 * np.finfo(float).eps * max(1.0, abs(T))
    if not np.isclose(steps * dt, T, rtol=0.0, atol=tolerance):
        raise ValueError("T must be an integer multiple of dt to working precision")

    M = validate_square_matrix(M)
    V = validate_square_matrix(V)
    X = normalize_rows(X0).copy()
    if X.shape[1] != V.shape[0]:
        raise ValueError("X0 and V have incompatible dimensions")

    ts = [0.0]
    trajectory = [X.copy()]
    for k in range(1, steps + 1):
        k1 = general_field(X, M, V, beta)
        k2 = general_field(X + 0.5 * dt * k1, M, V, beta)
        k3 = general_field(X + 0.5 * dt * k2, M, V, beta)
        k4 = general_field(X + dt * k3, M, V, beta)
        X = X + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        X = normalize_rows(X)
        if k % every == 0 or k == steps:
            ts.append(k * dt)
            trajectory.append(X.copy())

    return np.asarray(ts), np.asarray(trajectory)


# ======================================================================
#  Spectral analysis of the (symmetric) baseline V
# ======================================================================

@dataclass(frozen=True)
class SpectralData:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    lambda_max: float
    lambda_min: float
    e_top: np.ndarray
    positive_dominant: bool
    negative_definite: bool
    spectral_gap_top: float


def spectral_data(V: np.ndarray, *, tol: float = 1e-12) -> SpectralData:
    """Eigenbasis and basic spectral classification of symmetric V."""
    V = validate_symmetric_matrix(V, atol=tol)
    evals, evecs = np.linalg.eigh(V)
    d = V.shape[0]
    lambda_max = float(evals[-1])
    lambda_min = float(evals[0])
    e_top = evecs[:, -1]
    spectral_gap_top = float(evals[-1] - evals[-2]) if d >= 2 else np.inf
    positive_dominant = lambda_max > max(0.0, -lambda_min) + tol
    negative_definite = lambda_max < -tol

    identity = np.eye(d)
    if not np.allclose(evecs.T @ evecs, identity, rtol=0.0, atol=100 * tol):
        raise RuntimeError("eigenvectors are not numerically orthonormal")
    if not np.allclose(V @ e_top, lambda_max * e_top, rtol=0.0,
                        atol=100 * tol * max(1.0, np.linalg.norm(V))):
        raise RuntimeError("top eigenpair residual is unexpectedly large")

    return SpectralData(evals, evecs, lambda_max, lambda_min, e_top,
                         positive_dominant, negative_definite, spectral_gap_top)


# ======================================================================
#  Diagnostics
# ======================================================================

def pairwise_rho_min(trajectory: np.ndarray) -> np.ndarray:
    """rho_min(t) = min_{i != j} <x_i(t), x_j(t)>."""
    trajectory = np.asarray(trajectory, dtype=float)
    nt, n, _ = trajectory.shape
    if n < 2:
        return np.ones(nt)
    gram = np.einsum("tid,tjd->tij", trajectory, trajectory)
    diag = np.arange(n)
    gram[:, diag, diag] = np.inf
    return np.min(gram, axis=(1, 2))


def order_parameter(trajectory: np.ndarray) -> np.ndarray:
    """||mean_i x_i(t)|| in [0, 1]; 1 = full consensus."""
    return np.linalg.norm(np.mean(trajectory, axis=1), axis=1)


def consensus_error(trajectory: np.ndarray) -> np.ndarray:
    """max_i ||x_i - normalized(mean_j x_j)||."""
    mean = np.mean(trajectory, axis=1)
    mean_norm = np.linalg.norm(mean, axis=1)
    errors = np.full(trajectory.shape[0], np.nan)
    nonzero = mean_norm > 0.0
    directions = np.zeros_like(mean)
    directions[nonzero] = mean[nonzero] / mean_norm[nonzero, None]
    diff = trajectory[nonzero] - directions[nonzero, None, :]
    errors[nonzero] = np.max(np.linalg.norm(diff, axis=2), axis=1)
    return errors


def top_eigendirection_alignment(trajectory, e_top):
    """abs_alignment(t) = min_i |<x_i(t), e_top>|; signed_alignment(t) = min_i <x_i(t), e_top>."""
    dots = trajectory @ e_top
    return np.min(np.abs(dots), axis=1), np.min(dots, axis=1)


def angular_velocity_2d(trajectory: np.ndarray) -> np.ndarray:
    """d=2 only: per-step mean angular velocity of each row's phase angle."""
    theta = np.arctan2(trajectory[:, :, 1], trajectory[:, :, 0])
    dtheta = np.diff(np.unwrap(theta, axis=0), axis=0)
    return dtheta.mean(axis=1)


def sample_cap_initial_condition(*, n, d, e_top, delta, rng, max_attempts=100_000):
    """Sample points in the cap {x: x^T e_top >= delta}."""
    if not (0.0 < delta < 1.0):
        raise ValueError("require 0 < delta < 1")
    for _ in range(max_attempts):
        X0 = normalize_rows(rng.normal(size=(n, d)))
        X0 *= np.where(X0 @ e_top >= 0.0, 1.0, -1.0)[:, None]
        if np.all(X0 @ e_top >= delta):
            return X0
    raise RuntimeError("could not sample requested cap; decrease delta")


def sample_dispersed_initial_condition(*, n, d, rng):
    """Fully random points on S^{d-1}: no cap / hemisphere restriction."""
    return normalize_rows(rng.normal(size=(n, d)))


def consensus_manifold_residual(u, M, V, *, beta, n):
    """Check that all agents have identical velocity on the consensus manifold."""
    u = normalize_rows(np.asarray(u, dtype=float)[None, :])[0]
    F = general_field(np.repeat(u[None, :], n, axis=0), M, V, beta)
    return float(np.max(np.linalg.norm(F - F[0], axis=1)))


def rk4_refinement_check(X0, M, V, *, beta, T, dt):
    """Compare final states for dt, dt/2, dt/4 (numerical consistency, not a global bound)."""
    _, coarse = simulate_rk4(X0, M, V, beta=beta, T=T, dt=dt, every=1)
    _, medium = simulate_rk4(X0, M, V, beta=beta, T=T, dt=dt / 2.0, every=1)
    _, fine = simulate_rk4(X0, M, V, beta=beta, T=T, dt=dt / 4.0, every=1)
    Xc, Xm, Xf = coarse[-1], medium[-1], fine[-1]
    err_cm = float(np.max(np.linalg.norm(Xc - Xm, axis=1)))
    err_mf = float(np.max(np.linalg.norm(Xm - Xf, axis=1)))
    ratio = err_cm / err_mf if err_mf > 0.0 else np.inf
    return err_cm, err_mf, ratio


# ======================================================================
#  STAGE 0: symmetric baseline V and its spectrum
# ======================================================================

V = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=float)
V = validate_symmetric_matrix(V)
spec = spectral_data(V)

print("=" * 70)
print("STAGE 0: symmetric baseline")
print("=" * 70)
print("V =\n", V)
print("eigenvalues (ascending):", spec.eigenvalues)
print("lambda_max:", spec.lambda_max, " lambda_min:", spec.lambda_min)
print("top spectral gap:", spec.spectral_gap_top)
print("positive-dominant:", spec.positive_dominant, " negative-definite:", spec.negative_definite)
print("top eigenvector e_top:", spec.e_top)

# ======================================================================
#  STAGE 1: symmetric dynamics validation (Part A, M = V), eps = 0 case
#  This reproduces the validated symmetric-baseline script exactly.
# ======================================================================

beta0, n, trials, delta, T, dt, every = 1.0, 20, 20, 0.3, 15.0, 0.01, 10

residual = consensus_manifold_residual(spec.e_top, V, V, beta=beta0, n=n)
top_eq_vel = np.linalg.norm(
    general_field(np.repeat(spec.e_top[None, :], n, axis=0), V, V, beta0)[0]
)

rhos, orders, cons_errs, abs_aligns = [], [], [], []
ts_ref = None
for _ in range(trials):
    X0 = sample_cap_initial_condition(n=n, d=V.shape[0], e_top=spec.e_top, delta=delta, rng=rng)
    ts, tr = simulate_rk4(X0, V, V, beta=beta0, T=T, dt=dt, every=every)   # M = V here
    ts_ref = ts
    rhos.append(pairwise_rho_min(tr))
    orders.append(order_parameter(tr))
    cons_errs.append(consensus_error(tr))
    a, _ = top_eigendirection_alignment(tr, spec.e_top)
    abs_aligns.append(a)
rhos, orders, cons_errs, abs_aligns = map(np.asarray, (rhos, orders, cons_errs, abs_aligns))

print("\n" + "=" * 70)
print("STAGE 1: symmetric dynamics (M = V, eps = 0) -- validation")
print("=" * 70)
print(f"consensus-manifold invariance residual: {residual:.3e}")
print(f"velocity at top-eigenvector consensus:  {top_eq_vel:.3e}")
print(f"final rho_min: mean={rhos[:, -1].mean():.8f}, min={rhos[:, -1].min():.8f}")
print(f"final order param: mean={orders[:, -1].mean():.8f}, min={orders[:, -1].min():.8f}")
print(f"final consensus error: mean={cons_errs[:, -1].mean():.3e}")
print(f"final |alignment| with e_top: mean={abs_aligns[:, -1].mean():.8f}")

X0_refine = sample_cap_initial_condition(n=n, d=V.shape[0], e_top=spec.e_top, delta=delta, rng=rng)
e1_, e2_, ratio_ = rk4_refinement_check(X0_refine, V, V, beta=beta0, T=T, dt=dt)
print(f"RK4 refinement ratio (want ~16 for 4th order): {ratio_:.3f}")

# ======================================================================
#  STAGE 2: asymmetric extension -- reduction check (M(eps=0) = V)
# ======================================================================

A = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=float)     # antisymmetric generator, d = 2
if not np.allclose(A, -A.T):
    raise ValueError("A must be antisymmetric")


def run_condition(*, eps, beta, delta, T, dispersed, n=n, trials=trials, dt=dt, every=every):
    """One (eps, beta, delta) experimental condition; returns ts and diagnostics."""
    M = V + eps * A
    rho_l, ord_l, om_l = [], [], []
    ts_local = None
    for _ in range(trials):
        if dispersed:
            X0 = sample_dispersed_initial_condition(n=n, d=V.shape[0], rng=rng)
        else:
            X0 = sample_cap_initial_condition(n=n, d=V.shape[0], e_top=spec.e_top, delta=delta, rng=rng)
        ts_l, tr_l = simulate_rk4(X0, M, V, beta=beta, T=T, dt=dt, every=every)
        ts_local = ts_l
        rho_l.append(pairwise_rho_min(tr_l))
        ord_l.append(order_parameter(tr_l))
        om_l.append(angular_velocity_2d(tr_l))
    return ts_local, np.asarray(rho_l), np.asarray(ord_l), np.asarray(om_l)


print("\n" + "=" * 70)
print("STAGE 2: asymmetric reduction check (eps = 0 must match Stage 1)")
print("=" * 70)
ts_e0, rho_e0, ord_e0, om_e0 = run_condition(eps=0.0, beta=beta0, delta=delta, T=T, dispersed=False)
print(f"[eps=0] final rho_min min={rho_e0[:, -1].min():.6f}, final order mean={ord_e0[:, -1].mean():.6f}")
if abs(rho_e0[:, -1].min() - rhos[:, -1].min()) > 1e-3:
    print("WARNING: eps=0 reduction does not match Stage 1 baseline closely.")
else:
    print("OK: eps=0 asymmetric run matches the symmetric baseline.")

# ======================================================================
#  STAGE 3: exploring the transition -- larger eps, larger beta,
#  dispersed initial data (no cone), near-degenerate spectrum
# ======================================================================

print("\n" + "=" * 70)
print("STAGE 3: sweeping conditions to find a real asymmetric transition")
print("=" * 70)

conditions = [
    dict(label="baseline cone, eps small",      eps=0.3, beta=1.0,  delta=0.3, T=15.0, dispersed=False),
    dict(label="large eps, cone",                eps=3.0, beta=1.0,  delta=0.3, T=15.0, dispersed=False),
    dict(label="large eps, dispersed IC",         eps=3.0, beta=1.0,  delta=0.3, T=15.0, dispersed=True),
    dict(label="large eps, sharp beta, dispersed",eps=3.0, beta=8.0,  delta=0.3, T=15.0, dispersed=True),
    dict(label="very large eps=5, dispersed",     eps=5.0, beta=1.0,  delta=0.3, T=15.0, dispersed=True),
]

stage3_results = {}
for c in conditions:
    ts_c, rho_c, ord_c, om_c = run_condition(
        eps=c["eps"], beta=c["beta"], delta=c["delta"], T=c["T"], dispersed=c["dispersed"]
    )
    stage3_results[c["label"]] = (ts_c, rho_c, ord_c, om_c)
    late_om = om_c[:, -20:].mean()
    print(f"[{c['label']:32s}] final order: mean={ord_c[:, -1].mean():.4f} "
          f"min={ord_c[:, -1].min():.4f} | final rho_min mean={rho_c[:, -1].mean():.4f} "
          f"| late mean angular vel={late_om:+.5f}")

# ---------- near-degenerate spectrum (smaller spectral gap) ----------
V_close = np.array([[1.0, 0.95], [0.95, 1.0]], dtype=float)
V_close = validate_symmetric_matrix(V_close)
spec_close = spectral_data(V_close)
print(f"\nnear-degenerate V_close eigenvalues: {spec_close.eigenvalues} "
      f"(gap={spec_close.spectral_gap_top:.4f}, vs baseline gap={spec.spectral_gap_top:.4f})")

M_close = V_close + 3.0 * A
rho_cl, ord_cl, om_cl = [], [], []
ts_cl = None
for _ in range(trials):
    X0 = sample_dispersed_initial_condition(n=n, d=V_close.shape[0], rng=rng)
    ts_cl, tr_cl = simulate_rk4(X0, M_close, V_close, beta=1.0, T=15.0, dt=dt, every=every)
    rho_cl.append(pairwise_rho_min(tr_cl))
    ord_cl.append(order_parameter(tr_cl))
    om_cl.append(angular_velocity_2d(tr_cl))
rho_cl, ord_cl, om_cl = map(np.asarray, (rho_cl, ord_cl, om_cl))
print(f"[near-degenerate, eps=3, dispersed] final order: mean={ord_cl[:, -1].mean():.4f} "
      f"| final rho_min mean={rho_cl[:, -1].mean():.4f} "
      f"| late mean angular vel={om_cl[:, -20:].mean():+.5f}")

# ======================================================================
#  Figures
# ======================================================================

fig1, ax1 = plt.subplots(1, 3, figsize=(14, 4.2))
ax1[0].plot(ts_ref, orders.mean(axis=0), label=r"mean $|\bar{x}|$")
ax1[0].plot(ts_ref, rhos.mean(axis=0), label=r"mean $\rho_{\min}$")
ax1[0].set(title="Stage 1: symmetric baseline consensus", xlabel="t", ylabel="value", ylim=(-0.05, 1.05))
ax1[0].legend()
ax1[1].semilogy(ts_ref, np.maximum(cons_errs.mean(axis=0), np.finfo(float).tiny))
ax1[1].set(title="Stage 1: consensus error", xlabel="t", ylabel=r"mean $\max_i\|x_i-\hat{\bar x}\|$")
ax1[2].plot(ts_ref, abs_aligns.mean(axis=0))
ax1[2].set(title="Stage 1: alignment with top eigenvector", xlabel="t", ylabel="alignment", ylim=(-0.05, 1.05))
fig1.tight_layout()
fig1.savefig("stage1_symmetric.png", dpi=150)
plt.close(fig1)

fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.5))
for label, (ts_c, rho_c, ord_c, om_c) in stage3_results.items():
    ax2[0].plot(ts_c, ord_c.mean(axis=0), label=label)
ax2[0].set(title="Stage 3: order parameter across conditions", xlabel="t", ylabel=r"mean $|\bar x(t)|$")
ax2[0].legend(fontsize=7)
for label, (ts_c, rho_c, ord_c, om_c) in stage3_results.items():
    ax2[1].plot(ts_c[1:], om_c.mean(axis=0), label=label)
ax2[1].axhline(0, color="k", lw=0.5)
ax2[1].set(title="Stage 3: mean angular velocity across conditions", xlabel="t", ylabel="rad/step")
ax2[1].legend(fontsize=7)
fig2.tight_layout()
fig2.savefig("stage3_asymmetric_sweep.png", dpi=150)
plt.close(fig2)

print("\nAll stages complete. Figures saved: stage1_symmetric.png, stage3_asymmetric_sweep.png")
