"""Reproducible audit of symmetric spherical self-attention.

Paper baseline: Kuehn--Yoon, arXiv:2604.26085v1, Eq. (2.2):
    xdot_i=P^perp_{x_i} sum_j K_ij V x_j,
    K_ij=exp(beta <x_i,Vx_j>)/sum_m exp(beta <x_i,Vx_m>), V=V.T.

STAGE 2--3 implement only a *single*, paired asymmetric code audit.  They are
not an epsilon sweep and make no dynamical claim about the extension.
"""
from __future__ import annotations

from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(precision=8, suppress=True)
SEED = 0


def normalize_rows(X: np.ndarray) -> np.ndarray:
    """Return row-normalized finite vectors."""
    X = np.asarray(X, dtype=float)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    if X.ndim != 2 or not np.all(np.isfinite(X)) or np.any(norms == 0.0):
        raise ValueError("X must be a finite 2-D array with nonzero rows")
    return X / norms


def validate_square(M: np.ndarray, name: str = "matrix") -> np.ndarray:
    M = np.asarray(M, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1] or not np.all(np.isfinite(M)):
        raise ValueError(f"{name} must be a finite square matrix")
    return M


def validate_symmetric(V: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    V = validate_square(V, "V")
    if not np.allclose(V, V.T, rtol=0.0, atol=atol):
        raise ValueError("Eq. (2.2) requires V = V.T")
    return V


def softmax_rows(scores: np.ndarray) -> np.ndarray:
    """Stable row-wise softmax; each output row sums to one."""
    shifted = scores - np.max(scores, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=1, keepdims=True)


def attention_field(X: np.ndarray, M: np.ndarray, V: np.ndarray, beta: float) -> np.ndarray:
    """General extension: score <x_i,M x_j>, values V x_j.

    On the sphere, M=V=V.T is exactly paper Eq. (2.2).  The expression is
    deliberately evaluated on the supplied RK stages (not silently normalized):
    this is a smooth ambient extension used solely by classical RK4.
    """
    M, V = validate_square(M, "M"), validate_square(V, "V")
    X = np.asarray(X, dtype=float)
    if not np.isfinite(beta) or X.ndim != 2 or X.shape[1] != V.shape[0] or M.shape != V.shape:
        raise ValueError("incompatible finite X, M, V, or beta")
    weights = softmax_rows(beta * (X @ M @ X.T))
    attended = weights @ (X @ V.T)  # row j represents (V x_j)^T
    return attended - np.sum(X * attended, axis=1, keepdims=True) * X


def symmetric_field(X: np.ndarray, V: np.ndarray, beta: float) -> np.ndarray:
    """Paper Eq. (2.2), with V validated as symmetric."""
    V = validate_symmetric(V)
    return attention_field(X, V, V, beta)


def general_field(X: np.ndarray, M: np.ndarray, V: np.ndarray, beta: float) -> np.ndarray:
    """Alias retained for the requested eps=0 vector-field regression."""
    return attention_field(X, M, V, beta)


def simulate_rk4(X0: np.ndarray, M: np.ndarray, V: np.ndarray, *, beta: float, T: float,
                 dt: float, every: int = 10, save: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Classical RK4, followed by the sphere retraction X -> X/||X|| rowwise.

    The ODE vector field is tangent on the sphere.  A final smooth retraction
    preserves the fourth-order local accuracy of the ambient RK4 update; it is
    not applied at the RK stages.
    """
    if T < 0 or dt <= 0 or every < 1:
        raise ValueError("require T >= 0, dt > 0, every >= 1")
    steps = int(round(T / dt))
    if not np.isclose(steps * dt, T, rtol=0, atol=64*np.finfo(float).eps*max(1, T)):
        raise ValueError("T must be an integer multiple of dt")
    X = normalize_rows(X0).copy()
    M, V = validate_square(M, "M"), validate_square(V, "V")
    if X.shape[1] != V.shape[0] or M.shape != V.shape:
        raise ValueError("incompatible X0, M, V")
    times, states = [0.0], [X.copy()]
    for step in range(1, steps + 1):
        k1 = attention_field(X, M, V, beta)
        k2 = attention_field(X + .5*dt*k1, M, V, beta)
        k3 = attention_field(X + .5*dt*k2, M, V, beta)
        k4 = attention_field(X + dt*k3, M, V, beta)
        X = normalize_rows(X + dt*(k1 + 2*k2 + 2*k3 + k4)/6)
        if save and (step % every == 0 or step == steps):
            times.append(step*dt); states.append(X.copy())
    return np.asarray(times), np.asarray(states) if save else X[None, ...]


@dataclass(frozen=True)
class SpectralData:
    values: np.ndarray
    vectors: np.ndarray
    top: np.ndarray
    gap: float
    positive_dominant: bool
    negative_definite: bool


def spectral_data(V: np.ndarray, tol: float = 1e-12) -> SpectralData:
    V = validate_symmetric(V, tol)
    values, vectors = np.linalg.eigh(V)  # ascending; top is last column
    top = vectors[:, -1]
    return SpectralData(values, vectors, top, float(values[-1]-values[-2]),
        bool(values[-1] > max(0.0, -values[0]) + tol), bool(values[-1] < -tol))


# Paper Fig. 2 defines rho_min.  The remaining diagnostics below are NEW NUMERICAL DIAGNOSTICS.
def pairwise_rho_min(tr: np.ndarray) -> np.ndarray:
    gram = np.einsum("tid,tjd->tij", tr, tr)
    idx = np.arange(tr.shape[1]); gram[:, idx, idx] = np.inf
    return gram.min(axis=(1, 2))


def order_parameter(tr: np.ndarray) -> np.ndarray:
    return np.linalg.norm(tr.mean(axis=1), axis=1)


def consensus_error(tr: np.ndarray) -> np.ndarray:
    return np.max(np.linalg.norm(tr - tr.mean(axis=1)[:, None, :], axis=2), axis=1)


def modal_masses(tr: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    """Paper Sec. 3.2 / Sec. 7: m_k=n^-1 sum_i c_{i,k}^2."""
    coordinates = tr @ eigenvectors
    return np.mean(coordinates**2, axis=1)


def transverse_bound_ratio(tr: np.ndarray, times: np.ndarray, eigenvectors: np.ndarray,
                           eigenvalues: np.ndarray, delta: float) -> float:
    """Maximum of R_k(t)/[R_k(0)e^{-delta(lambda_1-|lambda_k|)t}].

    This is a direct numerical diagnostic for Theorem 6.1, not a proof.  The
    eigenvalues are in numpy's ascending order, so the dominant mode is last.
    """
    C = tr @ eigenvectors
    R = np.max(np.abs(C[:, :, :-1] / C[:, :, -1, None]), axis=1)
    rates = delta * (eigenvalues[-1] - np.abs(eigenvalues[:-1]))
    bound = R[0][None, :] * np.exp(-times[:, None] * rates[None, :])
    return float(np.max(R / np.maximum(bound, np.finfo(float).tiny)))


def angular_velocity_2d(tr: np.ndarray, times: np.ndarray) -> np.ndarray:
    """NEW diagnostic: mean phase velocity in radians per unit physical time."""
    if tr.shape[2] != 2 or len(times) != len(tr): raise ValueError("requires d=2 and matching times")
    phases = np.unwrap(np.arctan2(tr[:, :, 1], tr[:, :, 0]), axis=0)
    return np.diff(phases, axis=0).mean(axis=1) / np.diff(times)


def cap_initial_condition(n: int, d: int, e_top: np.ndarray, delta: float, rng: np.random.Generator) -> np.ndarray:
    # Direct construction avoids rejection sampling and guarantees c_{i,1} >= delta.
    z = rng.normal(size=(n, d)); z -= (z @ e_top)[:, None] * e_top
    z = normalize_rows(z)
    radii = rng.uniform(delta, 1.0, size=n)
    return radii[:, None]*e_top + np.sqrt(1-radii**2)[:, None]*z


def dispersed_initial_condition(n: int, d: int, rng: np.random.Generator) -> np.ndarray:
    return normalize_rows(rng.normal(size=(n, d)))


def final_state(X0: np.ndarray, M: np.ndarray, V: np.ndarray, beta: float, T: float, dt: float) -> np.ndarray:
    return simulate_rk4(X0, M, V, beta=beta, T=T, dt=dt, save=False)[1][-1]


def rk4_refinement(X0: np.ndarray, M: np.ndarray, V: np.ndarray, beta: float, T: float, dt: float) -> tuple[float, float, float]:
    a, b, c = (final_state(X0, M, V, beta, T, h) for h in (dt, dt/2, dt/4))
    e01, e12 = np.max(np.linalg.norm(a-b, axis=1)), np.max(np.linalg.norm(b-c, axis=1))
    return float(e01), float(e12), float(e01/e12)


# ====================================================================== STAGE 0
V = validate_symmetric(np.array([[1., .5], [.5, 1.]]))
spec = spectral_data(V); beta, n, T, dt, every, delta = 1., 20, 8., .01, 10, .3
eig_res = np.linalg.norm(V @ spec.top - spec.values[-1]*spec.top)
orth_res = np.linalg.norm(spec.vectors.T @ spec.vectors - np.eye(2))
print("STAGE 0 -- spectrum")
print(f"eigenvalues={spec.values}; gap={spec.gap:.6g}; positive_dominant={spec.positive_dominant}")
print(f"top eigenpair residual={eig_res:.3e}; orthonormality residual={orth_res:.3e}")

# ====================================================================== STAGE 1
rng = np.random.default_rng(SEED)
Xcone = cap_initial_condition(n, 2, spec.top, delta, rng)
ts, tr = simulate_rk4(Xcone, V, V, beta=beta, T=T, dt=dt, every=every)
rho, order, cerr, masses = pairwise_rho_min(tr), order_parameter(tr), consensus_error(tr), modal_masses(tr, spec.vectors)
sphere_error = np.max(np.abs(np.linalg.norm(tr, axis=2)-1))
consensus_field = np.max(np.linalg.norm(symmetric_field(np.repeat(spec.top[None], n, axis=0), V, beta), axis=1))
ratio_bound = transverse_bound_ratio(tr, ts, spec.vectors, spec.values, delta)
print("\nSTAGE 1 -- symmetric Eq. (2.2) cone baseline")
print(f"sphere error={sphere_error:.3e}; e_top consensus field={consensus_field:.3e}")
print(f"final rho_min={rho[-1]:.8f}; order={order[-1]:.8f}; consensus error={cerr[-1]:.3e}; top modal mass={masses[-1,-1]:.8f}")
print(f"Theorem 6.1 transverse-bound max ratio (diagnostic)={ratio_bound:.8f}")
for tcheck in (1.0, 4.0):
    e01, e12, ratio = rk4_refinement(Xcone, V, V, beta, tcheck, .08)
    print(f"RK4 T={tcheck:g}, h=.08/.04/.02: errors={e01:.3e},{e12:.3e}; ratio={ratio:.3f}")

# ====================================================================== STAGE 2
A = np.array([[0., 1.], [-1., 0.]])
if not np.allclose(A, -A.T): raise RuntimeError("A is not antisymmetric")
Xtest = dispersed_initial_condition(n, 2, np.random.default_rng(SEED + 1))
vf_diff = np.max(np.abs(general_field(Xtest, V, V, beta) - symmetric_field(Xtest, V, beta)))
sym_final = final_state(Xtest, V, V, beta, T, dt)
gen_final = final_state(Xtest, V, V, beta, T, dt)
traj_diff = np.max(np.abs(sym_final-gen_final))
print("\nSTAGE 2 -- eps=0 regression")
print(f"A+A.T max={np.max(np.abs(A+A.T)):.3e}; vector-field max abs difference={vf_diff:.3e}; trajectory final max abs difference={traj_diff:.3e}")

# ====================================================================== STAGE 3
# A single paired control, not a parameter search. eps is intentionally fixed once.
eps_audit = 3.0; M_audit = V + eps_audit*A
ts_d, tr_d = simulate_rk4(Xtest, V, V, beta=beta, T=T, dt=dt, every=every)
ts_a, tr_a = simulate_rk4(Xtest, M_audit, V, beta=beta, T=T, dt=dt, every=every)
print("\nSTAGE 3 -- single paired asymmetric implementation audit (NO EPSILON SWEEP)")
print(f"symmetric dispersed: final order={order_parameter(tr_d)[-1]:.8f}, rho_min={pairwise_rho_min(tr_d)[-1]:.8f}")
print(f"asymmetric dispersed eps={eps_audit:g}: final order={order_parameter(tr_a)[-1]:.8f}, rho_min={pairwise_rho_min(tr_a)[-1]:.8f}")
for h in (.02, .01, .005):
    t_h, q_h = simulate_rk4(Xtest, M_audit, V, beta=beta, T=2., dt=h, every=1)
    print(f"asymmetric angular velocity at dt={h:g}: late mean={angular_velocity_2d(q_h, t_h)[-50:].mean():+.8f} rad/time")

fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
ax[0].plot(ts, rho, label=r"$\rho_{min}$ (paper Fig. 2 quantity)"); ax[0].plot(ts, order, label="order (new)"); ax[0].legend(); ax[0].set(xlabel="time", ylim=(-1.05,1.05))
ax[1].semilogy(ts, np.maximum(cerr, np.finfo(float).tiny)); ax[1].set(xlabel="time", ylabel="consensus error (new)")
ax[2].plot(ts, masses[:, -1]); ax[2].set(xlabel="time", ylabel="top averaged modal mass (paper Sec. 7)")
fig.tight_layout(); fig.savefig("stage1_symmetric_audit.png", dpi=150); plt.close(fig)
print("Saved stage1_symmetric_audit.png")
