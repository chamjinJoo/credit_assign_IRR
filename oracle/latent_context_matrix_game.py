"""
Latent-Confounded Coordination Matrix Game  (proximal-faithful version)
=======================================================================
Reference implementation for a rebuttal experiment.

WHY THIS TOY
------------
Show that *reward-prediction accuracy* is the WRONG yardstick for a
credit-assignment baseline. We build a one-shot game where the true
interventional (do-)response of the focal agent, conditioned on its OBSERVED
context (X^i_t in the paper), is known in closed form. We then show:

    (1) a naive reward regression and the proximal outcome bridge can have
        essentially identical -- or the bridge can even have WORSE -- R-prediction
        MSE, yet
    (2) the naive regression's implied credit / counterfactual baseline is biased
        by an amount that grows with the unobserved-confounding strength, while
        the bridge recovers the true interventional response.

FOUR ESTIMATORS (2x2: identification x estimation, so the comparison is fair
at BOTH levels)
-----------------------------------------------------------------------------
    (1a) fit_bridge_oracle              : proximal outcome bridge, population ORACLE
    (1b) fit_bridge_mlp                 : proximal outcome bridge, MLP trained via a
                                           regularized kernel moment objective
                                           -- the estimation-level counterpart of (1a).
    (2a) fit_obs_regression_oracle      : observational E[R|A1,W,X], population ORACLE
                                           -- the *symmetric* counterpart of (1a).
    (2b) fit_reward_mlp                 : MSE-trained neural net (estimation-level MLP)
                                           -- the estimation-level counterpart of (2a).

    Headline argument #1 uses (1a) vs (2a): BOTH are exact, infinite-sample,
    non-parametric oracles, so the remaining credit gap is PURELY an
    identification (confounding) gap -- not an approximation or optimization
    artifact. This forecloses the reviewer's "unfair comparison" objection and
    upgrades the claim from "an MLP happens to fail" to "observational reward
    regression CANNOT recover credit under confounding, even with infinite data
    and unlimited capacity".

    Headline argument #2 uses (1a) vs (1b): (1b) is a trained network fit with
    the same regularized kernel objective used by models/torch/r_pci.py:
        r^T K (tikhonov_gamma I + stabilizer_lambda K)^(-1) r / B.
    If (1b) tracks (1a), the
    credit gap between the proximal and observational approaches is not an
    artifact of using an oracle bridge -- the real estimator recovers (most
    of) the identified quantity. (1b) and (2b) share the same architecture and
    initialization, optimizer, learning rate, minibatches, and update count.
    They differ only in objective (regularized kernel moment restriction vs.
    MSE), so their estimation-level comparison isolates the objective.

    For a bandwidth ablation, fit_bridge_mlp also accepts
    kernel_mode="legacy_rbf", sigma2=<positive float>. This changes only K; the
    bridge objective and all training hyperparameters remain unchanged.

ALIGNMENT WITH THE PAPER (p.7)
------------------------------
    * pi_tilde^i  conditions on the focal agent's history tau^i   (here: proxy Z)
    * the outcome bridge takes  (action, outcome-proxy, x)  ==  h(A1, W, X)
      where X is the focal agent's OBSERVED context (NOT the latent confounder)
    * the baseline averages the do-response over actions using pi_tilde^i:
          b_i(x, tau^i) = sum_a pi_tilde^i(a | tau^i) * E[R | do(A1=a), X=x]
      -> per-state (context-conditioned) baseline.

VARIABLE MAP (paper <-> code) -- note the rename vs. the earlier draft
---------------------------------------------------------------------
    U   unobserved global confounder            (handled by proxies)   [was "X"]
    X   observed focal context (paper's x)       conditioned on, bridge arg
    Z   treatment-side proxy = focal history tau^i        pi_tilde conditions on this
    W   outcome-side proxy   = others' obs o^{-i}         bridge arg
    A1  focal action {0,1};   A2  partner action {0,1}
    R   scalar team reward

DATA-GENERATING PROCESS (all vars binary in {-1,+1}; actions in {0,1})
---------------------------------------------------------------------
    U  ~ Uniform{-1,+1}                                 unobserved,  E[U]=0
    X  ~ Uniform{-1,+1},  X ⟂ U                         observed focal context
    Z  : P(Z=U)=q,    Z ⟂ (X, ·) | U                    treatment proxy (history)
    W  : P(W=U)=p_w,  W ⟂ (Z, X) | U                    outcome proxy   (others obs)
    A1 = 1[Z=+1], flipped w.p. eps                      focal action (confounded w/ U via Z)
    A2 = 1[W=+1]                                        partner action
    R  = beta * U * A1                                  <- pure confounding (E[U]=0 => 0 true credit)
       + gamma * 1[A1==A2]                              <- coordination through W~U
       + delta * X * A1                                 <- OBSERVED heterogeneity (the true credit)
       + noise,                                         <- noise ~ N(0, sigma^2)

KEY CLOSED FORMS
----------------
True interventional response, conditioned on observed context X=x:
    E[R | do(A1=a), X=x] = gamma/2 + delta * x * a        (beta term averages to 0)
    tau(x) = E[R|do(A1=1),X=x] - E[R|do(A1=0),X=x] = delta * x
  => the true per-context credit is  delta * x , and it does NOT depend on beta.
     beta is *pure confounding*: it contributes zero true credit but drives the
     MLP's bias. Sweeping beta gives a flat true-credit line that the bridge
     tracks and the MLP departs from -- the cleanest possible demonstration.

Naive-MLP credit (fits observational E[R|A1,W,X], then plugs in do):
    tau_MLP(x) = delta * x  +  beta * ubar(1)
    where ubar(a) = E_{W~marg}[ E[U | A1=a, W] ] != E[U|do(a)] = 0 whenever the
    proxy W is noisy (p_w < 1). The bias term is nonzero and grows with beta and
    with confounding strength; it vanishes at beta=0 or p_w=1 (the "diagnostic /
    weak-confounding" regime where both methods agree).

PCI assumptions (hold by construction; see verify_assumptions):
    treatment proxy : Z ⟂ R | (A1, U, X)
    outcome proxy   : W ⟂ A1 | (U, X)   and W informative about R through U
    completeness    : q, p_w in (1/2, 1], eps in (0, 1/2)

NOTE 
* fit_bridge_oracle and fit_obs_regression_oracle enumerate the known
    population DGP exactly. The empirical closed-form plug-in estimators are
    retained separately as finite-sample diagnostics.
* pi_tilde is stubbed as the empirical P(A1=a | Z=z). REPLACE with the
    paper's renormalized policy from p.7. The main plots track tau (the
    do-response DIFFERENCE), which is pi_tilde-independent, so the stub does
    not affect the headline result.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
class GameConfig:
    def __init__(
        self,
        beta: float = 1.0,    # unobserved-confounding strength (latent -> reward via A1)
        delta: float = 1.0,   # OBSERVED heterogeneity -> the TRUE per-context credit
        gamma: float = 0.5,   # coordination bonus 1[A1==A2]
        q: float = 0.85,      # treatment proxy fidelity  P(Z=U)   [confounding knob]
        p_w: float = 0.85,    # outcome  proxy fidelity  P(W=U)   [proxy-noise knob]
        eps: float = 0.05,    # focal exploration (action flip prob) -> positivity
        sigma: float = 0.1,   # reward noise std
    ):
        assert 0.5 < q <= 1.0 and 0.5 < p_w <= 1.0
        assert 0.0 < eps < 0.5
        self.beta, self.delta, self.gamma = beta, delta, gamma
        self.q, self.p_w, self.eps, self.sigma = q, p_w, eps, sigma


# --------------------------------------------------------------------------- #
# Data-generating process
# --------------------------------------------------------------------------- #
def _flip(base: np.ndarray, fidelity: float, rng: np.random.Generator) -> np.ndarray:
    keep = rng.random(base.shape) < fidelity
    return np.where(keep, base, -base)


def sample_dataset(cfg: GameConfig, n: int, seed: int = 0) -> dict:
    """Sample n episodes. Returns dict of arrays. A1,A2 in {0,1}; U,X,Z,W in {-1,+1}."""
    rng = np.random.default_rng(seed)

    U = rng.choice(np.array([-1, 1]), size=n)          # unobserved confounder
    X = rng.choice(np.array([-1, 1]), size=n)          # observed focal context; X ⟂ U
    Z = _flip(U, cfg.q, rng)                           # treatment proxy (focal history)
    W = _flip(U, cfg.p_w, rng)                         # outcome proxy (others' obs); W ⟂ Z | U

    a1 = (Z == 1).astype(int)
    flip = rng.random(n) < cfg.eps
    A1 = np.where(flip, 1 - a1, a1)                    # exploration -> positivity
    A2 = (W == 1).astype(int)

    noise = rng.normal(0.0, cfg.sigma, size=n)
    R = (cfg.beta * U * A1
         + cfg.gamma * (A1 == A2).astype(float)
         + cfg.delta * X * A1
         + noise)

    return dict(U=U, X=X, Z=Z, W=W, A1=A1, A2=A2, R=R)


# --------------------------------------------------------------------------- #
# Ground-truth interventional response (closed form), conditioned on observed X
# --------------------------------------------------------------------------- #
def true_do_response(cfg: GameConfig, a: int, x: int) -> float:
    """E[R | do(A1=a), X=x].  beta term -> 0 since E[U]=0 under do()."""
    return 0.5 * cfg.gamma + cfg.delta * x * a


def true_tau(cfg: GameConfig, x: int) -> float:
    """tau(x) = E[R|do(A1=1),X=x] - E[R|do(A1=0),X=x] = delta * x  (independent of beta)."""
    return cfg.delta * x


def true_marginal_tau(cfg: GameConfig) -> float:
    return 0.5 * (true_tau(cfg, +1) + true_tau(cfg, -1))   # = 0


# --------------------------------------------------------------------------- #
# Exact population probabilities and rewards
# --------------------------------------------------------------------------- #
def _proxy_prob(value: int, u: int, fidelity: float) -> float:
    return fidelity if value == u else 1.0 - fidelity


def _action_prob(a: int, z: int, eps: float) -> float:
    preferred = int(z == 1)
    return 1.0 - eps if a == preferred else eps


def _mean_reward(cfg: GameConfig, u: int, x: int, w: int, a: int) -> float:
    a2 = int(w == 1)
    return cfg.beta * u * a + cfg.gamma * float(a == a2) + cfg.delta * x * a


def _joint_prob(cfg: GameConfig, u: int, x: int, z: int, w: int, a: int) -> float:
    return (
        0.25
        * _proxy_prob(z, u, cfg.q)
        * _proxy_prob(w, u, cfg.p_w)
        * _action_prob(a, z, cfg.eps)
    )


# --------------------------------------------------------------------------- #
# Estimator 1a: proximal outcome bridge h(a,w,x), exact population oracle
# --------------------------------------------------------------------------- #
def fit_bridge_oracle(cfg: GameConfig) -> dict:
    """Solve the population 2x2 bridge systems by exact DGP enumeration."""
    h = {}
    for a in (0, 1):
        M = np.zeros((2, 2))
        for zi, z in enumerate((-1, 1)):
            weights = {
                w: sum(_joint_prob(cfg, u, 1, z, w, a) for u in (-1, 1))
                for w in (-1, 1)
            }
            denom = sum(weights.values())
            for wi, w in enumerate((-1, 1)):
                M[zi, wi] = weights[w] / denom

        for x in (-1, 1):
            r = np.zeros(2)
            for zi, z in enumerate((-1, 1)):
                numerator = 0.0
                denom = 0.0
                for u in (-1, 1):
                    for w in (-1, 1):
                        prob = _joint_prob(cfg, u, x, z, w, a)
                        numerator += prob * _mean_reward(cfg, u, x, w, a)
                        denom += prob
                r[zi] = numerator / denom
            values = np.linalg.solve(M, r)
            for wi, w in enumerate((-1, 1)):
                h[(a, w, x)] = float(values[wi])
    return h


def bridge_do_response(h: dict, a: int, x: int) -> float:
    """Population E[R|do(A1=a),X=x]; W is marginally uniform in this DGP."""
    return 0.5 * sum(h[(a, w, x)] for w in (-1, 1))


# --------------------------------------------------------------------------- #
# Finite-sample closed-form bridge plug-in estimator
# --------------------------------------------------------------------------- #
# Moment restriction:  E[ R - h(A1, W, X) | A1=a, Z=z, X=x ] = 0  for each (a,z,x).
# For fixed (a, x): 2x2 system over w indexed by z
#     r_{a,x}[z] = sum_w P(W=w | A1=a, Z=z) h(a, w, x)   ->  h = M_a^{-1} r_{a,x}
# M_a is x-independent (X ⟂ (U,W)), estimated by pooling over x for stability.
def fit_bridge_closed_form(data: dict) -> dict:
    """Estimate h by empirical cell probabilities and means."""
    A1, Z, W, X, R = (data[k] for k in ("A1", "Z", "W", "X", "R"))
    w_vals, z_vals, x_vals = np.array([-1, 1]), np.array([-1, 1]), np.array([-1, 1])

    # M_a[z, w] = P(W=w | A1=a, Z=z)   (pool over x)
    M = {}
    for a in (0, 1):
        Ma = np.zeros((2, 2))
        for zi, z in enumerate(z_vals):
            mask = (A1 == a) & (Z == z)
            if mask.sum() == 0:
                raise ValueError(f"No samples for A1={a}, Z={z}; increase n/eps.")
            for wj, w in enumerate(w_vals):
                Ma[zi, wj] = (W[mask] == w).mean()
        M[a] = Ma

    h = {}
    for a in (0, 1):
        for x in x_vals:
            r = np.zeros(2)
            for zi, z in enumerate(z_vals):
                mask = (A1 == a) & (Z == z) & (X == x)
                if mask.sum() == 0:
                    raise ValueError(f"No samples for A1={a}, Z={z}, X={x}.")
                r[zi] = R[mask].mean()                    # E[R | A1=a, Z=z, X=x]
            h_ax = np.linalg.solve(M[a], r)
            for wj, w in enumerate(w_vals):
                h[(a, int(w), int(x))] = float(h_ax[wj])
    return h


# --------------------------------------------------------------------------- #
# Estimator 1b: proximal outcome bridge trained with the r_pci.py objective
# --------------------------------------------------------------------------- #
#     L(phi) = r^T K (tikhonov_gamma I + stabilizer_lambda K)^(-1) r / B
#     r_i = R_i - h_phi(A1_i,W_i,X_i), c_i = (Z_i,A1_i,X_i).
# By default K uses the same product of standardized RBF blocks as the active
# Torch path. kernel_mode="legacy_rbf" selects exp(-||c-c'||^2/(2 sigma2)) on
# the natural (Z,A1,X) encoding for the legacy bandwidth ablation.
TRAIN_STEPS = 4000
TRAIN_BATCH_SIZE = 128
TRAIN_LR = 5e-3
LEGACY_SIGMA2 = 0.05


def _cond_features(A1: np.ndarray, Z: np.ndarray, X: np.ndarray) -> torch.Tensor:
    """Conditioning variables in their natural encodings."""
    A1f = A1.astype(np.float32)
    Zf = Z.astype(np.float32)
    Xf = X.astype(np.float32)
    return torch.tensor(np.stack([Zf, A1f, Xf], axis=1), dtype=torch.float32)


def _standardize(x: torch.Tensor) -> torch.Tensor:
    return (x - x.mean(0, keepdim=True)) / x.std(
        0, keepdim=True, unbiased=False
    ).clamp_min(1e-6)


def _rbf_kernel(x: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    x2 = x.pow(2).sum(dim=-1, keepdim=True)
    d2 = (x2 - 2.0 * x @ x.T + x2.T).clamp_min(0.0)
    return torch.exp(-gamma * d2)


def _conditioning_kernel(
    c: torch.Tensor,
    mode: str = "ours",
    sigma2: float = LEGACY_SIGMA2,
) -> torch.Tensor:
    if mode == "legacy_rbf":
        if sigma2 <= 0.0:
            raise ValueError(f"sigma2 must be positive, got {sigma2}")
        d2 = torch.cdist(c, c).pow(2)
        return torch.exp(-d2 / (2.0 * sigma2))
    if mode != "ours":
        raise ValueError(f"Unsupported kernel mode: {mode}")
    z = _standardize(c[:, 0:1])
    a = torch.nn.functional.one_hot(c[:, 1].long(), 2).float()
    x = _standardize(c[:, 2:3])
    return _rbf_kernel(z) * _rbf_kernel(a) * _rbf_kernel(x)


def _batch_indices(n: int, steps: int, batch: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    size = min(batch, n)
    for _ in range(steps):
        yield torch.randint(0, n, (size,), generator=gen)


def _regularized_bridge_loss(
    residual: torch.Tensor,
    kernel: torch.Tensor,
    tikhonov_gamma: float,
    stabilizer_lambda: float,
) -> torch.Tensor:
    B = len(residual)
    eye = torch.eye(B, dtype=kernel.dtype, device=kernel.device)
    system = tikhonov_gamma * eye + stabilizer_lambda * kernel
    solved = torch.linalg.solve(system, residual.unsqueeze(1))
    weighted = kernel @ solved
    return (residual.unsqueeze(1) * weighted).sum() / B


def fit_bridge_mlp(
    data: dict,
    steps: int = TRAIN_STEPS,
    batch: int = TRAIN_BATCH_SIZE,
    lr: float = TRAIN_LR,
    tikhonov_gamma: float = 1e-4,
    stabilizer_lambda: float = 1.0,
    kernel_mode: str = "ours",
    sigma2: float = LEGACY_SIGMA2,
    hidden: int = 64,
    seed: int = 0,
) -> "RewardMLP":
    """Fit h with the regularized bridge objective and selected kernel."""
    Xin = _features(data["A1"], data["W"], data["X"])
    Cin = _cond_features(data["A1"], data["Z"], data["X"])
    y = torch.tensor(data["R"], dtype=torch.float32)

    torch.manual_seed(seed)
    model = RewardMLP(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)

    for idx in _batch_indices(n, steps, batch, seed):
        residual = y[idx] - model(Xin[idx])
        kernel = _conditioning_kernel(Cin[idx], mode=kernel_mode, sigma2=sigma2)
        loss = _regularized_bridge_loss(
            residual, kernel, tikhonov_gamma, stabilizer_lambda
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def bridge_loss_at(
    model_or_table,
    data: dict,
    m: int = TRAIN_BATCH_SIZE,
    seed: int = 0,
    is_table: bool = False,
    tikhonov_gamma: float = 1e-4,
    stabilizer_lambda: float = 1.0,
    kernel_mode: str = "ours",
    sigma2: float = LEGACY_SIGMA2,
) -> float:
    """Evaluate the regularized bridge objective on one evaluation batch."""
    A1, Z, W, X, R = (data[k] for k in ("A1", "Z", "W", "X", "R"))
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(R), generator=gen)[:min(m, len(R))].numpy()
    if is_table:
        values = np.array([
            model_or_table[(int(A1[i]), int(W[i]), int(X[i]))]
            for i in idx
        ])
        values = torch.tensor(values, dtype=torch.float32)
    else:
        inputs = _features(A1[idx], W[idx], X[idx])
        with torch.no_grad():
            values = model_or_table(inputs)
    residual = torch.tensor(R[idx], dtype=torch.float32) - values
    kernel = _conditioning_kernel(
        _cond_features(A1[idx], Z[idx], X[idx]),
        mode=kernel_mode,
        sigma2=sigma2,
    )
    return float(_regularized_bridge_loss(
        residual, kernel, tikhonov_gamma, stabilizer_lambda
    ))


# --------------------------------------------------------------------------- #
# Estimator 2: naive MLP reward model  R ~ f(A1, W, X)   (MSE)
# --------------------------------------------------------------------------- #
class RewardMLP(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _features(A1: np.ndarray, W: np.ndarray, X: np.ndarray) -> torch.Tensor:
    """Model sees (A1, W, X) -- the SAME inputs as the bridge, on purpose.
    Identical information, yet the MLP fits observational E[R|A1,W,X] and plugging
    in do() leaves residual confounding; the bridge corrects it via the Z moment."""
    A1f = A1.astype(np.float32)
    Wf = ((W + 1) / 2).astype(np.float32)
    Xf = ((X + 1) / 2).astype(np.float32)
    return torch.tensor(np.stack([A1f, Wf, Xf], axis=1), dtype=torch.float32)


def fit_reward_mlp(
    data: dict,
    steps: int = TRAIN_STEPS,
    batch: int = TRAIN_BATCH_SIZE,
    lr: float = TRAIN_LR,
    hidden: int = 64,
    seed: int = 0,
) -> RewardMLP:
    """Fit E[R|A1,W,X] under the same training protocol as fit_bridge_mlp."""
    torch.manual_seed(seed)
    Xin = _features(data["A1"], data["W"], data["X"])
    y = torch.tensor(data["R"], dtype=torch.float32)
    model = RewardMLP(hidden=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(y)
    for idx in _batch_indices(n, steps, batch, seed):
        opt.zero_grad()
        nn.functional.mse_loss(model(Xin[idx]), y[idx]).backward()
        opt.step()
    return model


def mlp_do_response(model: RewardMLP, a: int, x: int) -> float:
    """Plug in A1=a,X=x and average over the population marginal P(W)=1/2."""
    out = 0.0
    for w in (-1, 1):
        feat = _features(np.array([a]), np.array([w]), np.array([x]))
        with torch.no_grad():
            out += 0.5 * float(model(feat).item())
    return out


# --------------------------------------------------------------------------- #
# Estimator 2a: exact population observational regression g(a,w,x)
# --------------------------------------------------------------------------- #
def fit_obs_regression_oracle(cfg: GameConfig) -> dict:
    """Enumerate the exact population E[R|A1=a,W=w,X=x]."""
    g = {}
    for a in (0, 1):
        for w in (-1, 1):
            for x in (-1, 1):
                numerator = 0.0
                denom = 0.0
                for u in (-1, 1):
                    for z in (-1, 1):
                        prob = _joint_prob(cfg, u, x, z, w, a)
                        numerator += prob * _mean_reward(cfg, u, x, w, a)
                        denom += prob
                g[(a, w, x)] = numerator / denom
    return g


def obs_regression_do_response(g: dict, a: int, x: int) -> float:
    """Naive population plug-in response, averaging over P(W)=1/2."""
    return 0.5 * sum(g[(a, w, x)] for w in (-1, 1))


# Finite-sample saturated plug-in diagnostic.
def fit_obs_regression_closed_form(data: dict) -> dict:
    """Estimate g[(a,w,x)] using empirical cell means."""
    A1, W, X, R = (data[k] for k in ("A1", "W", "X", "R"))
    g = {}
    for a in (0, 1):
        for w in (-1, 1):
            for x in (-1, 1):
                mask = (A1 == a) & (W == w) & (X == x)
                if mask.sum() == 0:
                    raise ValueError(f"Empty cell A1={a},W={w},X={x}; increase n.")
                g[(a, w, x)] = float(R[mask].mean())
    return g


def population_prediction_mse(cfg: GameConfig, predictor) -> float:
    """Exact population reward-prediction MSE, including noise variance."""
    mse = 0.0
    for u in (-1, 1):
        for x in (-1, 1):
            for z in (-1, 1):
                for w in (-1, 1):
                    for a in (0, 1):
                        prob = _joint_prob(cfg, u, x, z, w, a)
                        mean = _mean_reward(cfg, u, x, w, a)
                        pred = predictor(a, w, x)
                        mse += prob * ((mean - pred) ** 2 + cfg.sigma ** 2)
    return float(mse)


def _pred_mse_table(table: dict, cfg: GameConfig) -> float:
    return population_prediction_mse(cfg, lambda a, w, x: table[(a, w, x)])


def _pred_mse_mlp(model: RewardMLP, cfg: GameConfig) -> float:
    def predict(a, w, x):
        feat = _features(np.array([a]), np.array([w]), np.array([x]))
        with torch.no_grad():
            return float(model(feat).item())
    return population_prediction_mse(cfg, predict)


# --------------------------------------------------------------------------- #
# Per-state counterfactual baseline  b_i(x, tau^i)  averaged over pi_tilde(a|z)
# --------------------------------------------------------------------------- #
def counterfactual_baseline(do_response_fn, data: dict, cfg: GameConfig) -> np.ndarray:
    """b_i = sum_a pi_tilde(a | Z) * E[R | do(A1=a), X].
    pi_tilde(a|z) stubbed as empirical P(A1=a | Z=z). do_response_fn(a, x) -> float.
    Returns one baseline value per sample (conditioned on that sample's X and Z)."""
    A1, Z, X = data["A1"], data["Z"], data["X"]
    # empirical pi_tilde(a=1 | z)
    pi1 = {z: A1[Z == z].mean() for z in (-1, 1)}
    do = {(a, x): do_response_fn(a, x) for a in (0, 1) for x in (-1, 1)}
    b = np.empty(len(A1))
    for i in range(len(A1)):
        z, x = int(Z[i]), int(X[i])
        b[i] = (1 - pi1[z]) * do[(0, x)] + pi1[z] * do[(1, x)]
    return b


# --------------------------------------------------------------------------- #
# Assumption checks
# --------------------------------------------------------------------------- #
def verify_assumptions(data: dict) -> None:
    U, X, Z, W, A1 = (data[k] for k in ("U", "X", "Z", "W", "A1"))

    def cond_corr(u, v, given, name):
        dev = 0.0
        for g in np.unique(given):
            m = given == g
            if m.sum() < 20:
                continue
            uu, vv = u[m].astype(float), v[m].astype(float)
            if uu.std() < 1e-8 or vv.std() < 1e-8:
                continue
            dev = max(dev, abs(np.corrcoef(uu, vv)[0, 1]))
        print(f"  {'OK ' if dev < 0.15 else '!! '}{name}: max|corr| = {dev:.3f}")

    print("Assumption checks (want small within-stratum correlations):")
    cond_corr(Z, W, U, "Z ⟂ W | U")
    cond_corr(W, A1, U, "W ⟂ A1 | U")
    cond_corr(X, U, np.zeros_like(U), "X ⟂ U (unconditional)")
    for a in (0, 1):
        for z in (-1, 1):
            for x in (-1, 1):
                n = ((A1 == a) & (Z == z) & (X == x)).sum()
                if n < 50:
                    print(f"  !! thin cell A1={a},Z={z},X={x}: n={n}")


# --------------------------------------------------------------------------- #
# Confounding sweep
# --------------------------------------------------------------------------- #
def run_sweep(
    sweep_over: str,
    values,
    n: int = 150_000,
    base: dict | None = None,
    seed: int = 0,
    with_mlp: bool = True,
    with_bridge_mlp: bool = False,
    mlp_kwargs: dict | None = None,
    bridge_kwargs: dict | None = None,
) -> dict:
    """Run oracle and optional MLP comparisons under a shared training protocol."""
    base = base or {}
    mlp_kwargs = mlp_kwargs or {}
    bridge_kwargs = bridge_kwargs or {}
    rows = []
    for v in values:
        kw = dict(base)
        kw[sweep_over] = float(v)
        cfg = GameConfig(**kw)
        data = (
            sample_dataset(cfg, n=n, seed=seed)
            if with_mlp or with_bridge_mlp else None
        )

        h = fit_bridge_oracle(cfg)
        g = fit_obs_regression_oracle(cfg)
        mlp = fit_reward_mlp(data, seed=seed, **mlp_kwargs) if with_mlp else None
        bridge_fit_kwargs = dict(mlp_kwargs)
        bridge_fit_kwargs.update(bridge_kwargs)
        bmlp = (
            fit_bridge_mlp(data, seed=seed, **bridge_fit_kwargs)
            if with_bridge_mlp else None
        )

        tau_err_bridge = tau_err_obs = tau_err_mlp = tau_err_bridge_mlp = 0.0
        for x in (-1, 1):
            target = true_tau(cfg, x)
            bridge_tau = bridge_do_response(h, 1, x) - bridge_do_response(h, 0, x)
            obs_tau = (
                obs_regression_do_response(g, 1, x)
                - obs_regression_do_response(g, 0, x)
            )
            tau_err_bridge = max(tau_err_bridge, abs(bridge_tau - target))
            tau_err_obs = max(tau_err_obs, abs(obs_tau - target))
            if mlp is not None:
                mlp_tau = mlp_do_response(mlp, 1, x) - mlp_do_response(mlp, 0, x)
                tau_err_mlp = max(tau_err_mlp, abs(mlp_tau - target))
            if bmlp is not None:
                bmlp_tau = mlp_do_response(bmlp, 1, x) - mlp_do_response(bmlp, 0, x)
                tau_err_bridge_mlp = max(tau_err_bridge_mlp, abs(bmlp_tau - target))

        rows.append(dict(
            value=float(v),
            tau_err_bridge=tau_err_bridge,
            tau_err_obs=tau_err_obs,
            tau_err_mlp=(tau_err_mlp if mlp is not None else None),
            tau_err_bridge_mlp=(tau_err_bridge_mlp if bmlp is not None else None),
            mse_bridge=_pred_mse_table(h, cfg),
            mse_obs=_pred_mse_table(g, cfg),
            mse_mlp=(_pred_mse_mlp(mlp, cfg) if mlp is not None else None),
            mse_bridge_mlp=(_pred_mse_mlp(bmlp, cfg) if bmlp is not None else None),
        ))
        msg = (
            f"{sweep_over}={v:.3f} | tau_err bridge={tau_err_bridge:.3f} "
            f"obs-oracle={tau_err_obs:.3f}"
        )
        if mlp is not None:
            msg += f" obs-mlp={tau_err_mlp:.3f}"
        if bmlp is not None:
            msg += f" bridge-mlp={tau_err_bridge_mlp:.3f}"
        print(msg)
    return dict(sweep_over=sweep_over, values=[r["value"] for r in rows], rows=rows)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def make_plots(sweep: dict, out_prefix: str = "sweep") -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vals, rows = sweep["values"], sweep["rows"]
    xlabel = {"beta": r"unobserved-confounding strength  $\beta$",
              "q": "treatment proxy fidelity  q = P(Z=U)",
              "p_w": r"outcome proxy fidelity  $p_w = P(W=U)$",
              "delta": r"observed heterogeneity  $\delta$"}.get(
                  sweep["sweep_over"], sweep["sweep_over"])
    paths = []

    has_mlp = rows[0].get("tau_err_mlp") is not None
    has_bridge_mlp = rows[0].get("tau_err_bridge_mlp") is not None

    # Fig A: credit error. obs-oracle vs bridge = PURE identification gap (symmetric).
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, [r["tau_err_obs"] for r in rows], "o-",
            label="Obs. regression (oracle)", color="#c1272d")
    if has_mlp:
        ax.plot(vals, [r["tau_err_mlp"] for r in rows], "^--",
                label="Obs. regression (MLP, trained)", color="#e08214", alpha=0.8)
    ax.plot(vals, [r["tau_err_bridge"] for r in rows], "s-",
            label="Proximal bridge (oracle)", color="#0057b7")
    if has_bridge_mlp:
        ax.plot(vals, [r["tau_err_bridge_mlp"] for r in rows], "d--",
                label="Proximal bridge (MLP, trained)", color="#2ca02c", alpha=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel(r"credit error  $\max_x|\hat\tau(x)-\tau(x)|$")
    ax.set_title("Credit error vs confounding strength\n"
                 "(two oracles => gap is pure identification, not estimation)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = f"{out_prefix}_A_credit_error.png"; fig.savefig(p, dpi=150)
    paths.append(p); plt.close(fig)

    # Fig B: exact population R-prediction MSE — the wrong yardstick.
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(vals, [r["mse_obs"] for r in rows], "o-",
            label="Obs. regression (oracle)", color="#c1272d")
    if has_mlp:
        ax.plot(vals, [r["mse_mlp"] for r in rows], "^--",
                label="Obs. regression (MLP, trained)", color="#e08214", alpha=0.8)
    ax.plot(vals, [r["mse_bridge"] for r in rows], "s-",
            label="Proximal bridge (oracle)", color="#0057b7")
    if has_bridge_mlp:
        ax.plot(vals, [r["mse_bridge_mlp"] for r in rows], "d--",
                label="Proximal bridge (MLP, trained)", color="#2ca02c", alpha=0.8)
    ax.set_xlabel(xlabel); ax.set_ylabel("population reward-prediction MSE")
    ax.set_title("R-prediction MSE\n(the WRONG yardstick — bridge can even be worse)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); p = f"{out_prefix}_B_mse.png"; fig.savefig(p, dpi=150)
    paths.append(p); plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test():
    print("=" * 70)
    print("SELF-TEST: matched bridge objective and observational MLP protocol")
    print("=" * 70)
    cfg = GameConfig(beta=1.0, delta=1.0, gamma=0.5, q=0.85, p_w=0.85, eps=0.05)
    data = sample_dataset(cfg, n=50_000, seed=1)
    verify_assumptions(data)

    h = fit_bridge_oracle(cfg)
    g = fit_obs_regression_oracle(cfg)
    mlp = fit_reward_mlp(data, seed=1)
    bmlp = fit_bridge_mlp(data, seed=1)

    print("\ndo-response E[R|do(a),x]: true / bridge / obs-oracle / obs-mlp")
    for a in (0, 1):
        for x in (-1, 1):
            target = true_do_response(cfg, a, x)
            bridge = bridge_do_response(h, a, x)
            obs = obs_regression_do_response(g, a, x)
            pred = mlp_do_response(mlp, a, x)
            print(
                f"  a={a} x={x:+d} | true={target:+.3f} bridge={bridge:+.3f} "
                f"obs={obs:+.3f} obs-mlp={pred:+.3f}"
            )

    errors = {"bridge": 0.0, "obs": 0.0, "obs_mlp": 0.0, "bridge_mlp": 0.0}
    print("\ntau(x): true / bridge / obs-oracle / obs-mlp / bridge-mlp")
    for x in (-1, 1):
        target = true_tau(cfg, x)
        estimates = {
            "bridge": bridge_do_response(h, 1, x) - bridge_do_response(h, 0, x),
            "obs": obs_regression_do_response(g, 1, x) - obs_regression_do_response(g, 0, x),
            "obs_mlp": mlp_do_response(mlp, 1, x) - mlp_do_response(mlp, 0, x),
            "bridge_mlp": mlp_do_response(bmlp, 1, x) - mlp_do_response(bmlp, 0, x),
        }
        for name, estimate in estimates.items():
            errors[name] = max(errors[name], abs(estimate - target))
        print(
            f"  x={x:+d} | true={target:+.3f} bridge={estimates['bridge']:+.3f} "
            f"obs={estimates['obs']:+.3f} obs-mlp={estimates['obs_mlp']:+.3f} "
            f"bridge-mlp={estimates['bridge_mlp']:+.3f}"
        )

    print(
        "\npopulation R-pred MSE: "
        f"bridge={_pred_mse_table(h, cfg):.4f} "
        f"obs-oracle={_pred_mse_table(g, cfg):.4f} "
        f"obs-mlp={_pred_mse_mlp(mlp, cfg):.4f} "
        f"bridge-mlp={_pred_mse_mlp(bmlp, cfg):.4f}"
    )
    print(
        "max credit error: "
        + " ".join(f"{name}={value:.3f}" for name, value in errors.items())
    )

    assert errors["bridge"] < 1e-10, "population bridge oracle must be exact"
    assert errors["obs"] > 0.3, "observational oracle must retain confounding bias"
    assert errors["bridge_mlp"] < errors["obs_mlp"], (
        "matched-protocol bridge MLP should improve on observational MLP credit"
    )

    print("\nRegularized bridge objective on the same held batch:")
    print(f"  loss(bridge oracle) = {bridge_loss_at(h, data, is_table=True):+.6f}")
    print(f"  loss(obs-regression oracle) = {bridge_loss_at(g, data, is_table=True):+.6f}")

    cfg0 = GameConfig(beta=0.0, delta=1.0, gamma=0.5, q=0.85, p_w=0.85, eps=0.05)
    h0 = fit_bridge_oracle(cfg0)
    g0 = fit_obs_regression_oracle(cfg0)
    bridge0 = bridge_do_response(h0, 1, -1) - bridge_do_response(h0, 0, -1)
    obs0 = obs_regression_do_response(g0, 1, -1) - obs_regression_do_response(g0, 0, -1)
    assert abs(bridge0 - obs0) < 1e-10, "oracles must agree when beta=0"
    print("SELF-TEST PASSED ✓")


if __name__ == "__main__":
    _self_test()
    # Pure-identification sweeps use the two ORACLES (fast, no NN training).
    # Set with_mlp=True to overlay the trained observational MLP, and/or
    # with_bridge_mlp=True to overlay the trained proximal bridge MLP (both
    # slower -- the bridge MLP especially so, since it needs minibatched
    # regularized kernel solves at every sweep point).
    print("\n" + "=" * 70 + "\nSWEEP over beta (pure confounding strength)\n" + "=" * 70)
    s = run_sweep("beta", np.linspace(0.0, 2.0, 11),
                  base=dict(delta=1.0, gamma=0.5, q=0.85, p_w=0.85, eps=0.05),
                  n=120_000, with_mlp=False)
    print("saved:", make_plots(s, "sweep_beta"))
    print("\n" + "=" * 70 + "\nSWEEP over p_w (proxy noise)\n" + "=" * 70)
    s2 = run_sweep("p_w", np.linspace(0.55, 1.0, 10),
                   base=dict(beta=1.0, delta=1.0, gamma=0.5, q=0.85, eps=0.05),
                   n=120_000, with_mlp=False)
    print("saved:", make_plots(s2, "sweep_pw"))
