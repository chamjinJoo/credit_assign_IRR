"""
Bridge Gram matrix solver 수치 안정성 비교

비교 대상:
  A. Cholesky + 1e-3 고정 fallback  (이전 코드)
  B. eigh + 1e-3 고정 fallback      (중간 단계)
  C. eigh, 1e-3 없음 (현재 적용)   → r_pci.py 현재 코드와 동일

분석 항목:
  1. seed 별 차이 (3가지 데이터 시나리오 × 2가지 tikhonov_gamma)
  2. device 별 차이 (CPU vs GPU, 동일 seed에서 결과가 얼마나 다른가)
"""

import os
import jax
import jax.numpy as jnp
import numpy as np

# x64 모드 활성화 (import 직후 가장 먼저 설정해야 함)
jax.config.update("jax_enable_x64", True)

# CPU 강제 접근을 위해 미리 device 확인
_CPU = jax.devices("cpu")[0]
try:
    _GPU = jax.devices("gpu")[0]
    _HAS_GPU = True
except RuntimeError:
    _GPU = _CPU
    _HAS_GPU = False


# ── 데이터 생성 ────────────────────────────────────────────────────────────

def rbf_gram(x: jnp.ndarray, gamma: float) -> jnp.ndarray:
    diff = x[:, None, :] - x[None, :, :]
    dists_sq = jnp.sum(jnp.square(diff), axis=-1)
    return jnp.exp(-gamma * dists_sq)


def make_random_data(key, batch_size, z_dim, obs_dim, action_dim):
    """시나리오 1: 완전 랜덤"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    z             = jax.random.normal(k1, (batch_size, z_dim))
    obs           = jax.random.normal(k2, (batch_size, obs_dim))
    act_idx       = jax.random.randint(k3, (batch_size,), 0, action_dim)
    action_onehot = jax.nn.one_hot(act_idx, action_dim, dtype=jnp.float32)
    residuals     = jax.random.normal(k4, (batch_size, 1))
    return z, obs, action_onehot, residuals


def make_smax_like_data(key, batch_size, z_dim, obs_dim, action_dim, cluster_noise=0.05):
    """시나리오 2/3: SMAX-like (클러스터링된 행동/관측/은닉 상태)"""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    centers_z       = jax.random.normal(k1, (3, z_dim))
    cluster_ids     = jax.random.randint(k2, (batch_size,), 0, 3)
    z               = centers_z[cluster_ids] + cluster_noise * jax.random.normal(k3, (batch_size, z_dim))
    centers_obs     = jax.random.normal(k3, (2, obs_dim))
    obs_cluster_ids = jax.random.randint(k4, (batch_size,), 0, 2)
    obs             = centers_obs[obs_cluster_ids] + cluster_noise * jax.random.normal(k1, (batch_size, obs_dim))
    uniform         = jax.random.uniform(k2, (batch_size,))
    rand_act        = jax.random.randint(k4, (batch_size,), 0, action_dim - 1)
    act_idx         = jnp.where(uniform < 0.8, action_dim - 1, rand_act)
    action_onehot   = jax.nn.one_hot(act_idx, action_dim, dtype=jnp.float32)
    residuals       = jax.random.normal(k4, (batch_size, 1))
    return z, obs, action_onehot, residuals


def build_gram(z, obs, action_onehot, z_dim, obs_dim, action_dim):
    def standardize(x):
        return (x - x.mean(0, keepdims=True)) / jnp.clip(x.std(0, keepdims=True), 1e-6)
    return (
        rbf_gram(standardize(z),    1.0 / max(1, z_dim))
        * rbf_gram(action_onehot,   1.0 / max(1, action_dim))
        * rbf_gram(standardize(obs), 1.0 / max(1, obs_dim))
    )


# ── solver 3종 ─────────────────────────────────────────────────────────────

def solve_0(gram, residuals, gamma):
    """0. linalg.solve f32 (최초 버전)"""
    n   = gram.shape[0]
    reg = (gamma + 1e-3) * jnp.eye(n, dtype=gram.dtype) + gram
    return jnp.linalg.solve(reg, residuals)


def solve_A(gram, residuals, gamma):
    """A. Cholesky + 1e-3 f32 (이전 코드)"""
    n    = gram.shape[0]
    reg  = (gamma + 1e-3) * jnp.eye(n, dtype=gram.dtype) + gram
    chol = jnp.linalg.cholesky(reg)
    return jax.scipy.linalg.cho_solve((chol, True), residuals)


def solve_B(gram, residuals, gamma):
    """B. eigh + 1e-3 f32 (중간 단계)"""
    n               = gram.shape[0]
    reg             = (gamma + 1e-3) * jnp.eye(n, dtype=gram.dtype) + gram
    eigs, vecs      = jnp.linalg.eigh(reg)
    safe_inv        = jnp.where(eigs > 1e-4 * eigs[-1], 1.0 / eigs, 0.0)
    return vecs @ (safe_inv[:, None] * (vecs.T @ residuals))


def solve_C(gram, residuals, gamma):
    """C. eigh, 1e-3 없음 f32 — 현재 r_pci.py 적용 코드"""
    n               = gram.shape[0]
    reg             = gamma * jnp.eye(n, dtype=gram.dtype) + gram
    eigs, vecs      = jnp.linalg.eigh(reg)
    safe_inv        = jnp.where(eigs > 1e-4 * eigs[-1], 1.0 / eigs, 0.0)
    return vecs @ (safe_inv[:, None] * (vecs.T @ residuals))


def _to_f64(x):
    return x.astype(jnp.float64)


def solve_0_f64(gram, residuals, gamma):
    """0x. linalg.solve f64"""
    gram, residuals = _to_f64(gram), _to_f64(residuals)
    n   = gram.shape[0]
    reg = (gamma + 1e-3) * jnp.eye(n, dtype=jnp.float64) + gram
    return jnp.linalg.solve(reg, residuals).astype(jnp.float32)


def solve_C_f64(gram, residuals, gamma):
    """Cx. eigh f64, 1e-3 없음 — f64 버전"""
    gram, residuals = _to_f64(gram), _to_f64(residuals)
    n               = gram.shape[0]
    reg             = gamma * jnp.eye(n, dtype=jnp.float64) + gram
    eigs, vecs      = jnp.linalg.eigh(reg)
    safe_inv        = jnp.where(eigs > 1e-4 * eigs[-1], 1.0 / eigs, 0.0)
    return (vecs @ (safe_inv[:, None] * (vecs.T @ residuals))).astype(jnp.float32)


# f32 vs f64 비교용 솔버 쌍
SOLVERS_F32 = {"0_solve_f32": solve_0, "C_eigh_f32": solve_C}
SOLVERS_F64 = {"0_solve_f64": solve_0_f64, "C_eigh_f64": solve_C_f64}

# 전체 비교용 (섹션 1, 2)
SOLVERS = {"0_solve": solve_0, "A_chol+1e-3": solve_A, "B_eigh+1e-3": solve_B, "C_eigh(cur)": solve_C}


def gram_cond(gram, gamma):
    reg  = gamma * jnp.eye(gram.shape[0]) + gram
    eigs = jnp.linalg.eigvalsh(reg)
    return float(eigs[0]), float(eigs[-1] / jnp.clip(eigs[0], 1e-12))


# ── 섹션 1: seed 별 비교 ───────────────────────────────────────────────────

def section_seed(label, data_fn, seeds, gamma, batch_size, z_dim, obs_dim, action_dim, **kw):
    HDR = (f"  {'seed':>5}  {'λ_min':>9}  {'cond':>11}  "
           + "  ".join(f"{n:>12}  {'NaN':>3}" for n in SOLVERS))
    SEP = "  " + "-"*90
    print(f"\n  [{label}]  γ={gamma:.0e}")
    print(HDR); print(SEP)

    all_norms = {n: [] for n in SOLVERS}
    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        z, obs, aoh, res = data_fn(key, batch_size, z_dim, obs_dim, action_dim, **kw)
        gram = build_gram(z, obs, aoh, z_dim, obs_dim, action_dim)
        lmin, cond = gram_cond(gram, gamma)
        row = f"  {seed:>5}  {lmin:>9.5f}  {cond:>11.1f}"
        for name, fn in SOLVERS.items():
            sol     = fn(gram, res, gamma)
            norm    = float(jnp.linalg.norm(sol))
            has_nan = bool(jnp.any(jnp.isnan(sol)))
            all_norms[name].append(norm if not has_nan else float("nan"))
            row += f"  {norm:>12.4f}  {'NaN' if has_nan else 'OK ':>3}"
        print(row)

    print(f"\n  seed 간 std (작을수록 재현성 ↑):")
    for name, norms in all_norms.items():
        arr   = np.array(norms)
        valid = arr[~np.isnan(arr)]
        nans  = int(np.isnan(arr).sum())
        if len(valid):
            print(f"    {name}: mean={valid.mean():>10.3f}  std={valid.std():>8.3f}"
                  f"  max-min={valid.max()-valid.min():>8.3f}  NaN={nans}/{len(seeds)}")
        else:
            print(f"    {name}: 전부 NaN")


# ── 섹션 2: device 별 비교 ─────────────────────────────────────────────────

def _run_on_device(device, gram_np, res_np, gamma):
    """gram/residuals를 지정 device에 올려 세 solver 실행, numpy로 반환"""
    gram = jax.device_put(jnp.array(gram_np), device)
    res  = jax.device_put(jnp.array(res_np),  device)
    results = {}
    for name, fn in SOLVERS.items():
        sol = fn(gram, res, gamma)
        results[name] = np.array(jax.device_get(sol))
    return results


def section_device(label, data_fn, seeds, gamma, batch_size, z_dim, obs_dim, action_dim, **kw):
    """동일 seed에서 CPU / GPU 결과 차이 (절대 오차 norm)를 출력"""
    if not _HAS_GPU:
        print(f"\n  [{label}] GPU 없음 — device 비교 생략")
        return

    HDR = (f"  {'seed':>5}  {'cond':>11}  "
           + "  ".join(f"{'|CPU-GPU| '+n:>22}" for n in SOLVERS))
    SEP = "  " + "-"*110
    print(f"\n  [{label}]  γ={gamma:.0e}  (동일 seed, CPU vs GPU 절대 오차)")
    print(HDR); print(SEP)

    for seed in seeds:
        key = jax.random.PRNGKey(seed)
        # CPU에서 데이터 생성 후 numpy 변환 (device-neutral 기준점)
        with jax.default_device(_CPU):
            z, obs, aoh, res = data_fn(key, batch_size, z_dim, obs_dim, action_dim, **kw)
            gram = build_gram(z, obs, aoh, z_dim, obs_dim, action_dim)
        _, cond = gram_cond(gram, gamma)
        gram_np = np.array(jax.device_get(gram))
        res_np  = np.array(jax.device_get(res))

        cpu_out = _run_on_device(_CPU, gram_np, res_np, gamma)
        gpu_out = _run_on_device(_GPU, gram_np, res_np, gamma)

        row = f"  {seed:>5}  {cond:>11.1f}"
        for name in SOLVERS:
            diff = np.linalg.norm(cpu_out[name] - gpu_out[name])
            row += f"  {diff:>22.6f}"
        print(row)

    print(f"\n  해석: 0에 가까울수록 CPU/GPU 결과 일치 (재현성 ↑)")
    print(f"        조건수가 클수록 부동소수점 병렬 연산 순서 차이가 증폭됨")


# ── main ───────────────────────────────────────────────────────────────────

def main():
    print(f"\n디바이스: CPU={_CPU}  GPU={'없음' if not _HAS_GPU else _GPU}")

    seeds      = (0, 1, 2, 3, 42)
    batch_size = 256
    z_dim      = 128
    obs_dim    = 80
    action_dim = 9

    scenarios = [
        ("Random",   make_random_data,   {},                   "완전 랜덤"),
        ("SMAX",     make_smax_like_data, {"cluster_noise": 0.05},  "SMAX-like (noise=0.05)"),
        ("Extreme",  make_smax_like_data, {"cluster_noise": 0.001}, "극단 (noise=0.001)"),
    ]

    # ── 섹션 1: seed 별 비교 ──────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  [섹션 1] seed 별 solver norm 비교")
    print("=" * 90)
    for tag, fn, kw, desc in scenarios:
        print(f"\n{'─'*90}")
        print(f"  시나리오: {desc}")
        print(f"{'─'*90}")
        for gamma in (1e-4, 1e-2):
            section_seed(tag, fn, seeds, gamma, batch_size, z_dim, obs_dim, action_dim, **kw)

    # ── 섹션 2: device 별 비교 ────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  [섹션 2] CPU vs GPU 절대 오차 (동일 seed, 동일 solver)")
    print("  → 값이 0이면 완전 일치, 클수록 device 비결정성 문제")
    print("=" * 90)
    for tag, fn, kw, desc in scenarios:
        print(f"\n{'─'*90}")
        print(f"  시나리오: {desc}")
        print(f"{'─'*90}")
        for gamma in (1e-4, 1e-2):
            section_device(tag, fn, seeds, gamma, batch_size, z_dim, obs_dim, action_dim, **kw)

    # ── 섹션 3: f32 vs f64 비교 ───────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  [섹션 3] f32 vs f64 비교 (jax_enable_x64=True)")
    print("  0_solve / C_eigh 각각 f32·f64로 비교")
    print("  · seed 간 std: precision이 재현성에 미치는 영향")
    print("  · |f32-f64|: f32 수치 오차의 절대 크기 (f64를 기준값으로)")
    print("=" * 90)

    all_solvers_x64 = {**SOLVERS_F32, **SOLVERS_F64}
    solver_pairs = [("0_solve_f32", "0_solve_f64"), ("C_eigh_f32", "C_eigh_f64")]

    for tag, fn, kw, desc in scenarios:
        print(f"\n{'─'*90}")
        print(f"  시나리오: {desc}")
        print(f"{'─'*90}")
        for gamma in (1e-4, 1e-2):
            print(f"\n  γ={gamma:.0e}")
            hdr = (f"  {'seed':>5}  {'cond':>11}  "
                   + "  ".join(f"{n:>14}" for n in all_solvers_x64)
                   + "  " + "  ".join(f"{'|f32-f64| '+p[0][:7]:>20}" for p in solver_pairs))
            print(hdr)
            print("  " + "-" * 110)

            norms_by_solver = {n: [] for n in all_solvers_x64}
            for seed in seeds:
                key  = jax.random.PRNGKey(seed)
                z, obs, aoh, res = fn(key, batch_size, z_dim, obs_dim, action_dim, **kw)
                gram = build_gram(z, obs, aoh, z_dim, obs_dim, action_dim)
                _, cond = gram_cond(gram, gamma)

                solved_vals = {}
                row = f"  {seed:>5}  {cond:>11.1f}"
                for name, sfn in all_solvers_x64.items():
                    sol  = sfn(gram, res, gamma)
                    norm = float(jnp.linalg.norm(sol))
                    norms_by_solver[name].append(norm)
                    solved_vals[name] = np.array(jax.device_get(sol))
                    row += f"  {norm:>14.4f}"

                for f32_name, f64_name in solver_pairs:
                    diff = np.linalg.norm(solved_vals[f32_name] - solved_vals[f64_name])
                    row += f"  {diff:>20.6f}"
                print(row)

            print(f"\n  seed 간 std:")
            for name, norms in norms_by_solver.items():
                arr = np.array(norms)
                print(f"    {name}: mean={arr.mean():>10.3f}  std={arr.std():>8.3f}  max-min={arr.max()-arr.min():>8.3f}")

    print("\n" + "=" * 90)
    print("  해석 가이드")
    print("  섹션 1 — seed 간 std: 데이터 클러스터링에 따른 solver 민감도")
    print("  섹션 2 — CPU/GPU 오차: 부동소수점 병렬 연산 비결정성")
    print("  섹션 3 — |f32-f64|: f32 수치 오차의 절대 크기")
    print("    · |f32-f64| 가 크면 f32의 누적 오차가 결과를 왜곡 중")
    print("    · f64 std < f32 std 이면 precision 향상이 재현성에 직접 기여")
    print("=" * 90)


if __name__ == "__main__":
    main()
