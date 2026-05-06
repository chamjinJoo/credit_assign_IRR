"""
Analysis 1: Bridge Moment Restriction Residual
===============================================
Empirically validates Proposition 1 / Theorem 1.

The learned bridge h_φ should satisfy the observed moment restriction:
    E[R - h_φ(W, A, X) | Z, A, X] ≈ 0

We check this by collecting rollout data from trained checkpoints and
plotting the residual distribution R - h_φ(W, A, X) per scenario type,
comparing against a randomly-initialised bridge of the same architecture.

Usage:
    cd /home/chamjin/research/pci_marl
    python SH_analysis/bridge_residual_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from SH_analysis.sh_analysis_utils import (
    get_checkpoint_paths,
    load_bridge_and_agents,
    make_random_bridge,
    make_env,
    collect_rollout_data,
    bridge_residuals,
    SCENARIO_ORDER,
    ACTION_NAMES,
)

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

# ── Config ────────────────────────────────────────────────────────────────────

N_EPISODES_PER_SEED = 200
MAX_SEEDS           = 5
DEVICE              = "cpu"
SAVE_DIR            = os.path.dirname(__file__)

VARIANTS = [
    # (display_name, variant, model_type, obs_noise_std)
    ("PCI-IPPO\n(No Noise)",   "sh3",       "ipci", 0.0),
    ("PCI-MAPPO\n(No Noise)",  "sh3",       "mpci", 0.0),
    ("PCI-IPPO\n(Noisy Obs)",  "sh3_noise", "ipci", 0.1),
    ("PCI-MAPPO\n(Noisy Obs)", "sh3_noise", "mpci", 0.1),
]

SCENARIO_COLORS = {
    "Stag": "#1565C0",
    "Hare": "#E65100",
    "Null": "#424242",
}
RANDOM_ALPHA = 0.35


# ── Data collection ───────────────────────────────────────────────────────────

def collect_variant_data(variant: str, model_type: str, obs_noise_std: float):
    """Return (records, trained_bridge, random_bridge) for one variant."""
    env          = make_env(obs_noise_std=obs_noise_std)
    ckpt_paths   = get_checkpoint_paths(variant, model_type)[:MAX_SEEDS]

    if not ckpt_paths:
        print(f"[WARN] No checkpoints found for {variant}/{model_type}")
        return None, None, None

    all_records   = []
    trained_bridge = None

    for path in ckpt_paths:
        print(f"  Loading {path}")
        bnet, agents = load_bridge_and_agents(path, DEVICE)
        if trained_bridge is None:
            trained_bridge = bnet
        records = collect_rollout_data(agents, env, n_episodes=N_EPISODES_PER_SEED)
        all_records.extend(records)

    random_bridge = make_random_bridge()
    print(f"  Collected {len(all_records)} transitions ({len(ckpt_paths)} seeds)")
    return all_records, trained_bridge, random_bridge


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_residual_violin(ax, records, trained_bridge, random_bridge, title: str):
    """
    Violin plot of residuals per scenario for trained and random bridges.
    Trained = solid; random = hatched.
    """
    scenarios = np.array([r["scenario"] for r in records])
    res_tr    = bridge_residuals(trained_bridge, records, DEVICE)
    res_ra    = bridge_residuals(random_bridge,  records, DEVICE)

    positions    = np.array([0.0, 1.5, 3.0])
    offset       = 0.28
    _annotations = []  # collected during loop, placed after ylim is finalized

    for idx, sc in enumerate(SCENARIO_ORDER):
        mask = scenarios == sc
        n    = mask.sum()
        if n == 0:
            continue
        color = SCENARIO_COLORS[sc]
        x_tr  = positions[idx] - offset
        x_ra  = positions[idx] + offset

        # Trained bridge violin
        vp_tr = ax.violinplot([res_tr[mask]], positions=[x_tr], widths=0.45,
                              showmedians=True, showextrema=False)
        for body in vp_tr["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.85)
        vp_tr["cmedians"].set_color("white")
        vp_tr["cmedians"].set_linewidth(2)

        # Random bridge violin (lighter + dashed outline)
        vp_ra = ax.violinplot([res_ra[mask]], positions=[x_ra], widths=0.45,
                              showmedians=True, showextrema=False)
        for body in vp_ra["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(RANDOM_ALPHA)
        vp_ra["cmedians"].set_color("black")
        vp_ra["cmedians"].set_linewidth(1.5)
        vp_ra["cmedians"].set_linestyle("--")

        # Annotate mean ± std for trained (stored for post-loop placement)
        mean_tr = res_tr[mask].mean()
        std_tr  = res_tr[mask].std()
        # Store annotation info; placed after all violins to use final ylim
        _annotations.append((x_tr, mean_tr, std_tr, color))

    ax.axhline(0, color="black", linewidth=1.5, linestyle="--", alpha=0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(SCENARIO_ORDER)
    ax.set_title(title)
    ax.set_ylabel(r"Residual  $R - \hat{h}_\varphi(W, A, X)$")
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    # Place annotations now that ylim is finalized
    y_top = ax.get_ylim()[1]
    for x_tr, mean_tr, std_tr, color in _annotations:
        ax.text(x_tr, y_top * 0.97,
                f"μ={mean_tr:.2f}\nσ={std_tr:.2f}",
                ha="center", va="top", fontsize=8.5, color=color)


def plot_mean_abs_residual(ax, all_variant_data):
    """
    Bar chart: mean absolute residual across scenarios for each variant.
    all_variant_data: list of (display_name, records, trained_bridge, random_bridge)
    """
    variant_names = [v[0].replace("\n", " ") for v in all_variant_data]
    n_variants    = len(variant_names)
    x             = np.arange(n_variants)
    width         = 0.35

    means_tr, stds_tr = [], []
    means_ra, stds_ra = [], []

    for display, records, trained, random in all_variant_data:
        res_tr = bridge_residuals(trained, records, DEVICE)
        res_ra = bridge_residuals(random,  records, DEVICE)
        means_tr.append(np.abs(res_tr).mean())
        stds_tr.append(np.abs(res_tr).std() / np.sqrt(len(res_tr)))
        means_ra.append(np.abs(res_ra).mean())
        stds_ra.append(np.abs(res_ra).std() / np.sqrt(len(res_ra)))

    ax.bar(x - width / 2, means_tr, width, yerr=stds_tr,
           label="Trained Bridge", color="#37474F", alpha=0.85,
           error_kw=dict(elinewidth=1.5))
    ax.bar(x + width / 2, means_ra, width, yerr=stds_ra,
           label="Random Bridge", color="#B0BEC5", alpha=0.85,
           error_kw=dict(elinewidth=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names, fontsize=10)
    ax.set_ylabel("Mean  |R − ĥ|  (↓ better)")
    ax.set_title("Mean Absolute Residual: Trained vs. Random Bridge")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")


def print_statistics(variant_name: str, records, trained_bridge, random_bridge):
    scenarios = np.array([r["scenario"] for r in records])
    res_tr    = bridge_residuals(trained_bridge, records, DEVICE)
    res_ra    = bridge_residuals(random_bridge,  records, DEVICE)

    print(f"\n{'='*60}")
    print(f"  {variant_name}")
    print(f"{'='*60}")
    print(f"  {'Scenario':8s}  {'N':>7s}  {'Trained μ':>10s}  {'Trained σ':>10s}  "
          f"{'Random μ':>10s}  {'Random σ':>10s}  {'|Δμ|/σ_r':>9s}")
    print(f"  {'-'*80}")
    for sc in SCENARIO_ORDER:
        mask = scenarios == sc
        n    = mask.sum()
        if n == 0:
            continue
        mu_tr = res_tr[mask].mean()
        sd_tr = res_tr[mask].std()
        mu_ra = res_ra[mask].mean()
        sd_ra = res_ra[mask].std()
        ratio = abs(mu_tr) / (sd_ra + 1e-8)
        print(f"  {sc:8s}  {n:7d}  {mu_tr:+10.3f}  {sd_tr:10.3f}  "
              f"{mu_ra:+10.3f}  {sd_ra:10.3f}  {ratio:9.3f}")

    # Overall t-test: trained residuals vs 0
    t_stat, p_val = stats.ttest_1samp(res_tr, 0.0)
    print(f"\n  Overall trained residuals vs 0:  t={t_stat:.3f}, p={p_val:.4f}")
    print(f"  Overall trained MAE = {np.abs(res_tr).mean():.4f}")
    print(f"  Overall random  MAE = {np.abs(res_ra).mean():.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Collecting rollout data ...")
    variant_data = []
    for display, variant, model_type, noise in VARIANTS:
        print(f"\n[{display.replace(chr(10), ' ')}]")
        records, trained, random = collect_variant_data(variant, model_type, noise)
        if records is None:
            continue
        variant_data.append((display, records, trained, random))

    if not variant_data:
        print("No data collected. Check checkpoint paths.")
        return

    # ── Figure 1: per-variant violin plots ──────────────────────────────────
    n = len(variant_data)
    fig1, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, (display, records, trained, random) in zip(axes, variant_data):
        plot_residual_violin(ax, records, trained, random, title=display)

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor="#455A64", alpha=0.85, label="Trained bridge"),
        mpatches.Patch(facecolor="#B0BEC5", alpha=0.85, label="Random bridge (baseline)"),
    ]
    fig1.legend(handles=legend_handles, loc="upper center", ncol=2, fontsize=11,
                bbox_to_anchor=(0.5, 1.01), framealpha=0.9)
    fig1.suptitle(
        r"Bridge Moment Restriction Residual  $R - \hat{h}_\varphi(W, A, X)$"
        "\n(near-zero ⟹ moment condition satisfied)",
        fontsize=13, y=1.07,
    )
    plt.tight_layout()
    out1 = os.path.join(SAVE_DIR, "bridge_residual_violin.pdf")
    fig1.savefig(out1, dpi=180, bbox_inches="tight")
    print(f"\nSaved: {out1}")

    # ── Figure 2: MAE bar chart ──────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(max(4, 2 * n), 4))
    plot_mean_abs_residual(ax2, variant_data)  # pass full (display, records, trained, random)
    plt.tight_layout()
    out2 = os.path.join(SAVE_DIR, "bridge_residual_mae.pdf")
    fig2.savefig(out2, dpi=180, bbox_inches="tight")
    print(f"Saved: {out2}")

    # ── Console statistics ───────────────────────────────────────────────────
    for display, records, trained, random in variant_data:
        print_statistics(display.replace("\n", " "), records, trained, random)

    plt.show()


if __name__ == "__main__":
    main()
