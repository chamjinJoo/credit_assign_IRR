#!/usr/bin/env python3
"""Shape-matched JAX solve microbenchmark for PCI bridge cost reports."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-csv", required=True, help="bridge_cost_seed_pairs_*.csv from the timing report")
    parser.add_argument("--output", required=True, help="CSV path to write solve benchmark rows")
    parser.add_argument("--env", action="append", default=["smax", "lbf"], help="Environment key to benchmark; repeatable")
    parser.add_argument("--rhs-dim", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


@jax.jit
def solve_blocks(reg_gram, residuals):
    return jax.vmap(lambda reg_n, res_n: jnp.linalg.solve(reg_n, res_n))(reg_gram, residuals)


def make_spd(key, num_agents: int, batch_size: int, rhs_dim: int):
    mat_key, res_key = jax.random.split(key)
    raw = jax.random.normal(mat_key, (num_agents, batch_size, batch_size), dtype=jnp.float32)
    eye = jnp.eye(batch_size, dtype=jnp.float32)[None, :, :]
    reg = (raw @ jnp.swapaxes(raw, -1, -2)) / float(batch_size)
    reg = reg + 1.0e-2 * eye
    residuals = jax.random.normal(res_key, (num_agents, batch_size, rhs_dim), dtype=jnp.float32)
    return reg, residuals


def benchmark_one(key, num_agents: int, batch_size: int, rhs_dim: int, repeats: int, warmup: int):
    reg, residuals = make_spd(key, num_agents, batch_size, rhs_dim)
    jax.block_until_ready((reg, residuals))
    for _ in range(warmup):
        solved = solve_blocks(reg, residuals)
        solved.block_until_ready()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        solved = solve_blocks(reg, residuals)
        solved.block_until_ready()
        times.append(time.perf_counter() - start)
    return sum(times) / len(times), min(times), max(times)


def main() -> int:
    args = parse_args()
    include_envs = set(args.env)
    rows = []
    with Path(args.pair_csv).open(newline="") as f:
        for row in csv.DictReader(f):
            if row.get("env") not in include_envs:
                continue
            batch_size = int(round(float(row["gram_batch_size"])))
            num_agents = int(float(row["num_agents"]))
            chunks = float(row.get("causal_num_minibatches") or row.get("causal_num_minibatches_mean") or 1.0)
            if "causal_num_minibatches" not in row:
                chunks = 1.0
            bridge_epoch = int(float(row.get("bridge_epoch") or 1))
            key = jax.random.fold_in(jax.random.PRNGKey(args.seed), len(rows))
            per_chunk_mean, per_chunk_min, per_chunk_max = benchmark_one(
                key, num_agents, batch_size, args.rhs_dim, args.repeats, args.warmup
            )
            rows.append({
                "env": row["env"],
                "backbone": row["backbone"],
                "seed": row.get("seed", ""),
                "source": "jax_shape_microbenchmark",
                "num_agents": num_agents,
                "gram_batch_size": batch_size,
                "rhs_dim": args.rhs_dim,
                "causal_num_minibatches": chunks,
                "bridge_epoch": bridge_epoch,
                "repeats": args.repeats,
                "warmup": args.warmup,
                "solve_time_per_chunk_sec_mean": per_chunk_mean,
                "solve_time_per_chunk_sec_min": per_chunk_min,
                "solve_time_per_chunk_sec_max": per_chunk_max,
                "bridge_solve_time_sec": per_chunk_mean * chunks * bridge_epoch,
                "jax_backend": jax.default_backend(),
                "jax_devices": str(jax.devices()),
            })
    fieldnames = [
        "env", "backbone", "seed", "source", "num_agents", "gram_batch_size", "rhs_dim",
        "causal_num_minibatches", "bridge_epoch", "repeats", "warmup",
        "solve_time_per_chunk_sec_mean", "solve_time_per_chunk_sec_min", "solve_time_per_chunk_sec_max",
        "bridge_solve_time_sec", "jax_backend", "jax_devices",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
