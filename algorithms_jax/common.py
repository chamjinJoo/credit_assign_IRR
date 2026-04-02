from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

VERY_NEGATIVE = -1.0e9

def make_gru_cell(hidden_dim: int) -> nn.GRUCell:
    return nn.GRUCell(
        features=hidden_dim,
        kernel_init=nn.initializers.xavier_uniform(),
        recurrent_kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.zeros_init(),
    )

class RecurrentActor(nn.Module):
    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_actions: jnp.ndarray,
        hidden: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([obs, prev_actions], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        hidden = hidden * masks[..., None]
        gru = make_gru_cell(self.hidden_dim)
        new_hidden, core_out = gru(hidden, x)
        logits = nn.Dense(self.action_dim)(core_out)
        return new_hidden, logits


class RecurrentCritic(nn.Module):
    hidden_dim: int
    output_dim: int = 1

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_actions: jnp.ndarray,
        hidden: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = jnp.concatenate([obs, prev_actions], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)

        hidden = hidden * masks[..., None]
        gru = make_gru_cell(self.hidden_dim)
        new_hidden, core_out = gru(hidden, x)
        values = nn.Dense(self.output_dim)(core_out)
        if self.output_dim == 1:
            values = values.squeeze(-1)
        return new_hidden, values


class FeedForwardActor(nn.Module):
    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_actions: jnp.ndarray,
        hidden: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del prev_actions, masks
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        logits = nn.Dense(self.action_dim)(x)
        return hidden, logits


class FeedForwardCritic(nn.Module):
    hidden_dim: int
    output_dim: int = 1

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_actions: jnp.ndarray,
        hidden: jnp.ndarray,
        masks: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        del prev_actions, masks
        x = nn.Dense(self.hidden_dim)(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.tanh(x)
        values = nn.Dense(self.output_dim)(x)
        if self.output_dim == 1:
            values = values.squeeze(-1)
        return hidden, values


class CausalBridgeNetwork(nn.Module):
    hidden_dim: int
    action_dim: int
    num_agents: int

    @nn.compact
    def __call__(
        self,
        local_obs: jnp.ndarray,
        others_obs: jnp.ndarray,
        hidden_others: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        action_onehot = jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim)
        x = jnp.concatenate([local_obs, others_obs, hidden_others, action_onehot], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_agents)(x)


def make_train_state(
    module: nn.Module,
    params_key: jax.Array,
    sample_inputs: tuple[jnp.ndarray, ...],
    learning_rate: float,
    max_grad_norm: float,
) -> train_state.TrainState:
    params = module.init(params_key, *sample_inputs)["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(learning_rate),
    )
    return train_state.TrainState.create(apply_fn=module.apply, params=params, tx=tx)


def mask_logits(logits: jnp.ndarray, available_actions: jnp.ndarray) -> jnp.ndarray:
    valid = available_actions > 0
    masked_logits = jnp.where(valid, logits, VERY_NEGATIVE)
    fallback = jnp.zeros_like(masked_logits)
    return jnp.where(jnp.all(~valid, axis=-1, keepdims=True), fallback, masked_logits)


def categorical_stats(
    logits: jnp.ndarray,
    available_actions: jnp.ndarray,
    rng: jax.Array,
    deterministic: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    masked_logits = mask_logits(logits, available_actions)
    probs = jax.nn.softmax(masked_logits, axis=-1)

    def _greedy(_: None) -> jnp.ndarray:
        return jnp.argmax(masked_logits, axis=-1)

    def _sample(_: None) -> jnp.ndarray:
        return jax.random.categorical(rng, masked_logits, axis=-1)

    actions = jax.lax.cond(deterministic, _greedy, _sample, operand=None)
    selected = jnp.take_along_axis(probs, actions[..., None], axis=-1).squeeze(-1)
    log_probs = jnp.log(jnp.clip(selected, a_min=1.0e-8))
    return actions.astype(jnp.int32), log_probs, probs


def add_agent_id_features(obs: jnp.ndarray) -> jnp.ndarray:
    num_agents = obs.shape[-2]
    eye = jnp.eye(num_agents, dtype=obs.dtype)
    leading = (1,) * (obs.ndim - 2)
    agent_id = jnp.reshape(eye, leading + eye.shape)
    agent_id = jnp.broadcast_to(agent_id, obs.shape[:-2] + eye.shape)
    return jnp.concatenate([obs, agent_id], axis=-1)


def gather_other_agent_obs(obs: jnp.ndarray) -> jnp.ndarray:
    num_agents = obs.shape[-2]
    others = []
    for agent_id in range(num_agents):
        indices = [idx for idx in range(num_agents) if idx != agent_id]
        other_obs = obs[..., indices, :].reshape(obs.shape[:-2] + (-1,))
        others.append(other_obs)
    return jnp.stack(others, axis=-2)


def action_one_hot(actions: jnp.ndarray, action_dim: int) -> jnp.ndarray:
    return jax.nn.one_hot(actions.astype(jnp.int32), action_dim, dtype=jnp.float32)


def global_norm(tree: Any) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(tree)
    sq_norm = sum(jnp.sum(jnp.square(x)) for x in leaves)
    return jnp.sqrt(sq_norm + 1.0e-8)


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    last_value: jnp.ndarray,
    dones: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dones = dones.astype(jnp.float32)
    discount = gamma * gae_lambda

    def scan_fn(carry, xs):
        gae, next_value = carry
        reward_t, value_t, done_t = xs
        done_mask = 1.0 - done_t
        while done_mask.ndim < reward_t.ndim:
            done_mask = done_mask[..., None]
        delta = reward_t + gamma * next_value * done_mask - value_t
        gae = delta + discount * done_mask * gae
        return (gae, value_t), (gae, gae + value_t)

    init = (jnp.zeros_like(last_value), last_value)
    (_, _), (advantages, returns) = jax.lax.scan(
        scan_fn,
        init,
        (rewards, values, dones),
        reverse=True,
    )
    return advantages, returns


def tree_to_host(tree: Any) -> Any:
    return jax.device_get(tree)


def stack_metrics(metrics: list[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    keys = set().union(*(metric.keys() for metric in metrics))
    return {
        key: float(sum(metric.get(key, 0.0) for metric in metrics) / len(metrics))
        for key in keys
    }
