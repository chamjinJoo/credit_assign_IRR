from __future__ import annotations

import math
from typing import Any, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training import train_state

VERY_NEGATIVE = -1.0e9

# ------- Network Initialization follows the convention. -------
_ORTH_RELU = nn.initializers.orthogonal(scale=math.sqrt(2)) # hidden Dense layers
_ORTH_UNIT = nn.initializers.orthogonal(scale=1.0)          # GRU kernels, critic/bridge output layers
_ORTH_POLICY = nn.initializers.orthogonal(scale=0.01)       # actor output logits
# ---------------------------------------------------------------

def make_gru_cell(hidden_dim: int) -> nn.GRUCell:
    # Torch RNNLayer applies nn.init.orthogonal_ to all GRU weight matrices.
    return nn.GRUCell(
        features=hidden_dim,
        kernel_init=_ORTH_UNIT,
        recurrent_kernel_init=_ORTH_UNIT,
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
        # Standard orthogonal init (on-policy convention):
        #   Dense(orth,√2) → ReLU → LayerNorm → GRU(orth,1) → LayerNorm → Dense(orth,0.01)
        x = jnp.concatenate([obs, prev_actions], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        hidden = hidden * masks[..., None]
        gru = make_gru_cell(self.hidden_dim)
        new_hidden, core_out = gru(hidden, x)
        core_out = nn.LayerNorm()(core_out)

        logits = nn.Dense(self.action_dim, kernel_init=_ORTH_POLICY, bias_init=nn.initializers.zeros)(core_out)
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
        # Standard orthogonal init (on-policy convention):
        #   Dense(orth,√2) → ReLU → LayerNorm → GRU(orth,1) → LayerNorm → Dense(orth,1)
        x = jnp.concatenate([obs, prev_actions], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)

        hidden = hidden * masks[..., None]
        gru = make_gru_cell(self.hidden_dim)
        new_hidden, core_out = gru(hidden, x)
        core_out = nn.LayerNorm()(core_out)

        values = nn.Dense(self.output_dim, kernel_init=_ORTH_UNIT, bias_init=nn.initializers.zeros)(core_out)
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
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(obs)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.action_dim, kernel_init=_ORTH_POLICY, bias_init=nn.initializers.zeros)(x)
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
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(obs)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        values = nn.Dense(self.output_dim, kernel_init=_ORTH_UNIT, bias_init=nn.initializers.zeros)(x)
        if self.output_dim == 1:
            values = values.squeeze(-1)
        return hidden, values


class DeltaCritic(nn.Module):
    """Per-agent feedforward value network for delta (PCI) returns.

    Input : concatenation of (obs^i_t, actor_hidden^i_t).
            actor_hidden already encodes the temporal history when using a
            recurrent policy, so no extra GRU is needed here.
    Output: scalar V^i_delta — estimated value of the delta return stream.
    """

    hidden_dim: int
    input_dim: int  # policy_obs_dim + actor_state_dim

    @nn.compact
    def __call__(self, obs: jnp.ndarray, actor_hidden: jnp.ndarray) -> jnp.ndarray:
        if actor_hidden.shape[-1] > 0:
            x = jnp.concatenate([obs, actor_hidden], axis=-1)
        else:
            x = obs
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        
        return nn.Dense(1, kernel_init=_ORTH_UNIT, bias_init=nn.initializers.zeros)(x).squeeze(-1) # default

class CausalBridgeNetwork(nn.Module):
    hidden_dim: int
    action_dim: int
    num_agents: int

    @nn.compact
    def __call__(
        self,
        local_obs: jnp.ndarray,
        others_obs: jnp.ndarray,
        bridge_context: jnp.ndarray,
        actions: jnp.ndarray,
    ) -> jnp.ndarray:
        # Matches Torch OutcomeBridgeNetwork (MLPBase, use_orthogonal=True,
        # use_feature_normalization=True, use_layer_norm=True):
        #   LayerNorm(input) → [Dense(orth,√2) → ReLU → LayerNorm] × 2 → Dense(output)
        action_onehot = jax.nn.one_hot(actions.astype(jnp.int32), self.action_dim)
        x = jnp.concatenate([local_obs, others_obs, bridge_context, action_onehot], axis=-1)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dense(self.hidden_dim, kernel_init=_ORTH_RELU, bias_init=nn.initializers.zeros)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        return nn.Dense(
            self.num_agents, 
            kernel_init=nn.initializers.orthogonal(1e-2), 
            bias_init=nn.initializers.zeros
        )(x)
        # return nn.Dense(self.num_agents, kernel_init=_ORTH_UNIT)(x)


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


class DynamicLearningRateTrainState(train_state.TrainState):
    learning_rate: jnp.ndarray

    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        updates = jax.tree_util.tree_map(lambda update: update * self.learning_rate, updates)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


def make_dynamic_lr_train_state(
    module: nn.Module,
    params_key: jax.Array,
    sample_inputs: tuple[jnp.ndarray, ...],
    learning_rate: float,
    max_grad_norm: float,
) -> DynamicLearningRateTrainState:
    params = module.init(params_key, *sample_inputs)["params"]
    tx = optax.chain(
        optax.clip_by_global_norm(max_grad_norm),
        optax.adam(1.0),
    )
    return DynamicLearningRateTrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=tx,
        learning_rate=jnp.asarray(learning_rate, dtype=jnp.float32),
    )


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
