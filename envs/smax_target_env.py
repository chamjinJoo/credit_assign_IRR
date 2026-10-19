from __future__ import annotations

from typing import Any, Dict

import numpy as np


def swap_to_target_aware_inner_smax(
    base_env,
    *,
    enemy_shoots: bool,
    attack_mode: str,
    env_kwargs: Dict[str, Any],
    allowed_target_matrix: np.ndarray,
    use_partial_damage_restore: bool,
):
    import jax
    import jax.numpy as jnp
    from functools import partial
    from jaxmarl.environments.smax.smax_env import SMAX
    from jaxmarl.environments.smax.heuristic_enemy import create_heuristic_policy

    class TargetAwareSMAX(SMAX):
        def __init__(
            self,
            allowed_target_matrix: np.ndarray,
            use_partial_damage_restore: bool = False,
            **kwargs,
        ) -> None:
            super().__init__(**kwargs)
            self.use_allowed_targets = True
            self.use_partial_damage_restore = bool(use_partial_damage_restore)
            self.allowed_target_matrix = jnp.asarray(allowed_target_matrix, dtype=jnp.float32)

        @partial(jax.jit, static_argnums=(0,))
        def _world_step(self, key, state, actions):
            def update_position(idx, vec):
                pos = state.unit_positions[idx]
                new_pos = (
                    pos
                    + vec
                    * self.unit_type_velocities[state.unit_types[idx]]
                    * self.time_per_step
                )
                new_pos = jnp.maximum(
                    jnp.minimum(new_pos, jnp.array([self.map_width, self.map_height])),
                    jnp.zeros((2,)),
                )
                return new_pos

            def update_agent_health(idx, action, key):
                attacked_idx = jax.lax.cond(
                    idx < self.num_allies,
                    lambda: action + self.num_allies - self.num_movement_actions,
                    lambda: self.num_allies - 1 - (action - self.num_movement_actions),
                )
                attacked_idx = jax.lax.select(
                    action < self.num_movement_actions,
                    idx,
                    attacked_idx,
                )
                attack_valid = (
                    (
                        jnp.linalg.norm(
                            state.unit_positions[idx] - state.unit_positions[attacked_idx]
                        )
                        < self.unit_type_attack_ranges[state.unit_types[idx]]
                    )
                    & state.unit_alive[idx]
                    & state.unit_alive[attacked_idx]
                )
                attack_valid = attack_valid & (idx != attacked_idx)
                attack_valid = attack_valid & (state.unit_weapon_cooldowns[idx] <= 0.0)
                health_diff = jax.lax.select(
                    attack_valid,
                    -self.unit_type_attacks[state.unit_types[idx]],
                    0.0,
                )
                cooldown_deviation = jax.random.uniform(
                    key,
                    minval=-self.time_per_step,
                    maxval=2 * self.time_per_step,
                )
                cooldown = (
                    self.unit_type_weapon_cooldowns[state.unit_types[idx]]
                    + cooldown_deviation
                )
                cooldown_diff = jax.lax.select(
                    attack_valid,
                    cooldown - state.unit_weapon_cooldowns[idx],
                    -self.time_per_step,
                )
                return health_diff, attacked_idx, cooldown_diff

            def perform_agent_action(idx, action, key):
                movement_action, attack_action = action
                new_pos = update_position(idx, movement_action)
                health_diff, attacked_idxes, cooldown_diff = update_agent_health(
                    idx,
                    attack_action,
                    key,
                )
                return new_pos, (health_diff, attacked_idxes), cooldown_diff

            keys = jax.random.split(key, num=self.num_agents)
            pos, (health_diff, attacked_idxes), cooldown_diff = jax.vmap(
                perform_agent_action
            )(jnp.arange(self.num_agents), actions, keys)

            ally_attack_actions = actions[1][: self.num_allies]
            ally_attack_mask = ally_attack_actions >= self.num_movement_actions
            ally_target_enemy = jnp.clip(
                ally_attack_actions - self.num_movement_actions,
                0,
                self.num_enemies - 1,
            )
            ally_attack_one_hot = jax.nn.one_hot(
                ally_target_enemy,
                self.num_enemies,
                dtype=jnp.float32,
            ) * ally_attack_mask[:, None].astype(jnp.float32)
            allowed_for_attack = self.allowed_target_matrix[
                jnp.arange(self.num_allies),
                ally_target_enemy,
            ]

            if self.use_partial_damage_restore:
                ally_keep_fraction = jnp.where(
                    ally_attack_mask,
                    allowed_for_attack,
                    1.0,
                )
            else:
                off_target_one_hot = ally_attack_one_hot * (1.0 - allowed_for_attack[:, None])
                off_target_attacks = off_target_one_hot.sum(axis=0)
                keep_fraction = jnp.where(off_target_attacks > 0.0, 0.0, 1.0)
                ally_keep_fraction = jnp.where(
                    ally_attack_mask,
                    keep_fraction[ally_target_enemy],
                    1.0,
                )
            health_diff = health_diff.at[: self.num_allies].set(
                health_diff[: self.num_allies] * ally_keep_fraction
            )

            dnums = jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            )
            unit_health = jnp.maximum(
                jax.lax.scatter_add(
                    state.unit_health,
                    jnp.expand_dims(attacked_idxes, 1),
                    health_diff,
                    dnums,
                ),
                0.0,
            )
            unit_weapon_cooldowns = state.unit_weapon_cooldowns + cooldown_diff
            state = state.replace(
                unit_health=unit_health,
                unit_positions=pos,
                unit_weapon_cooldowns=unit_weapon_cooldowns,
            )
            return state

    base_env._env = TargetAwareSMAX(
        allowed_target_matrix=allowed_target_matrix,
        use_partial_damage_restore=use_partial_damage_restore,
        **env_kwargs,
    )
    base_env.num_agents = base_env._env.num_allies
    base_env.num_enemies = base_env._env.num_enemies
    base_env.num_allies = base_env._env.num_allies
    base_env.observation_spaces = {
        agent: base_env._env.observation_spaces[agent] for agent in base_env.agents
    }
    base_env.action_spaces = {
        agent: base_env._env.action_spaces[agent] for agent in base_env.agents
    }
    base_env.heuristic_policy = create_heuristic_policy(
        base_env._env,
        1,
        shoot=enemy_shoots,
        attack_mode=attack_mode,
    )
    return base_env
