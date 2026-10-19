from typing import Any, Dict, List, Optional, Tuple

import copy
import matplotlib.pyplot as plt
import numpy as np

from .base import MultiAgentEnv


MATPLOTLIB_AVAILABLE = True

class StagHuntEnv(MultiAgentEnv):
    """
    Reference: https://github.com/NullDefault/Gym-Stag-Hunt/tree/master
    """
    # Action constants
    STAY = 0
    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4
    
    def __init__(
        self, 
        num_agents: int = 3,
        max_steps: int = 50,
        stag_follows: bool = False,
        run_away_after_maul: bool = False,
        forage_quantity: int = None,
        global_reward: bool = False,
        forage_reward: float = 3.0,
        stag_reward: float = 5.0,
        mauling_punishment: float = 0.0,
        coop_bonus: float = 0.0,
        required_agents: Dict[int, List[int]] = None,
        seed: int = None,
        grid_observation: bool = False,
        vision: int = 2,
        obs_noise_std: float = 0.0,
        **kwargs):

        super().__init__(num_agents=num_agents, seed=seed)

        self.grid_observation = grid_observation
        self.vision = vision

        self.max_steps = max_steps
        self.stag_follows = stag_follows
        self.run_away_after_maul = run_away_after_maul

        self.forage_quantity = forage_quantity if forage_quantity is not None else num_agents
        self.step_count = 0
        
        # Grid size (7x7)
        self.grid_size_x = 7
        self.grid_size_y = 7
        
        # Action space: 0=stay, 1=up, 2=down, 3=left, 4=right
        self.action_dim = 5
        
        # Observation space: coordinate(2) + 5x5 grid
        if self.grid_observation:
            # self.obs_dim = (2 * self.vision + 1)**2 * 3 # bitmask obs
            self.obs_dim = (2 * self.vision + 1)**2 * (2*num_agents + 1) 
        else:
            # num_agents + num_stags + num_hares. All have (x, y) coordinates
            self.obs_dim = (num_agents + num_agents + self.forage_quantity) * 3
            
        # add noise to observation
        self.obs_noise_std = obs_noise_std

        # Environment state
        self.agent_positions = {}  # {agent_id: (x, y)}
        self.stag_positions = {}  # {stag_id: (x, y)}
        self.stag_alive = {}  # {stag_id: bool}
        self.hare_positions = {}  # {hare_id: (x, y)} for individual hunting
        self.hare_alive = {}  # {hare_id: bool}
              
        # Rewards
        self.global_reward = global_reward
        self.forage_reward = forage_reward
        self.stag_reward = stag_reward
        self.mauling_punishment = mauling_punishment
        self.coop_bonus = coop_bonus

        # Allocate agents to each stag: stag_i -> (agent i, agent i+1), with wrap-around for the last stag.
        if required_agents is not None:
            self.required_agents = required_agents
        else:
            self.required_agents = {}
            for stag_id in range(num_agents):
                if stag_id == num_agents - 1:  # Last stag wraps to agent 0.
                    self.required_agents[stag_id] = [stag_id, 0]
                else:
                    self.required_agents[stag_id] = [stag_id, stag_id + 1]
    
    def print_env_setting(self):
        print(f"mauling_punishment: {self.mauling_punishment}, coop_bonus: {self.coop_bonus}")
        print(f"stag allocated agents: {self.required_agents}")
            
    def get_env_state(self):
        return copy.deepcopy(self.__dict__)

    def set_env_state(self, state):
        self.__dict__.update(copy.deepcopy(state))

    def _get_empty_positions(self):
        occupied_positions = set()
        
        for agent_id in self.agents:
            if agent_id in self.agent_positions:
                occupied_positions.add(self.agent_positions[agent_id])
        for stag_pos in self.stag_positions.values():
            occupied_positions.add(stag_pos)
        for hare_pos in self.hare_positions.values():
            occupied_positions.add(hare_pos)

        empty_positions = []
        for x in range(self.grid_size_x):
            for y in range(self.grid_size_y):
                if (x, y) not in occupied_positions:
                    empty_positions.append((x, y))
        return empty_positions

    def _spawn_object_randomly(self):
        empty_positions = self._get_empty_positions()
        if empty_positions:
            if self._np_random is not None:
                return empty_positions[self._np_random.randint(len(empty_positions))]
            else:
                return empty_positions[np.random.randint(len(empty_positions))]
        else:
            raise ValueError("No empty positions available for spawning object")
    
    def _spawn_stags_with_separation(self, num_stags):
        stag_positions = []
        empty_positions = self._get_empty_positions()
        
        if len(empty_positions) < num_stags:
            raise ValueError(f"Not enough empty positions for {num_stags} stags")
        
        if self._np_random is not None:
            first_stag_pos = empty_positions[self._np_random.randint(len(empty_positions))]
        else:
            first_stag_pos = empty_positions[np.random.randint(len(empty_positions))]
        
        stag_positions.append(first_stag_pos)
        available_positions = [pos for pos in empty_positions if pos != first_stag_pos]
        
        for _ in range(num_stags - 1):
            valid_positions = []
            
            for pos in available_positions:
                is_valid = True
                for existing_pos in stag_positions:
                    x_diff = abs(pos[0] - existing_pos[0])
                    y_diff = abs(pos[1] - existing_pos[1])
                    if x_diff + y_diff == 1:
                        is_valid = False
                        break
                
                if is_valid:
                    valid_positions.append(pos)
            
            if not valid_positions:
                raise ValueError("No valid positions available for stag separation")
            
            if self._np_random is not None:
                selected_pos = valid_positions[self._np_random.randint(len(valid_positions))]
            else:
                selected_pos = valid_positions[np.random.randint(len(valid_positions))]
            stag_positions.append(selected_pos)
            available_positions.remove(selected_pos)
        
        return stag_positions

    def reset(self) -> Dict[int, np.ndarray]:
        self.step_count = 0
        self.stag_positions = {}
        self.stag_alive = {}
   
        for i, agent_id in enumerate(self.agents):
            self.agent_positions[agent_id] = self._spawn_object_randomly() # random spawn
        
        stag_positions = self._spawn_stags_with_separation(self.num_agents)
        for i, stag_id in enumerate(self.agents):
            self.stag_positions[stag_id] = stag_positions[i]
            self.stag_alive[stag_id] = True
    
        try:
            self.hare_positions = {}
            self.hare_alive = {}
            for hare_id in range(self.forage_quantity):
                hare_pos = self._spawn_object_randomly()
                self.hare_positions[hare_id] = hare_pos
                self.hare_alive[hare_id] = True 
        except ValueError as e:
            print(f"Warning during reset: {e}")
            self.hare_positions = {0: (1, 1), 1: (3, 3)}
            self.hare_alive = {0: True, 1: True}
        
        return self._get_observations()
    
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        self.step_count += 1
        self._process_movement_actions(actions)
        
        huntable_objects_pre = self._get_huntable_objects()
        
        # Process hunting and rewards.
        rewards = {agent_id: 0.0 for agent_id in self.agents}
        hunting_info = self._process_hunting(rewards)
        
        if self.global_reward: # all agents share the same reward
            g_reward = sum(rewards.values())
            for agent_id in self.agents:
                rewards[agent_id] = g_reward

        alive_stags = sum(1 for alive in self.stag_alive.values() if alive)
        alive_hares = sum(1 for alive in self.hare_alive.values() if alive)
        done = (self.step_count >= self.max_steps or 
                (alive_stags == 0 and alive_hares == 0))
        
        observations = self._get_observations()
        
        huntable_objects = huntable_objects_pre
        
        # Info
        info = {
            'step_count': self.step_count,
            'agent_positions': self.agent_positions.copy(),
            'stag_positions': self.stag_positions.copy(),
            'stag_alive': self.stag_alive.copy(),
            'hare_positions': self.hare_positions.copy(),
            'hare_alive': self.hare_alive.copy(),
            'stag_hunts': hunting_info['stag_hunts'],  # {stag_id: [participating_agent_ids]}
            'stag_attempts': hunting_info['stag_attempts'],  # {stag_id: [agents who attempted to hunt]}
            'hare_hunts': hunting_info['hare_hunts'],  # [agent_ids who hunted hares]
            'hare_attempts': hunting_info['hare_attempts'],  # [agent_ids who attempted to hunt hares]
            'huntable_objects': huntable_objects
        }
        dones = {agent_id: done for agent_id in self.agents}
            
        return observations, rewards, dones, info
    
    def _get_huntable_objects(self) -> Dict[str, Any]:
        """       
        Returns:
            Dict containing huntable stags and hares with their requirements
            - huntable_stags:
            - other_stags_by_agent:
            - huntable_hares:
        """
        huntable_stags = {}
        huntable_hares = []

        other_stags_by_agent = {agent_id: {} for agent_id in self.agents}
        
        for stag_id, (stag_x, stag_y) in self.stag_positions.items():
            if not self.stag_alive[stag_id]:
                continue
                

            required_agents = self.required_agents[stag_id]
            huntable_positions = [
                (stag_x, stag_y),
                (stag_x, stag_y + 1), 
                (stag_x, stag_y - 1),
                (stag_x + 1, stag_y),
                (stag_x - 1, stag_y)
            ]
            
            if all(agent_id in self.agent_positions for agent_id in required_agents):
                agent_positions = {agent_id: self.agent_positions[agent_id] for agent_id in required_agents}
                
                all_agents_can_hunt = all(
                    agent_pos in huntable_positions 
                    for agent_pos in agent_positions.values()
                )
                
                if all_agents_can_hunt:
                    huntable_stags[stag_id] = {
                        'position': (stag_x, stag_y),
                        'required_agents': required_agents,
                        'agent_positions': agent_positions,
                        'huntable_positions': huntable_positions
                    }
            
            for agent_id in self.agents:
                if agent_id not in required_agents:
                    agent_pos = self.agent_positions.get(agent_id)
                    if agent_pos and agent_pos in huntable_positions:
                        other_stags_by_agent[agent_id][stag_id] = {
                            'position': (stag_x, stag_y),
                            'required_agents': required_agents,
                            'agent_position': agent_pos,
                            'huntable_positions': huntable_positions
                        }
        
        for hare_id, (hare_x, hare_y) in self.hare_positions.items():
            if not self.hare_alive[hare_id]:
                continue

            huntable_positions = [
                (hare_x, hare_y + 1),
                (hare_x, hare_y - 1),
                (hare_x + 1, hare_y),
                (hare_x - 1, hare_y)
            ]
            
            hunting_agents = []
            for agent_id, agent_pos in self.agent_positions.items():
                if agent_pos in huntable_positions:
                    hunting_agents.append(agent_id)
            
            if hunting_agents:
                huntable_hares.append({
                    'hare_id': hare_id,
                    'position': (hare_x, hare_y),
                    'hunting_agents': hunting_agents,
                    'huntable_positions': huntable_positions
                })
        
        return {
            'huntable_stags': huntable_stags,
            'other_stags_by_agent': other_stags_by_agent,
            'huntable_hares': huntable_hares,
            'total_huntable_stags': len(huntable_stags),
            'total_huntable_hares': len(huntable_hares)
        }

    def _process_movement_actions(self, actions: Dict[int, int]):
        new_positions = {}
        
        for agent_id, action in actions.items():
            current_x, current_y = self.agent_positions[agent_id]

            if action == self.STAY:
                new_x, new_y = current_x, current_y
            elif action == self.UP:
                new_x, new_y = current_x, current_y + 1
            elif action == self.DOWN:
                new_x, new_y = current_x, current_y - 1
            elif action == self.LEFT:
                new_x, new_y = current_x - 1, current_y
            elif action == self.RIGHT:
                new_x, new_y = current_x + 1, current_y
            else:
                new_x, new_y = current_x, current_y
            
            if (0 <= new_x < self.grid_size_x and 0 <= new_y < self.grid_size_y):
                new_positions[agent_id] = (new_x, new_y)
            else:
                new_positions[agent_id] = (current_x, current_y)
        
        final_positions = {}

        for agent_id, new_pos in new_positions.items():
            final_positions[agent_id] = new_pos

        self.agent_positions.update(final_positions)

    def _process_hunting(self, rewards: Dict[int, float]) -> Dict:
        """
        Returns:
            hunting_info:
                - 'stag_hunts': {stag_id: [participating_agent_ids]}
                - 'hare_hunts': [agent_ids who hunted hares]
        """
        hunting_info = {
            'stag_hunts': {},  # {stag_id: [agents who participated successfully]}
            'stag_attempts': {},  # {stag_id: [agents who attempted to hunt]}
            'hare_hunts': [],   # [agents who hunted hares successfully]
            'hare_attempts': []  # [agents who attempted to hunt hares]
        }
        
        agent_foraged = {agent_id: False for agent_id in self.agents}
        
        for hare_id, hare_pos in self.hare_positions.items():
            if not self.hare_alive.get(hare_id, False):
                continue
                
            agents_at_hare = []
            for agent_id in self.agents:
                if self.agent_positions[agent_id] == hare_pos:
                    agents_at_hare.append(agent_id)
            
            if agents_at_hare:
                hunting_info['hare_attempts'].extend(agents_at_hare)
                
                # hare hunting tie-break
                if self._np_random is not None:
                    selected_agent = agents_at_hare[self._np_random.randint(len(agents_at_hare))]
                else:
                    selected_agent = agents_at_hare[np.random.randint(len(agents_at_hare))]
                rewards[selected_agent] += self.forage_reward
                agent_foraged[selected_agent] = True
                hunting_info['hare_hunts'].append(selected_agent)

                self.hare_alive[hare_id] = False

        for stag_id, stag_pos in self.stag_positions.items():
            if not self.stag_alive.get(stag_id, False):
                continue
            agents_at_stag = []
            for agent_id in self.agents:
                if self.agent_positions[agent_id] == stag_pos:
                    agents_at_stag.append(agent_id)
            
            required_agents = self.required_agents[stag_id]

            if agents_at_stag:
                hunting_info['stag_attempts'][stag_id] = agents_at_stag.copy()
            
            for agent_id in agents_at_stag:
                if agent_id not in required_agents:
                    rewards[agent_id] += self.mauling_punishment
  
            if all(agent_id in agents_at_stag for agent_id in required_agents):
                rewards[stag_id] += self.stag_reward
                for agent_id in agents_at_stag:
                    if agent_id != stag_id:
                        rewards[agent_id] += self.coop_bonus

                hunting_info['stag_hunts'][stag_id] = agents_at_stag.copy()
                self.stag_alive[stag_id] = False
            
            if self.run_away_after_maul and len(agents_at_stag) > 0:
                self.stag_alive[stag_id] = False
        
        return hunting_info
    
    def _get_grid_observations(self) -> Dict[int, np.ndarray]:
        observations = {}
        
        for agent_id in self.agents:
            agent_x, agent_y = self.agent_positions[agent_id]
            grid_obs = [] 
            
            # collect grid information over the vision range around an agent
            for dy in range(-self.vision, self.vision + 1):
                for dx in range(-self.vision, self.vision + 1):
                    check_x, check_y = agent_x + dx, agent_y + dy
                    check_pos = (check_x, check_y)
                    
                    # n+num_agents+1 dimensional tuple: (agent0, agent1, ..., agent_n-1, stag0, stag1, ..., stag_n-1, hare)
                    cell_info = [0.0] * (self.num_agents + self.num_agents + 1)
                    
                    # out of the grid is mask with -1
                    if not (0 <= check_x < self.grid_size_x and 0 <= check_y < self.grid_size_y):
                        cell_info = [-1.0] * (self.num_agents + self.num_agents + 1)
                    else:
                        # check agents (allow overlapping in a cell)
                        for other_agent_id in self.agents:
                            if self.agent_positions[other_agent_id] == check_pos:
                                cell_info[other_agent_id] = 1.0
                        
                        # check alive stags
                        for stag_id, stag_pos in self.stag_positions.items():
                            if stag_pos == check_pos and self.stag_alive.get(stag_id, False):
                                cell_info[self.num_agents + stag_id] = 1.0
                        
                        # check alive hares
                        for hare_id, hare_pos in self.hare_positions.items():
                            if hare_pos == check_pos and self.hare_alive.get(hare_id, False):
                                cell_info[self.num_agents + self.num_agents] = 1.0
                                break
                    grid_obs.extend(cell_info)
     
            observations[agent_id] = np.array(grid_obs, dtype=np.float32)

        return observations
    
    def _get_observations(self) -> Dict[int, np.ndarray]:
        observations = self._get_grid_observations()
        # add noise to observations if std > 0
        if self.obs_noise_std > 0:
            for agent_id in observations:
                observations[agent_id] = observations[agent_id] + np.random.normal(0, self.obs_noise_std, size=observations[agent_id].shape)
                
        return observations
    
    def get_obs_dim(self) -> int:
        return self.obs_dim
    
    def get_action_dim(self) -> int:
        return self.action_dim
    
    def render(self, mode='human'):
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib is needed.")
            return None
            
        if mode == 'human':
            self._render_human()
        elif mode == 'rgb_array':
            return self._render_rgb_array()
        else:
            raise ValueError(f"Unsupported rendering mode: {mode}")
    
    def _render_human(self):
        if not hasattr(self, '_fig') or self._fig is None:
            self._fig, self._ax = plt.subplots(figsize=(8, 8))
            self._ax.set_xlim(-0.5, self.grid_size_x - 0.5)
            self._ax.set_ylim(-0.5, self.grid_size_y - 0.5)
            self._ax.set_aspect('equal')
            self._ax.grid(True, alpha=0.3)
            self._ax.set_title('Stag Hunt Environment', fontsize=14, fontweight='bold')
            
            self._ax.set_xlabel('X coordinate')
            self._ax.set_ylabel('Y coordinate')

            self._ax.set_xticks(range(self.grid_size_x))
            self._ax.set_yticks(range(self.grid_size_y))

            self._legend_elements = []

        for element in self._ax.collections + self._ax.patches + self._ax.texts:
            if hasattr(element, 'remove') and element is not None:
                try:
                    element.remove()
                except:
                    pass

        self._ax.set_xlim(-0.5, self.grid_size_x - 0.5)
        self._ax.set_ylim(-0.5, self.grid_size_y - 0.5)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for agent_id in self.agents:
            if agent_id in self.agent_positions:
                x, y = self.agent_positions[agent_id]
                color = colors[agent_id % len(colors)]
                
                vision_alpha = 0.2
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        vision_x, vision_y = x + dx, y + dy
                        if (0 <= vision_x < self.grid_size_x and 
                            0 <= vision_y < self.grid_size_y):
                            vision_rect = plt.Rectangle((vision_x-0.5, vision_y-0.5), 1, 1, 
                                                     color=color, alpha=vision_alpha, 
                                                     ec='none', linewidth=0)
                            self._ax.add_patch(vision_rect)

                circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, ec='black', linewidth=2)
                self._ax.add_patch(circle)
                self._ax.text(x, y, f'A{agent_id}', ha='center', va='center', 
                            fontsize=10, fontweight='bold', color='white')

        for stag_id in self.stag_positions:
            if not self.stag_alive.get(stag_id, False):
                continue
            x, y = self.stag_positions[stag_id]
            color = colors[stag_id % len(colors)]

            triangle = plt.Polygon([(x, y-0.3), (x-0.3, y+0.3), (x+0.3, y+0.3)], 
                                 color=color, alpha=0.8, ec='black', linewidth=2)
            self._ax.add_patch(triangle)

            self._ax.text(x, y+0.1, f'S{stag_id}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        for hare_id, (x, y) in self.hare_positions.items():
            if not self.hare_alive.get(hare_id, False):
                continue
                
            rect = plt.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                               color='gray', alpha=0.7, ec='black', linewidth=1)
            self._ax.add_patch(rect)
            self._ax.text(x, y, f'H{hare_id}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Agent'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red', 
                      markersize=10, label='Stag'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', 
                      markersize=10, label='Hare'),
            plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.2, label='Agent Vision (5x5)')
        ]
        
        if hasattr(self._ax, 'legend_') and self._ax.legend_ is not None:
            self._ax.legend_.remove()
        
        self._ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.0, 1.0))
        
        info_text = f"Step: {self.step_count}/{self.max_steps}"
        self._ax.text(0.02, 0.98, info_text, transform=self._ax.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.draw()
        plt.pause(0.02)
    
    def _render_rgb_array(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(-0.5, self.grid_size_x - 0.5)
        ax.set_ylim(-0.5, self.grid_size_y - 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('Stag Hunt Environment', fontsize=14, fontweight='bold')
        
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

        ax.set_xticks(range(self.grid_size_x))
        ax.set_yticks(range(self.grid_size_y))
        
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']

        for agent_id in self.agents:
            if agent_id in self.agent_positions:
                x, y = self.agent_positions[agent_id]
                color = colors[agent_id % len(colors)]

                vision_alpha = 0.2
                for dx in [-2, -1, 0, 1, 2]:
                    for dy in [-2, -1, 0, 1, 2]:
                        vision_x, vision_y = x + dx, y + dy
                        if (0 <= vision_x < self.grid_size_x and 
                            0 <= vision_y < self.grid_size_y):
                            vision_rect = plt.Rectangle((vision_x-0.5, vision_y-0.5), 1, 1, 
                                                     color=color, alpha=vision_alpha, 
                                                     ec='none', linewidth=0)
                            ax.add_patch(vision_rect)
                
                circle = plt.Circle((x, y), 0.3, color=color, alpha=0.8, ec='black', linewidth=2)
                ax.add_patch(circle)

                ax.text(x, y, f'A{agent_id}', ha='center', va='center', 
                            fontsize=10, fontweight='bold', color='white')

        for stag_id in self.stag_positions:
            if not self.stag_alive.get(stag_id, False):
                continue
            x, y = self.stag_positions[stag_id]
            color = colors[stag_id % len(colors)]  # Reuse colors cyclically.
            
            # Draw stag as a triangle.
            triangle = plt.Polygon([(x, y-0.3), (x-0.3, y+0.3), (x+0.3, y+0.3)], 
                                 color=color, alpha=0.8, ec='black', linewidth=2)
            ax.add_patch(triangle)
            
            # Label the stag.
            ax.text(x, y+0.1, f'S{stag_id}', ha='center', va='center', 
                        fontsize=10, fontweight='bold', color='white')
        
        # Render only alive hares as gray rectangles.
        for hare_id, (x, y) in self.hare_positions.items():
            if not self.hare_alive.get(hare_id, False):
                continue
                
            rect = plt.Rectangle((x-0.25, y-0.25), 0.5, 0.5, 
                               color='gray', alpha=0.7, ec='black', linewidth=1)
            ax.add_patch(rect)
            
            # Label the hare.
            ax.text(x, y, f'H{hare_id}', ha='center', va='center', 
                        fontsize=8, fontweight='bold', color='white')
        
        # Show step info.
        info_text = f"Step: {self.step_count}/{self.max_steps}"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Convert to an RGB array.
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        arr = np.asarray(buf)
        
        plt.close(fig)
        
        return arr[:, :, :3]  # Drop the alpha channel.
    
    def close_render(self):
        """Close the render window."""
        if hasattr(self, '_fig') and self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None

