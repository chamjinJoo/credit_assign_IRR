import numpy as np
from typing import Dict, Tuple, List, Any, Optional
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import copy
from envs.smax_target_env import swap_to_target_aware_inner_smax
MATPLOTLIB_AVAILABLE = True

class MultiAgentEnv(ABC):
    """
    Abstract base class for multi-agent environments.
    """
    
    def __init__(self, num_agents: int, seed: int = None):
        self.num_agents = num_agents
        self.agents = list(range(num_agents))
        self._seed = seed
        self._np_random = None
        
        # Initialize random number generator if seed is provided
        if seed is not None:
            self.seed(seed)
        
    def seed(self, seed: int = None):
        """Set seed for environment random number generator."""
        if seed is not None:
            self._seed = seed
            self._np_random = np.random.RandomState(seed)
        return [self._seed]
    
    @abstractmethod
    def reset(self) -> Dict[int, np.ndarray]:
        """Reset environment and return initial observations."""
        pass
    
    @abstractmethod  
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],  # observations
        Dict[int, float],       # rewards
        Dict[int, bool],        # dones
        Dict[str, Any]          # info
    ]:
        """Take actions and return next observations, rewards, dones, info."""
        pass
    
    @abstractmethod
    def get_obs_dim(self) -> int:
        """Get observation dimension."""
        pass
    
    @abstractmethod
    def get_action_dim(self) -> int:
        """Get action dimension."""
        pass

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

class SMAXWrapper(MultiAgentEnv):
    """
    JaxMARL SMAX(HeuristicEnemySMAX) Wrapper.
    """

    @staticmethod
    def _build_smax_env_kwargs(
        map_name: Optional[str],
        num_allies: int,
        num_enemies: int,
        max_steps: int,
        map_width: int,
        map_height: int,
        world_steps_per_env_step: int,
        time_per_step: float,
        see_enemy_actions: bool,
        won_battle_bonus: float,
        walls_cause_death: bool,
        observation_type: str,
        smacv2_position_generation: bool,
        smacv2_unit_type_generation: bool,
    ) -> Dict[str, Any]:
        if observation_type not in {"unit_list", "conic"}:
            raise ValueError(
                f"Unsupported SMAX observation_type: {observation_type}. "
                "Expected 'unit_list' or 'conic'."
            )

        env_kwargs: Dict[str, Any] = {}
        if map_name is not None:
            try:
                from jaxmarl.environments.smax.smax_env import map_name_to_scenario

                scenario = map_name_to_scenario(map_name)
                env_kwargs["scenario"] = scenario
                env_kwargs["num_allies"] = scenario.num_allies
                env_kwargs["num_enemies"] = scenario.num_enemies
                env_kwargs["smacv2_position_generation"] = scenario.smacv2_position_generation
                env_kwargs["smacv2_unit_type_generation"] = scenario.smacv2_unit_type_generation
                print(
                    f"[SMAXWrapper] '{map_name}': "
                    f"{scenario.num_allies} allies vs {scenario.num_enemies} enemies"
                )
            except KeyError:
                print(
                    f"[SMAXWrapper] Warning: Unknown map_name '{map_name}', "
                    f"falling back to num_allies={num_allies}, num_enemies={num_enemies}"
                )
                env_kwargs["num_allies"] = num_allies
                env_kwargs["num_enemies"] = num_enemies
                env_kwargs["smacv2_position_generation"] = smacv2_position_generation
                env_kwargs["smacv2_unit_type_generation"] = smacv2_unit_type_generation
        else:
            env_kwargs["num_allies"] = num_allies
            env_kwargs["num_enemies"] = num_enemies
            env_kwargs["smacv2_position_generation"] = smacv2_position_generation
            env_kwargs["smacv2_unit_type_generation"] = smacv2_unit_type_generation

        env_kwargs["max_steps"] = max_steps
        env_kwargs["map_width"] = map_width
        env_kwargs["map_height"] = map_height
        env_kwargs["world_steps_per_env_step"] = world_steps_per_env_step
        env_kwargs["time_per_step"] = time_per_step
        env_kwargs["see_enemy_actions"] = see_enemy_actions
        env_kwargs["won_battle_bonus"] = won_battle_bonus
        env_kwargs["walls_cause_death"] = walls_cause_death
        env_kwargs["observation_type"] = observation_type
        return env_kwargs

    @staticmethod
    def _build_default_allowed_targets(
        num_allies: int,
        num_enemies: int,
        allowed_target_window: Optional[int] = None,
    ) -> Dict[int, List[int]]:
        import math

        step = num_enemies / max(1, num_allies)
        if allowed_target_window is None:
            window = min(num_enemies, max(2, math.ceil(step) + 1))
        else:
            window = min(num_enemies, max(1, int(allowed_target_window)))
        allowed_targets: Dict[int, List[int]] = {}
        for ally_id in range(num_allies):
            start = round(ally_id * step)
            allowed_targets[ally_id] = [
                (start + j) % num_enemies for j in range(window)
            ]
        return allowed_targets

    @staticmethod
    def _build_cooperative_kill_assignments(
        num_allies: int,
        num_enemies: int,
        allowed_targets: Optional[Dict[int, List[int]]] = None,
        cooperative_size: int = 2,
    ) -> Dict[int, List[int]]:
        required_agents: Dict[int, List[int]] = {}
        cooperative_size = max(1, int(cooperative_size))

        for enemy_id in range(num_enemies):
            allowed_agents = []
            if allowed_targets is not None:
                allowed_agents = [
                    agent_id
                    for agent_id in range(num_allies)
                    if enemy_id in allowed_targets.get(agent_id, [])
                ]

            preferred_owner = enemy_id % max(1, num_allies)
            primary_pool = allowed_agents if allowed_agents else list(range(num_allies))
            owner = preferred_owner if preferred_owner in primary_pool else primary_pool[0]

            coalition: List[int] = [owner]
            seen = {owner}
            target_size = min(cooperative_size, num_allies)

            for offset in range(1, num_allies):
                if len(coalition) >= target_size:
                    break
                candidate = (owner + offset) % num_allies
                if candidate in primary_pool and candidate not in seen:
                    coalition.append(candidate)
                    seen.add(candidate)

            if len(coalition) < target_size:
                for offset in range(1, num_allies):
                    if len(coalition) >= target_size:
                        break
                    candidate = (owner + offset) % num_allies
                    if candidate not in seen:
                        coalition.append(candidate)
                        seen.add(candidate)

            required_agents[enemy_id] = coalition

        return required_agents

    @staticmethod
    def _augment_allowed_targets_with_required_agents(
        allowed_targets: Optional[Dict[int, List[int]]],
        required_agents: Dict[int, List[int]],
        num_allies: int,
    ) -> Optional[Dict[int, List[int]]]:
        if allowed_targets is None:
            return None

        merged = {
            agent_id: list(targets)
            for agent_id, targets in allowed_targets.items()
        }
        for agent_id in range(num_allies):
            merged.setdefault(agent_id, [])

        for enemy_id, assigned_agents in required_agents.items():
            for agent_id in assigned_agents:
                if enemy_id not in merged[agent_id]:
                    merged[agent_id].append(enemy_id)
                    merged[agent_id].sort()

        return merged

    @staticmethod
    def _build_allowed_target_matrix(
        num_allies: int,
        num_enemies: int,
        allowed_targets: Optional[Dict[int, List[int]]],
    ) -> np.ndarray:
        matrix = np.ones((num_allies, num_enemies), dtype=np.float32)
        if allowed_targets is None:
            return matrix
        matrix.fill(0.0)
        for agent_id in range(num_allies):
            for enemy_id in allowed_targets.get(agent_id, []):
                if 0 <= enemy_id < num_enemies:
                    matrix[agent_id, enemy_id] = 1.0
        return matrix

    @staticmethod
    def _build_cooperative_kill_reward_tables(
        required_agents: Dict[int, List[int]],
        num_allies: int,
        num_enemies: int,
        owner_ratio: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        required_mask = np.zeros((num_enemies, num_allies), dtype=np.bool_)
        reward_weights = np.zeros((num_enemies, num_allies), dtype=np.float32)

        owner_ratio = float(np.clip(owner_ratio, 0.0, 1.0))
        for enemy_id in range(num_enemies):
            assigned_agents = list(required_agents.get(enemy_id, []))
            if not assigned_agents:
                continue

            for agent_id in assigned_agents:
                required_mask[enemy_id, agent_id] = True

            if len(assigned_agents) == 1:
                reward_weights[enemy_id, assigned_agents[0]] = 1.0
                continue

            owner = assigned_agents[0]
            partners = assigned_agents[1:]
            reward_weights[enemy_id, owner] = owner_ratio
            partner_share = (1.0 - owner_ratio) / max(1, len(partners))
            for agent_id in partners:
                reward_weights[enemy_id, agent_id] = partner_share

        return required_mask, reward_weights

    def __init__(
        self,
        num_allies: int = 5,
        num_enemies: int = 5,
        map_name: str = None,
        seed: int = None,
        max_steps: int = 100,
        use_individual_rewards: bool = False,
        enemy_shoots: bool = True,
        attack_mode: str = "closest",
        map_width: int = 32,
        map_height: int = 32,
        world_steps_per_env_step: int = 8,
        time_per_step: float = 1.0 / 16,
        see_enemy_actions: bool = True,
        won_battle_bonus: float = 1.0,
        walls_cause_death: bool = True,
        observation_type: str = "unit_list",
        smacv2_position_generation: bool = False,
        smacv2_unit_type_generation: bool = False,
        use_allowed_targets: bool = False,
        allowed_target_window: Optional[int] = None,
        allowed_targets: Dict[int, List[int]] = None,
        use_partial_damage_restore: bool = False,
        use_cooperative_kill_bonus: bool = False,
        cooperative_bonus_mode: str = "kill",
        cooperative_kill_window: int = 2,
        cooperative_kill_bonus: float = 2.0,
        cooperative_kill_owner_ratio: float = 0.7,
        cooperative_size: int = 2,
        **kwargs
    ):
        try:
            import jax
            import jax.numpy as jnp
            from jaxmarl.environments.smax import HeuristicEnemySMAX
        except ImportError:
            raise ImportError(
                "JaxMARL is not installed. Please install it using:\n"
                "  pip install jaxmarl"
            )

        self._jax = jax
        self._jnp = jnp

        env_kwargs = self._build_smax_env_kwargs(
            map_name=map_name,
            num_allies=num_allies,
            num_enemies=num_enemies,
            max_steps=max_steps,
            map_width=map_width,
            map_height=map_height,
            world_steps_per_env_step=world_steps_per_env_step,
            time_per_step=time_per_step,
            see_enemy_actions=see_enemy_actions,
            won_battle_bonus=won_battle_bonus,
            walls_cause_death=walls_cause_death,
            observation_type=observation_type,
            smacv2_position_generation=smacv2_position_generation,
            smacv2_unit_type_generation=smacv2_unit_type_generation,
        )

        self._env = HeuristicEnemySMAX(
            enemy_shoots=enemy_shoots,
            attack_mode=attack_mode,
            **env_kwargs
        )

        _num_allies = self._env.num_allies
        _num_enemies = self._env.num_enemies

        super().__init__(num_agents=_num_allies, seed=seed)

        self.num_enemies = _num_enemies
        self.use_individual_rewards = use_individual_rewards
        self.map_name = map_name
        self.max_steps = max_steps
        self.use_cooperative_kill_bonus = use_cooperative_kill_bonus
        self.cooperative_bonus_mode = str(cooperative_bonus_mode).lower()
        if self.cooperative_bonus_mode not in ("kill", "damage"):
            raise ValueError(
                f"Unsupported cooperative_bonus_mode: {cooperative_bonus_mode}. "
                "Use 'kill' or 'damage'."
            )
        self.cooperative_size = max(1, int(cooperative_size))
        self.cooperative_kill_window = max(1, int(cooperative_kill_window))
        self.cooperative_kill_bonus = float(cooperative_kill_bonus)
        self.cooperative_kill_owner_ratio = float(cooperative_kill_owner_ratio)

        self._agent_keys = [f"ally_{i}" for i in range(_num_allies)]
        self._id_to_key = {i: f"ally_{i}" for i in range(_num_allies)}
        self._key_to_id = {f"ally_{i}": i for i in range(_num_allies)}

        self.obs_dim = self._env.obs_size
        self.action_dim = self._env.num_ally_actions
        self._unit_type_health_np = np.array(self._env.unit_type_health, dtype=np.float32)
        self._unit_type_health_jax = self._jnp.array(self._env.unit_type_health)
        self._use_self_play_reward = bool(self._env.use_self_play_reward)
        self._won_battle_bonus = float(self._env.won_battle_bonus)

        # JAX PRNGKey
        _seed_val = seed if seed is not None else 0
        self._rng = jax.random.PRNGKey(_seed_val)
        self._state = None

        # allowed_targets
        self.allowed_target_window = None if allowed_target_window is None else max(1, int(allowed_target_window))
        target_window_covers_all = (
            allowed_targets is None
            and self.allowed_target_window is not None
            and self.allowed_target_window >= _num_enemies
        )
        self.use_allowed_targets = bool(use_allowed_targets) and not target_window_covers_all
        self._num_movement_actions = self._env.num_movement_actions  # Usually 5.
        if self.use_allowed_targets:
            if allowed_targets is not None:
                self.allowed_targets = allowed_targets
            else:
                self.allowed_targets = self._build_default_allowed_targets(
                    _num_allies,
                    _num_enemies,
                    self.allowed_target_window,
                )
        else:
            self.allowed_targets = None
        self.use_partial_damage_restore = bool(use_partial_damage_restore)

        self.cooperative_kill_required_agents = self._build_cooperative_kill_assignments(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
            self.cooperative_size,
        )
        if self.use_allowed_targets and self.use_cooperative_kill_bonus:
            self.allowed_targets = self._augment_allowed_targets_with_required_agents(
                self.allowed_targets,
                self.cooperative_kill_required_agents,
                _num_allies,
            )
        self._allowed_target_matrix_np = self._build_allowed_target_matrix(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
        )
        self._allowed_target_matrix_jax = self._jnp.array(self._allowed_target_matrix_np)
        if self.use_allowed_targets:
            self._env = swap_to_target_aware_inner_smax(
                self._env,
                enemy_shoots=enemy_shoots,
                attack_mode=attack_mode,
                env_kwargs=env_kwargs,
                allowed_target_matrix=self._allowed_target_matrix_np,
                use_partial_damage_restore=self.use_partial_damage_restore,
            )
        self._coop_required_mask_np, self._coop_reward_weights_np = (
            self._build_cooperative_kill_reward_tables(
                self.cooperative_kill_required_agents,
                _num_allies,
                _num_enemies,
                self.cooperative_kill_owner_ratio,
            )
        )
        self._coop_attack_memory = np.zeros((_num_allies, _num_enemies), dtype=np.int32)

        if self.use_allowed_targets:
            mode = "partial" if self.use_partial_damage_restore else "full"
            print(
                f"[SMAXWrapper] allowed_targets ({mode} source-level damage mask, "
                f"window={self.allowed_target_window}): {self.allowed_targets}"
            )

        print(f"[SMAXWrapper] obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"[SMAXWrapper] {_num_allies} allies vs {_num_enemies} enemies (heuristic)")

        # Warm up JAX JIT before entering the training loop.
        self._warmup()

    def _warmup(self):
        import time
        import numpy as np

        t0 = time.time()

        for episode_idx in range(3):
            self._rng, reset_key = self._jax.random.split(self._rng)
            obs_dict, warmup_state = self._env.reset(reset_key)
            self._jax.block_until_ready(list(obs_dict.values()))

            for step_idx in range(self._env.max_steps + 5):
                avail = self._env.get_avail_actions(warmup_state)
                self._jax.block_until_ready(list(avail.values()))

                dummy_actions = {}
                for key in self._agent_keys:
                    avail_arr = np.array(avail[key])
                    valid = np.where(avail_arr > 0)[0]
                    chosen = int(valid[0]) if len(valid) > 0 else 4
                    dummy_actions[key] = self._jnp.int32(chosen)

                self._rng, step_key = self._jax.random.split(self._rng)
                obs2, warmup_state2, rew2, done2, _ = self._env.step_env(
                    step_key, warmup_state, dummy_actions
                )
                self._jax.block_until_ready(list(obs2.values()))
                warmup_state = warmup_state2

                all_done = bool(np.array(done2.get("__all__", False)))
                if all_done:
                    break

        self._state = None

    def _apply_cooperative_kill_bonus_numpy(
        self,
        enemy_unit_types: np.ndarray,
        prev_enemy_health: np.ndarray,
        next_enemy_health: np.ndarray,
        prev_enemy_alive: np.ndarray,
        next_enemy_alive: np.ndarray,
        actions_by_agent: Dict[int, int],
    ) -> Tuple[np.ndarray, List[int]]:
        if not self.use_cooperative_kill_bonus:
            return np.zeros(self.num_agents, dtype=np.float32), []

        attack_memory = np.maximum(self._coop_attack_memory - 1, 0)
        for agent_id in range(self.num_agents):
            action = int(actions_by_agent.get(agent_id, 0))
            if action >= self._num_movement_actions:
                enemy_id = action - self._num_movement_actions
                if 0 <= enemy_id < self.num_enemies:
                    attack_memory[agent_id, enemy_id] = self.cooperative_kill_window

        killed_mask = prev_enemy_alive & (~next_enemy_alive)
        damage = np.maximum(prev_enemy_health - next_enemy_health, 0.0)
        damage_ratio = damage / np.maximum(
            self._unit_type_health_np[np.array(enemy_unit_types, dtype=np.int32)],
            1.0e-8,
        )
        damaged_mask = damage > 1.0e-8
        recent_attacks = attack_memory > 0
        coop_ready = np.all(
            np.where(self._coop_required_mask_np, recent_attacks.T, True),
            axis=-1,
        )
        if self.cooperative_bonus_mode == "kill":
            coop_success = killed_mask & coop_ready
            event_scale = coop_success.astype(np.float32) * self.cooperative_kill_bonus
        else:
            coop_success = damaged_mask & coop_ready
            event_scale = damage_ratio.astype(np.float32) * coop_success.astype(np.float32) * self.cooperative_kill_bonus

        bonus = event_scale @ self._coop_reward_weights_np
        self._coop_attack_memory = attack_memory
        return bonus.astype(np.float32), np.where(coop_success)[0].tolist()

    def _compute_damage_masked_enemy_mask_numpy(
        self,
        actions_by_agent: Dict[int, int],
    ) -> np.ndarray:
        masked_enemies = np.zeros(self.num_enemies, dtype=np.bool_)
        if not self.use_allowed_targets or self.allowed_targets is None:
            return masked_enemies

        for agent_id in range(self.num_agents):
            action = int(actions_by_agent.get(agent_id, 0))
            if action < self._num_movement_actions:
                continue
            enemy_id = action - self._num_movement_actions
            if enemy_id < 0 or enemy_id >= self.num_enemies:
                continue
            if self._allowed_target_matrix_np[agent_id, enemy_id] <= 0.0:
                masked_enemies[enemy_id] = True

        return masked_enemies

    def _compute_team_reward_numpy(self, prev_state, next_state) -> float:
        health_before = np.array(prev_state.state.unit_health, dtype=np.float32)
        health_after = np.array(next_state.state.unit_health, dtype=np.float32)
        unit_types = np.array(next_state.state.unit_types, dtype=np.int32)
        normalized_delta = (health_after - health_before) / self._unit_type_health_np[unit_types]
        enemy_health_decrease = np.sum(normalized_delta[self._env.num_allies:])
        enemy_health_decrease_reward = abs(enemy_health_decrease) / float(self.num_enemies)
        if self._use_self_play_reward:
            enemy_health_decrease_reward = 0.0

        alive = np.array(next_state.state.unit_alive, dtype=np.bool_)
        won_battle = bool(np.all(~alive[self._env.num_allies:]))
        lost_battle = bool(np.all(~alive[:self._env.num_allies]))
        lost_battle_bonus = (
            -self._won_battle_bonus
            if lost_battle and self._use_self_play_reward and not won_battle
            else 0.0
        )
        won_battle_bonus = self._won_battle_bonus if won_battle and not lost_battle else 0.0
        return float(enemy_health_decrease_reward + won_battle_bonus + lost_battle_bonus)

    def reset(self) -> Dict[int, np.ndarray]:
        """Reset the environment and return the initial observations."""
        self._rng, reset_key = self._jax.random.split(self._rng)
        obs_dict, self._state = self._env.reset(reset_key)
        self._coop_attack_memory = np.zeros((self.num_agents, self.num_enemies), dtype=np.int32)

        # Dict[str, jax.Array] → Dict[int, np.ndarray]
        observations = {}
        for i, key in enumerate(self._agent_keys):
            observations[i] = np.array(obs_dict[key], dtype=np.float32)

        return observations

    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[str, Any]
    ]:
        assert self._state is not None, "call reset() before calling this method."
        prev_state = self._state
        prev_enemy_alive = np.array(
            prev_state.state.unit_alive[self._env.num_allies:],
            dtype=np.bool_,
        )
        prev_enemy_health = np.array(
            prev_state.state.unit_health[self._env.num_allies:],
            dtype=np.float32,
        )
        enemy_unit_types = np.array(
            prev_state.state.unit_types[self._env.num_allies:],
            dtype=np.int32,
        )

        # Dict[int, int] → Dict[str, jnp.int32]
        jax_actions = {
            self._id_to_key[i]: self._jnp.int32(a)
            for i, a in actions.items()
        }
        
        self._rng, step_key = self._jax.random.split(self._rng)
        _, next_state, _, _, _ = \
            self._env.step_env(step_key, self._state, jax_actions)
        self._state = next_state
        damage_masked_enemies = np.where(
            self._compute_damage_masked_enemy_mask_numpy(actions)
        )[0].tolist()
        obs_dict = self._env.get_obs(self._state)
        observations = {}
        for i, key in enumerate(self._agent_keys):
            observations[i] = np.array(obs_dict[key], dtype=np.float32)

        team_reward = self._compute_team_reward_numpy(prev_state, self._state)
        rewards = {i: team_reward for i in range(self.num_agents)}

        next_enemy_alive = np.array(
            self._state.state.unit_alive[self._env.num_allies:],
            dtype=np.bool_,
        )
        next_enemy_health = np.array(
            self._state.state.unit_health[self._env.num_allies:],
            dtype=np.float32,
        )
        coop_bonus, cooperative_kills = self._apply_cooperative_kill_bonus_numpy(
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
            actions,
        )
        for agent_id in range(self.num_agents):
            rewards[agent_id] += float(coop_bonus[agent_id])

        all_done = bool(np.array(self._env.is_terminal(self._state)).item())

        dones = {i: all_done for i in range(self.num_agents)}

        # Mark the battle as won when all enemies are dead.
        enemies_alive = self._state.state.unit_alive[self._env.num_allies:]
        battle_won = bool(np.array(~self._jnp.any(enemies_alive)).item())

        info = {
            'battle_won': battle_won,
            'episode_done': all_done,
            'cooperative_kills': cooperative_kills,
            'damage_masked_enemies': damage_masked_enemies,
        }

        return observations, rewards, dones, info

    def get_avail_actions(self) -> List[List[int]]:
        """
        Returns:
            List[List[int]]: shape (num_agents, action_dim), values are 0 or 1.
        """
        assert self._state is not None, "Call reset() before get_avail_actions()."

        avail_dict = self._env.get_avail_actions(self._state)

        avail_list = []
        for i, key in enumerate(self._agent_keys):
            mask = np.array(avail_dict[key], dtype=np.int32)
            avail_list.append(mask.tolist())

        return avail_list

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def get_env_info(self) -> Dict[str, Any]:
        return {
            'obs_shape': self.obs_dim,
            'n_actions': self.action_dim,
            'n_agents': self.num_agents,
            'episode_limit': self.max_steps,
        }

    def close(self):
        pass

# SMAXBatchWrapper: jax.vmap based parallel SMAX 
class SMAXBatchWrapper(MultiAgentEnv):
    """
    JAX vmap for parallel SMAX environments.

    Args:
        n_envs: (vmap batchsize)
        map_name: e.g., 'smacv2_5_units', '3m'
        seed:
        ... (other SMAXWrapper params...)
    """

    def __init__(
        self,
        n_envs: int = 4,
        map_name: Optional[str] = None,
        num_allies: int = 5,
        num_enemies: int = 5,
        seed: Optional[int] = None,
        max_steps: int = 100,
        use_individual_rewards: bool = False,
        enemy_shoots: bool = True,
        attack_mode: str = 'closest',
        map_width: int = 32,
        map_height: int = 32,
        world_steps_per_env_step: int = 8,
        time_per_step: float = 1.0 / 16,
        see_enemy_actions: bool = True,
        won_battle_bonus: float = 1.0,
        walls_cause_death: bool = True,
        observation_type: str = "unit_list",
        smacv2_position_generation: bool = False,
        smacv2_unit_type_generation: bool = False,
        use_allowed_targets: bool = True,
        allowed_target_window: Optional[int] = None,
        allowed_targets: Dict[int, List[int]] = None,
        use_partial_damage_restore: bool = False,
        use_cooperative_kill_bonus: bool = True,
        cooperative_bonus_mode: str = "kill",
        cooperative_size: int = 2,
        cooperative_kill_window: int = 2,
        cooperative_kill_bonus: float = 2.0,
        cooperative_kill_owner_ratio: float = 0.7,
        warmup_episodes: int = 3,
    ):
        try:
            import jax
            import jax.numpy as jnp
            from jaxmarl.environments.smax import HeuristicEnemySMAX
        except ImportError as e:
            raise ImportError(f"jax/jaxmarl must be installed: {e}")

        self._jax = jax
        self._jnp = jnp

        self.n_envs = n_envs
        self.n_rollout_threads = n_envs

        env_kwargs = SMAXWrapper._build_smax_env_kwargs(
            map_name=map_name,
            num_allies=num_allies,
            num_enemies=num_enemies,
            max_steps=max_steps,
            map_width=map_width,
            map_height=map_height,
            world_steps_per_env_step=world_steps_per_env_step,
            time_per_step=time_per_step,
            see_enemy_actions=see_enemy_actions,
            won_battle_bonus=won_battle_bonus,
            walls_cause_death=walls_cause_death,
            observation_type=observation_type,
            smacv2_position_generation=smacv2_position_generation,
            smacv2_unit_type_generation=smacv2_unit_type_generation,
        )
        if map_name is not None and "scenario" in env_kwargs:
            print(
                f"[SMAXBatchWrapper] '{map_name}': "
                f"{env_kwargs['num_allies']}v{env_kwargs['num_enemies']}, n_envs={n_envs}"
            )

        self._env = HeuristicEnemySMAX(
            enemy_shoots=enemy_shoots,
            attack_mode=attack_mode,
            **env_kwargs
        )

        _num_allies = self._env.num_allies
        _num_enemies = self._env.num_enemies
        super().__init__(num_agents=_num_allies, seed=seed)

        self.num_enemies = _num_enemies
        self.map_name = map_name
        self.max_steps = max_steps
        self.use_individual_rewards = use_individual_rewards
        self.use_cooperative_kill_bonus = use_cooperative_kill_bonus
        self.cooperative_bonus_mode = str(cooperative_bonus_mode).lower()
        if self.cooperative_bonus_mode not in ("kill", "damage"):
            raise ValueError(
                f"Unsupported cooperative_bonus_mode: {cooperative_bonus_mode}. "
                "Use 'kill' or 'damage'."
            )
        self.cooperative_size = max(1, int(cooperative_size))
        self.cooperative_kill_window = max(1, int(cooperative_kill_window))
        self.cooperative_kill_bonus = float(cooperative_kill_bonus)
        self.cooperative_kill_owner_ratio = float(cooperative_kill_owner_ratio)
        self._agent_keys = [f"ally_{i}" for i in range(_num_allies)]
        self._id_to_key = {i: f"ally_{i}" for i in range(_num_allies)}

        self.obs_dim = self._env.obs_size
        self.action_dim = self._env.num_ally_actions
        self._num_movement_actions = self._env.num_movement_actions
        self.use_partial_damage_restore = bool(use_partial_damage_restore)
        self.allowed_target_window = None if allowed_target_window is None else max(1, int(allowed_target_window))
        self._unit_type_health_jax = jnp.array(self._env.unit_type_health)
        self._unit_type_health_np = np.array(self._env.unit_type_health, dtype=np.float32)
        self._use_self_play_reward = bool(self._env.use_self_play_reward)
        self._won_battle_bonus = float(self._env.won_battle_bonus)

        # target allocation
        target_window_covers_all = (
            allowed_targets is None
            and self.allowed_target_window is not None
            and self.allowed_target_window >= _num_enemies
        )
        self.use_allowed_targets = bool(use_allowed_targets) and not target_window_covers_all
        if self.use_allowed_targets:
            if allowed_targets is not None:
                self.allowed_targets = allowed_targets
            else:
                self.allowed_targets = SMAXWrapper._build_default_allowed_targets(
                    _num_allies,
                    _num_enemies,
                    self.allowed_target_window,
                )
        else:
            self.allowed_targets = None
        self.cooperative_kill_required_agents = SMAXWrapper._build_cooperative_kill_assignments(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
            self.cooperative_size,
        )
        if self.use_allowed_targets and self.use_cooperative_kill_bonus:
            self.allowed_targets = SMAXWrapper._augment_allowed_targets_with_required_agents(
                self.allowed_targets,
                self.cooperative_kill_required_agents,
                _num_allies,
            )
        self._allowed_target_matrix_np = SMAXWrapper._build_allowed_target_matrix(
            _num_allies,
            _num_enemies,
            self.allowed_targets,
        )
        self._allowed_target_matrix_jax = jnp.array(self._allowed_target_matrix_np)
        if self.use_allowed_targets:
            self._env = swap_to_target_aware_inner_smax(
                self._env,
                enemy_shoots=enemy_shoots,
                attack_mode=attack_mode,
                env_kwargs=env_kwargs,
                allowed_target_matrix=self._allowed_target_matrix_np,
                use_partial_damage_restore=self.use_partial_damage_restore,
            )

        self._coop_required_mask_np, self._coop_reward_weights_np = (
            SMAXWrapper._build_cooperative_kill_reward_tables(
                self.cooperative_kill_required_agents,
                _num_allies,
                _num_enemies,
                self.cooperative_kill_owner_ratio,
            )
        )
        self._coop_required_mask_jax = jnp.array(self._coop_required_mask_np)
        self._coop_reward_weights_jax = jnp.array(self._coop_reward_weights_np)

        _seed_val = seed if seed is not None else 0
        self._rng = jax.random.PRNGKey(_seed_val)
        self._warmup_episodes = max(0, int(warmup_episodes))
        self._batch_states = None  # Batched states with shape (n_envs, ...).
        self._coop_attack_memory_np = np.zeros((n_envs, _num_allies, _num_enemies), dtype=np.int32)
        self._coop_attack_memory_jax = jnp.zeros((n_envs, _num_allies, _num_enemies), dtype=jnp.int32)

        self._batch_reset = jax.jit(jax.vmap(self._env.reset))
        self._batch_step = jax.jit(jax.vmap(self._env.step_env))
        self._batch_obs = jax.jit(jax.vmap(self._env.get_obs))
        self._batch_avail = jax.jit(jax.vmap(self._env.get_avail_actions))
        self._batch_terminal = jax.jit(jax.vmap(self._env.is_terminal))

        if self.use_allowed_targets:
            mode = "partial" if self.use_partial_damage_restore else "full"
            print(
                f"[SMAXBatchWrapper] allowed_targets ({mode} source-level damage mask, "
                f"window={self.allowed_target_window}): {self.allowed_targets}"
            )
        if self.use_cooperative_kill_bonus:
            print(
                "[SMAXBatchWrapper] cooperative_kill_bonus: "
                f"mode={self.cooperative_bonus_mode}, "
                f"cooperative_size={self.cooperative_size}, "
                f"window={self.cooperative_kill_window}, bonus={self.cooperative_kill_bonus}, "
                f"required_agents={self.cooperative_kill_required_agents}"
            )
        print(f"[SMAXBatchWrapper] obs_dim={self.obs_dim}, action_dim={self.action_dim}")
        print(f"[SMAXBatchWrapper] {_num_allies}v{_num_enemies} × {n_envs} emvs")

        if self._warmup_episodes > 0:
            self._warmup()

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------

    def _warmup(self):
        """JIT compile vmap functions."""
        import time
        import numpy as np

        t0 = time.time()

        for ep in range(self._warmup_episodes):
            self._rng, *env_keys = self._jax.random.split(self._rng, self.n_envs + 1)
            keys_arr = self._jnp.stack(env_keys)

            obs_b, batch_st = self._batch_reset(keys_arr)
            self._jax.block_until_ready(obs_b)

            for _ in range(self._env.max_steps + 5):
                avail_b = self._batch_avail(batch_st)
                self._jax.block_until_ready(avail_b)

                dummy_actions = {}
                for key in self._agent_keys:
                    avail_np = np.array(avail_b[key])          # (n_envs, action_dim)
                    chosen = np.argmax(avail_np, axis=1)        # (n_envs,)
                    dummy_actions[key] = self._jnp.array(chosen, dtype=self._jnp.int32)

                self._rng, *step_keys = self._jax.random.split(self._rng, self.n_envs + 1)
                step_keys_arr = self._jnp.stack(step_keys)

                obs_b2, batch_st2, _, dones_b, _ = self._batch_step(
                    step_keys_arr, batch_st, dummy_actions
                )
                self._jax.block_until_ready(obs_b2)
                batch_st = batch_st2

                if np.all(np.array(dones_b["__all__"])):
                    break

        self._batch_states = None

    def _pack_batch_state(self, env_state, coop_attack_memory):
        return {
            "env_state": env_state,
            "coop_attack_memory": coop_attack_memory,
        }

    def _unpack_batch_state(self, batch_state):
        if isinstance(batch_state, dict) and "env_state" in batch_state:
            return batch_state["env_state"], batch_state["coop_attack_memory"]
        zero_memory = self._jnp.zeros(
            (self.n_envs, self.num_agents, self.num_enemies),
            dtype=self._jnp.int32,
        )
        return batch_state, zero_memory

    def _apply_cooperative_kill_bonus_numpy_batch(
        self,
        attack_memory: np.ndarray,
        actions: np.ndarray,
        enemy_unit_types: np.ndarray,
        prev_enemy_health: np.ndarray,
        next_enemy_health: np.ndarray,
        prev_enemy_alive: np.ndarray,
        next_enemy_alive: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not self.use_cooperative_kill_bonus:
            zero_bonus = np.zeros((actions.shape[0], self.num_agents), dtype=np.float32)
            zero_success = np.zeros((actions.shape[0], self.num_enemies), dtype=np.bool_)
            return zero_bonus, attack_memory, zero_success

        updated_memory = np.maximum(attack_memory - 1, 0)
        attack_targets = np.clip(actions - self._num_movement_actions, 0, self.num_enemies - 1)
        attack_mask = actions >= self._num_movement_actions
        for enemy_id in range(self.num_enemies):
            hit_enemy = attack_mask & (attack_targets == enemy_id)
            updated_memory[:, :, enemy_id] = np.where(
                hit_enemy,
                self.cooperative_kill_window,
                updated_memory[:, :, enemy_id],
            )

        killed_mask = prev_enemy_alive & (~next_enemy_alive)
        enemy_max_health = np.maximum(self._unit_type_health_np[enemy_unit_types], 1.0e-8)
        damage = np.maximum(prev_enemy_health - next_enemy_health, 0.0)
        damage_ratio = damage / enemy_max_health
        damaged_mask = damage > 1.0e-8
        recent_attacks = np.transpose(updated_memory > 0, (0, 2, 1))
        coop_ready = np.all(
            np.where(self._coop_required_mask_np[None, :, :], recent_attacks, True),
            axis=-1,
        )
        if self.cooperative_bonus_mode == "kill":
            coop_success = killed_mask & coop_ready
            event_scale = coop_success.astype(np.float32) * self.cooperative_kill_bonus
        else:
            coop_success = damaged_mask & coop_ready
            event_scale = damage_ratio.astype(np.float32) * coop_success.astype(np.float32) * self.cooperative_kill_bonus
        bonus = event_scale @ self._coop_reward_weights_np
        return bonus.astype(np.float32), updated_memory, coop_success

    def _apply_cooperative_kill_bonus_jax(
        self,
        attack_memory,
        actions,
        enemy_unit_types,
        prev_enemy_health,
        next_enemy_health,
        prev_enemy_alive,
        next_enemy_alive,
    ):
        if not self.use_cooperative_kill_bonus:
            zero_bonus = self._jnp.zeros((actions.shape[0], self.num_agents), dtype=self._jnp.float32)
            zero_success = self._jnp.zeros((actions.shape[0], self.num_enemies), dtype=bool)
            return zero_bonus, attack_memory, zero_success

        updated_memory = self._jnp.maximum(attack_memory - 1, 0)
        attack_mask = actions >= self._num_movement_actions
        attack_targets = self._jnp.clip(actions - self._num_movement_actions, 0, self.num_enemies - 1)
        current_targets = self._jax.nn.one_hot(
            attack_targets,
            self.num_enemies,
            dtype=self._jnp.int32,
        )
        current_targets = (
            current_targets
            * attack_mask[..., None].astype(self._jnp.int32)
            * self.cooperative_kill_window
        )
        updated_memory = self._jnp.maximum(updated_memory, current_targets)

        killed_mask = prev_enemy_alive & (~next_enemy_alive)
        enemy_max_health = self._jnp.maximum(self._unit_type_health_jax[enemy_unit_types], 1.0e-8)
        damage = self._jnp.maximum(prev_enemy_health - next_enemy_health, 0.0)
        damage_ratio = damage / enemy_max_health
        damaged_mask = damage > 1.0e-8
        recent_attacks = self._jnp.transpose(updated_memory > 0, (0, 2, 1))
        coop_ready = self._jnp.all(
            self._jnp.where(self._coop_required_mask_jax[None, :, :], recent_attacks, True),
            axis=-1,
        )
        if self.cooperative_bonus_mode == "kill":
            coop_success = killed_mask & coop_ready
            event_scale = coop_success.astype(self._jnp.float32) * self.cooperative_kill_bonus
        else:
            coop_success = damaged_mask & coop_ready
            event_scale = damage_ratio.astype(self._jnp.float32) * coop_success.astype(self._jnp.float32) * self.cooperative_kill_bonus
        bonus = event_scale @ self._coop_reward_weights_jax
        return bonus, updated_memory, coop_success

    def _compute_damage_masked_enemy_mask_jax(self, actions):
        if not self.use_allowed_targets or self.allowed_targets is None:
            return self._jnp.zeros((actions.shape[0], self.num_enemies), dtype=bool)

        attack_mask = actions >= self._num_movement_actions
        attack_targets = self._jnp.clip(
            actions - self._num_movement_actions,
            0,
            self.num_enemies - 1,
        )
        attack_one_hot = self._jax.nn.one_hot(
            attack_targets,
            self.num_enemies,
            dtype=self._jnp.float32,
        )
        attack_one_hot = attack_one_hot * attack_mask[..., None].astype(self._jnp.float32)
        off_target_counts = self._jnp.sum(
            attack_one_hot * (1.0 - self._allowed_target_matrix_jax[None, :, :]),
            axis=1,
        )
        return off_target_counts > 0.0

    def _compute_team_rewards_jax(self, prev_env_state, next_env_state):
        health_before = prev_env_state.state.unit_health
        health_after = next_env_state.state.unit_health
        unit_types = next_env_state.state.unit_types
        normalized_delta = (health_after - health_before) / self._unit_type_health_jax[unit_types]
        enemy_health_decrease = self._jnp.sum(
            normalized_delta[:, self._env.num_allies:],
            axis=1,
        )
        enemy_health_decrease_reward = self._jnp.abs(enemy_health_decrease) / float(self.num_enemies)
        if self._use_self_play_reward:
            enemy_health_decrease_reward = self._jnp.zeros_like(enemy_health_decrease_reward)

        alive = next_env_state.state.unit_alive
        won_battle = self._jnp.all(~alive[:, self._env.num_allies:], axis=1)
        lost_battle = self._jnp.all(~alive[:, :self._env.num_allies], axis=1)
        lost_battle_bonus = self._jnp.where(
            lost_battle & self._use_self_play_reward & ~won_battle,
            -self._won_battle_bonus,
            0.0,
        )
        won_battle_bonus = self._jnp.where(
            won_battle & ~lost_battle,
            self._won_battle_bonus,
            0.0,
        )
        team_rewards = enemy_health_decrease_reward + won_battle_bonus + lost_battle_bonus
        return self._jnp.broadcast_to(team_rewards[:, None], (team_rewards.shape[0], self.num_agents))

    # ------------------------------------------------------------------
    # ParallelEnvWrapper interface
    # ------------------------------------------------------------------

    def reset(self):
        """
        Returns:
            obs_list:   List[Dict[int, np.ndarray]]
            avail_list: List[Dict[int, List[int]]]
        """
        self._rng, *env_keys = self._jax.random.split(self._rng, self.n_envs + 1)
        keys_arr = self._jnp.stack(env_keys)

        obs_b, self._batch_states = self._batch_reset(keys_arr)
        self._jax.block_until_ready(obs_b)
        self._coop_attack_memory_np = np.zeros(
            (self.n_envs, self.num_agents, self.num_enemies),
            dtype=np.int32,
        )

        avail_b = self._batch_avail(self._batch_states)
        return self._obs_to_list(obs_b), self._avail_to_list(avail_b)

    def step(self, actions_batch):
        """
        Args:
            actions_batch: List[Dict[int, int]]

        Returns: (obs, rewards, dones, infos, avail)
        """
        import numpy as np

        jax_actions = self._actions_to_jax(actions_batch)
        actions_np = np.stack(
            [np.array(jax_actions[self._id_to_key[agent_id]], dtype=np.int32) for agent_id in range(self.num_agents)],
            axis=1,
        )
        prev_enemy_alive = np.array(
            self._batch_states.state.unit_alive[:, self._env.num_allies:],
            dtype=np.bool_,
        )
        prev_enemy_health = np.array(
            self._batch_states.state.unit_health[:, self._env.num_allies:],
            dtype=np.float32,
        )
        enemy_unit_types = np.array(
            self._batch_states.state.unit_types[:, self._env.num_allies:],
            dtype=np.int32,
        )

        self._rng, *step_keys = self._jax.random.split(self._rng, self.n_envs + 1)
        step_keys_arr = self._jnp.stack(step_keys)

        _, new_states, _, _, _ = self._batch_step(
            step_keys_arr, self._batch_states, jax_actions
        )
        corrected_states = new_states
        damage_masked = self._compute_damage_masked_enemy_mask_jax(
            self._jnp.array(actions_np, dtype=self._jnp.int32)
        )
        obs_b = self._batch_obs(corrected_states)
        self._jax.block_until_ready(obs_b)
        next_enemy_alive = np.array(
            corrected_states.state.unit_alive[:, self._env.num_allies:],
            dtype=np.bool_,
        )
        next_enemy_health = np.array(
            corrected_states.state.unit_health[:, self._env.num_allies:],
            dtype=np.float32,
        )
        rew_stacked = np.array(
            self._compute_team_rewards_jax(self._batch_states, corrected_states),
            dtype=np.float32,
        )
        coop_bonus, self._coop_attack_memory_np, _ = self._apply_cooperative_kill_bonus_numpy_batch(
            self._coop_attack_memory_np,
            actions_np,
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
        )
        rew_stacked = rew_stacked + coop_bonus

        all_done_np = np.array(self._batch_terminal(corrected_states))   # (n_envs,) bool
        if np.any(all_done_np):
            self._coop_attack_memory_np[all_done_np] = 0

        # reset the episode-ended environments ──────────────────────────────
        if np.any(all_done_np):
            self._rng, *reset_keys = self._jax.random.split(self._rng, self.n_envs + 1)
            reset_keys_arr = self._jnp.stack(reset_keys)
            reset_obs_b, reset_states = self._batch_reset(reset_keys_arr)
            self._jax.block_until_ready(reset_obs_b)

            done_flags = self._jnp.array(all_done_np)   # (n_envs,)

            def _select(new, reset):
                cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
                return self._jnp.where(cond, reset, new)

            self._batch_states = self._jax.tree_map(_select, corrected_states, reset_states)
            final_obs_b = _select(
                self._stack_agent_dict(obs_b),
                self._stack_agent_dict(reset_obs_b),
            )
        else:
            self._batch_states = corrected_states
            final_obs_b = self._stack_agent_dict(obs_b)
        # ──────────────────────────────────────────────────────────

        avail_b = self._batch_avail(self._batch_states)
        damage_masked_np = np.array(damage_masked)
        damage_masked_lists = [np.where(damage_masked_np[e])[0].tolist() for e in range(self.n_envs)]

        agent_dones_np = np.logical_not(
            np.array(corrected_states.state.unit_alive[:, : self.num_agents])
        )

        return (
            self._obs_to_list(final_obs_b),
            [
                {agent_id: float(rew_stacked[e, agent_id]) for agent_id in range(self.num_agents)}
                for e in range(self.n_envs)
            ],
            self._dones_to_list(agent_dones_np),
            self._infos_to_list(corrected_states, all_done_np, damage_masked_lists),
            self._avail_to_list(avail_b),
        )

    def get_avail_actions_batch(self):
        avail_b = self._batch_avail(self._batch_states)
        return self._avail_to_list(avail_b)

    def _stack_agent_dict(self, data_b):
        return self._jnp.stack([data_b[key] for key in self._agent_keys], axis=1)

    def _mask_avail_jax(self, avail_stacked):
        return avail_stacked

    def reset_jax(self):
        self._rng, batch_state, obs, avail = self.reset_jax_pure(self._rng)
        self._batch_states = batch_state["env_state"]
        self._coop_attack_memory_jax = batch_state["coop_attack_memory"]
        return obs, avail

    def reset_jax_pure(self, rng):
        rng, *env_keys = self._jax.random.split(rng, self.n_envs + 1)
        keys_arr = self._jnp.stack(env_keys)
        return rng, *self.reset_jax_pure_from_keys(keys_arr)

    def reset_jax_pure_from_keys(self, keys_arr):
        if keys_arr.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} reset keys, got {keys_arr.shape[0]}")
        obs_b, env_state = self._batch_reset(keys_arr)
        coop_attack_memory = self._jnp.zeros(
            (self.n_envs, self.num_agents, self.num_enemies),
            dtype=self._jnp.int32,
        )
        avail_b = self._batch_avail(env_state)
        return self._pack_batch_state(env_state, coop_attack_memory), self._stack_agent_dict(obs_b), self._mask_avail_jax(self._stack_agent_dict(avail_b))

    def step_jax(self, actions):
        batch_state = self._pack_batch_state(self._batch_states, self._coop_attack_memory_jax)
        (
            self._rng,
            next_batch_state,
            obs,
            avail,
            rewards,
            dones,
            dones_all,
            battle_won,
        ) = self.step_jax_pure(self._rng, batch_state, actions)
        self._batch_states = next_batch_state["env_state"]
        self._coop_attack_memory_jax = next_batch_state["coop_attack_memory"]
        return obs, avail, rewards, dones, dones_all, battle_won

    def step_jax_pure(self, rng, batch_states, actions):
        env_state, coop_attack_memory = self._unpack_batch_state(batch_states)
        jax_actions = {
            self._id_to_key[i]: actions[:, i].astype(self._jnp.int32)
            for i in range(self.num_agents)
        }

        rng, *step_keys = self._jax.random.split(rng, self.n_envs + 1)
        step_keys_arr = self._jnp.stack(step_keys)
        prev_enemy_alive = env_state.state.unit_alive[:, self._env.num_allies:]
        enemy_unit_types = env_state.state.unit_types[:, self._env.num_allies:]
        prev_enemy_health = env_state.state.unit_health[:, self._env.num_allies:]
        _, next_env_state, _, _, _ = self._batch_step(
            step_keys_arr,
            env_state,
            jax_actions,
        )
        obs_b = self._batch_obs(next_env_state)
        agent_dones = ~next_env_state.state.unit_alive[:, : self.num_agents]
        next_enemy_health = next_env_state.state.unit_health[:, self._env.num_allies:]
        next_enemy_alive = next_env_state.state.unit_alive[:, self._env.num_allies:]
        battle_won = ~self._jnp.any(next_enemy_alive, axis=-1)
        rewards_stacked = self._compute_team_rewards_jax(env_state, next_env_state)
        coop_bonus, coop_attack_memory, _ = self._apply_cooperative_kill_bonus_jax(
            coop_attack_memory,
            actions,
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
        )
        rewards_stacked = rewards_stacked + coop_bonus

        rng, *reset_keys = self._jax.random.split(rng, self.n_envs + 1)
        reset_keys_arr = self._jnp.stack(reset_keys)
        reset_obs_b, reset_states = self._batch_reset(reset_keys_arr)
        done_flags = self._batch_terminal(next_env_state)

        def _select(new, reset):
            cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
            return self._jnp.where(cond, reset, new)

        final_env_state = self._jax.tree_map(_select, next_env_state, reset_states)
        final_obs_b = _select(
            self._stack_agent_dict(obs_b),
            self._stack_agent_dict(reset_obs_b),
        )
        avail_b = self._batch_avail(final_env_state)
        coop_attack_memory = self._jnp.where(
            done_flags[:, None, None],
            self._jnp.zeros_like(coop_attack_memory),
            coop_attack_memory,
        )

        return (
            rng,
            self._pack_batch_state(final_env_state, coop_attack_memory),
            final_obs_b,
            self._mask_avail_jax(self._stack_agent_dict(avail_b)),
            rewards_stacked,
            agent_dones,
            done_flags,
            battle_won,
        )

    def step_jax_pure_from_keys(self, rng_keys, batch_states, actions):
        if rng_keys.shape[0] != self.n_envs:
            raise ValueError(f"Expected {self.n_envs} RNG keys, got {rng_keys.shape[0]}")

        env_state, coop_attack_memory = self._unpack_batch_state(batch_states)
        jax_actions = {
            self._id_to_key[i]: actions[:, i].astype(self._jnp.int32)
            for i in range(self.num_agents)
        }

        step_split = self._jax.vmap(lambda key: self._jax.random.split(key, 2))(rng_keys)
        next_rng_keys = step_split[:, 0]
        step_keys_arr = step_split[:, 1]
        prev_enemy_alive = env_state.state.unit_alive[:, self._env.num_allies:]
        enemy_unit_types = env_state.state.unit_types[:, self._env.num_allies:]
        prev_enemy_health = env_state.state.unit_health[:, self._env.num_allies:]
        _, next_env_state, _, _, _ = self._batch_step(
            step_keys_arr,
            env_state,
            jax_actions,
        )
        obs_b = self._batch_obs(next_env_state)
        agent_dones = ~next_env_state.state.unit_alive[:, : self.num_agents]
        next_enemy_health = next_env_state.state.unit_health[:, self._env.num_allies:]
        next_enemy_alive = next_env_state.state.unit_alive[:, self._env.num_allies:]
        battle_won = ~self._jnp.any(next_enemy_alive, axis=-1)
        rewards_stacked = self._compute_team_rewards_jax(env_state, next_env_state)
        coop_bonus, coop_attack_memory, _ = self._apply_cooperative_kill_bonus_jax(
            coop_attack_memory,
            actions,
            enemy_unit_types,
            prev_enemy_health,
            next_enemy_health,
            prev_enemy_alive,
            next_enemy_alive,
        )
        rewards_stacked = rewards_stacked + coop_bonus

        reset_split = self._jax.vmap(lambda key: self._jax.random.split(key, 2))(next_rng_keys)
        final_rng_keys = reset_split[:, 0]
        reset_keys_arr = reset_split[:, 1]
        reset_obs_b, reset_states = self._batch_reset(reset_keys_arr)
        done_flags = self._batch_terminal(next_env_state)

        def _select(new, reset):
            cond = done_flags.reshape((-1,) + (1,) * (new.ndim - 1))
            return self._jnp.where(cond, reset, new)

        final_env_state = self._jax.tree_map(_select, next_env_state, reset_states)
        final_obs_b = _select(
            self._stack_agent_dict(obs_b),
            self._stack_agent_dict(reset_obs_b),
        )
        avail_b = self._batch_avail(final_env_state)
        coop_attack_memory = self._jnp.where(
            done_flags[:, None, None],
            self._jnp.zeros_like(coop_attack_memory),
            coop_attack_memory,
        )

        return (
            final_rng_keys,
            self._pack_batch_state(final_env_state, coop_attack_memory),
            final_obs_b,
            self._mask_avail_jax(self._stack_agent_dict(avail_b)),
            rewards_stacked,
            agent_dones,
            done_flags,
            battle_won,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _obs_to_list(self, obs_b) -> List:
        import numpy as np
        if isinstance(obs_b, dict):
            obs_np = {k: np.array(obs_b[k]) for k in self._agent_keys}
            return [
                {i: obs_np[k][e].astype(np.float32) for i, k in enumerate(self._agent_keys)}
                for e in range(self.n_envs)
            ]

        obs_np = np.array(obs_b)
        return [
            {i: obs_np[e, i].astype(np.float32) for i in range(self.num_agents)}
            for e in range(self.n_envs)
        ]

    def _dones_to_list(self, done_np) -> List:
        return [{i: bool(done_np[e, i]) for i in range(self.num_agents)}
                for e in range(self.n_envs)]

    def _infos_to_list(self, new_states, all_done_np, damage_masked_lists=None) -> List:
        import numpy as np
        enemies_alive = np.array(
            new_states.state.unit_alive[:, self._env.num_allies:]  # (n_envs, n_enemies)
        )
        battle_won = ~np.any(enemies_alive, axis=1)   # (n_envs,)
        info_list = []
        for e in range(self.n_envs):
            info = {'battle_won': bool(battle_won[e]), 'episode_done': bool(all_done_np[e])}
            if damage_masked_lists is not None:
                info['damage_masked_enemies'] = damage_masked_lists[e]
            info_list.append(info)
        return info_list

    def _avail_to_list(self, avail_b) -> List:
        import numpy as np
        avail_np = {k: np.array(avail_b[k]) for k in self._agent_keys}
        return [
            {i: avail_np[k][e].tolist() for i, k in enumerate(self._agent_keys)}
            for e in range(self.n_envs)
        ]

    def _actions_to_jax(self, actions_batch):
        """List[Dict[int, int]] → Dict[str, jnp.ndarray (n_envs,)]"""
        return {
            key: self._jnp.array(
                [int(actions_batch[e][agent_id]) for e in range(self.n_envs)],
                dtype=self._jnp.int32,
            )
            for agent_id, key in self._id_to_key.items()
        }

    # ------------------------------------------------------------------
    # MultiAgentEnv interface
    # ------------------------------------------------------------------

    def get_obs_dim(self) -> int:
        return self.obs_dim

    def get_action_dim(self) -> int:
        return self.action_dim

    def close(self):
        pass
