import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
import os
import json
from collections import defaultdict
import pandas as pd

if TYPE_CHECKING:
    from ..algorithms.pci_mappo import PCIMAPPOAlgorithm
    from ..envs.multi_agent_env import StagHuntEnv


class CausalEffectEvaluator:
    """
    Bridge network의 causal effect 추정 성능을 평가하는 클래스.
    StagHunt 환경에서 다른 agent의 reward가 0, 3, 5일 때의 causal effect를 추정하고 시각화합니다.
    """
    
    def __init__(
        self,
        algorithm: "PCIMAPPOAlgorithm",
        env: "StagHuntEnv",
        num_eval_episodes: int = 100,
        device: str = "cpu"
    ):
        self.algorithm = algorithm
        self.env = env
        self.num_eval_episodes = num_eval_episodes
        self.device = device
        
        # StagHunt 환경의 reward 구조
        self.reward_scenarios = {
            'no_reward': 0.0,      # 다른 agent가 보상을 받지 못하는 경우
            'forage_reward': 3.0,  # 다른 agent가 식물 수확 보상을 받는 경우  
            'stag_reward': 5.0     # 다른 agent가 Stag 사냥 보상을 받는 경우
        }
        
        # 결과 저장용
        self.causal_effects = defaultdict(list)
        self.episode_data = []
        
    def evaluate_causal_effects(self) -> Dict[str, Dict[str, float]]:
        """
        여러 에피소드를 실행하여 causal effect 추정치의 통계를 계산합니다.
        
        Returns:
            Dict containing statistics for each reward scenario
        """
        print(f"Causal effect 평가 시작: {self.num_eval_episodes} 에피소드")
        
        # 알고리즘을 평가 모드로 설정
        self.algorithm.outcome_bridge_net.eval()
        self.algorithm.action_bridge_net.eval()
        
        for episode in range(self.num_eval_episodes):
            if episode % 10 == 0:
                print(f"에피소드 {episode}/{self.num_eval_episodes} 진행 중...")
            
            episode_causal_effects = self._run_single_episode()
            
            # 각 reward 시나리오별로 causal effect 저장
            for scenario, effects in episode_causal_effects.items():
                self.causal_effects[scenario].extend(effects)
        
        # 통계 계산
        statistics = self._compute_statistics()
        
        print("Causal effect 평가 완료!")
        return statistics
    
    def _run_single_episode(self) -> Dict[str, List[float]]:
        """
        단일 에피소드를 실행하고 causal effect를 추정합니다.
        
        Returns:
            Dict mapping reward scenarios to lists of causal effects
        """
        observations = self.env.reset()
        prev_observations = observations.copy()
        
        episode_causal_effects = defaultdict(list)
        done = False
        step_count = 0
        
        while not done and step_count < self.env.max_steps:
            # 현재 관찰을 이전 관찰로 저장
            prev_observations = observations.copy()
            
            # 관찰을 torch tensor로 변환
            torch_observations = {}
            for agent_id, obs in observations.items():
                torch_observations[agent_id] = torch.tensor(obs, dtype=torch.float32, device=self.device)
            
            # 액션 선택 (평가 모드에서는 epsilon-greedy 사용하지 않음)
            actions = self.algorithm.get_actions(torch_observations)
            action_dict = {agent_id: action for agent_id, (action, _) in actions.items()}
            
            # 환경 스텝
            next_observations, rewards, dones, info = self.env.step(action_dict)
            
            # Causal effect 추정
            causal_effects = self._estimate_causal_effects(
                prev_observations, observations, action_dict, rewards
            )
            
            # Reward 시나리오별로 분류하여 저장
            for agent_id, reward in rewards.items():
                other_agent_id = 1 - agent_id  # 2-agent 환경에서 다른 agent
                other_reward = rewards[other_agent_id]
                
                # Reward 시나리오 분류
                if other_reward == 0.0:
                    scenario = 'no_reward'
                elif other_reward == 3.0:
                    scenario = 'forage_reward'
                elif other_reward == 5.0:
                    scenario = 'stag_reward'
                else:
                    continue  # 예상하지 못한 reward 값은 건너뛰기
                
                if agent_id in causal_effects:
                    episode_causal_effects[scenario].append(causal_effects[agent_id])
            
            observations = next_observations
            done = all(dones.values())
            step_count += 1
        
        return episode_causal_effects
    
    def _estimate_causal_effects(
        self,
        prev_observations: Dict[int, np.ndarray],
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float]
    ) -> Dict[int, float]:
        """
        현재 상태에서 각 agent의 causal effect를 추정합니다.
        
        Args:
            prev_observations: 이전 관찰 (w_t)
            observations: 현재 관찰 (z_t)
            actions: 현재 액션
            rewards: 현재 보상
            
        Returns:
            Dict mapping agent_id to causal effect estimate
        """
        causal_effects = {}
        
        with torch.no_grad():
            # 관찰을 torch tensor로 변환
            torch_prev_observations = {}
            torch_observations = {}
            for agent_id in range(self.algorithm.num_agents):
                torch_prev_observations[agent_id] = torch.tensor(
                    prev_observations[agent_id], dtype=torch.float32, device=self.device
                )
                torch_observations[agent_id] = torch.tensor(
                    observations[agent_id], dtype=torch.float32, device=self.device
                )
            
            # 관찰을 텐서로 변환
            w_t = self.algorithm._concatenate_observations(torch_prev_observations)
            z_t = self.algorithm._concatenate_observations(torch_observations)
            
            for agent_id in range(self.algorithm.num_agents):
                # Causal reward shaping 함수를 사용하여 causal effect 추정
                delta_t, _ = self.algorithm.causal_intrinsic_reward(
                    agent_id=agent_id,
                    w_t=w_t,
                    z_t=z_t,
                    o_t=torch_observations[agent_id],
                    current_actions=actions,
                    current_rewards=rewards
                )
                causal_effects[agent_id] = delta_t
        
        return causal_effects
    
    def _compute_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        수집된 causal effect 데이터의 통계를 계산합니다.
        
        Returns:
            Dict containing mean, std, and other statistics for each scenario
        """
        statistics = {}
        
        for scenario, effects in self.causal_effects.items():
            if len(effects) == 0:
                statistics[scenario] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'count': 0,
                    'min': 0.0,
                    'max': 0.0,
                    'median': 0.0
                }
                continue
            
            effects_array = np.array(effects)
            statistics[scenario] = {
                'mean': float(np.mean(effects_array)),
                'std': float(np.std(effects_array)),
                'count': len(effects_array),
                'min': float(np.min(effects_array)),
                'max': float(np.max(effects_array)),
                'median': float(np.median(effects_array))
            }
        
        return statistics
    
    def plot_causal_effects(
        self,
        statistics: Dict[str, Dict[str, float]],
        save_path: Optional[str] = None
    ) -> None:
        """
        Causal effect 추정 결과를 error bar plot으로 시각화합니다.
        
        Args:
            statistics: _compute_statistics()에서 반환된 통계 데이터
            save_path: 저장할 경로 (None이면 표시만)
        """
        # 데이터 준비
        scenarios = list(statistics.keys())
        means = [statistics[scenario]['mean'] for scenario in scenarios]
        stds = [statistics[scenario]['std'] for scenario in scenarios]
        counts = [statistics[scenario]['count'] for scenario in scenarios]

        scenario_labels = {
            'no_reward': 'No Reward (0)',
            'forage_reward': 'Hare Hunting (3)',
            'stag_reward': 'Stag Hunting (5)'
        }
        
        # 샘플 수를 포함한 라벨 생성
        labels = []
        for scenario in scenarios:
            base_label = scenario_labels.get(scenario, scenario)
            sample_count = counts[scenarios.index(scenario)]
            labels.append(f'{base_label}\n(n={sample_count})')
        
        plt.figure(figsize=(6, 4))
        
        # Error bar plot
        x_pos = np.arange(len(scenarios))
        plt.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        
        # 플롯 설정
        plt.xlabel('Reward of the other', fontsize=9)
        plt.ylabel('Estimated Causal Effect (Δ)', fontsize=9)
        plt.title('Causal Effect Estimation', fontsize=11)
        plt.xticks(x_pos, labels, fontsize=9)
        
        # 0선 표시
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 저장 또는 표시
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"Causal effect plot이 저장되었습니다: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def save_results(
        self,
        statistics: Dict[str, Dict[str, float]],
        save_dir: str
    ) -> None:
        """
        평가 결과를 JSON 파일로 저장합니다.
        
        Args:
            statistics: _compute_statistics()에서 반환된 통계 데이터
            save_dir: 저장할 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # 통계 데이터 저장
        stats_path = os.path.join(save_dir, 'causal_effect_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, indent=2, ensure_ascii=False)
        
        # 원시 데이터도 저장 (선택사항)
        raw_data_path = os.path.join(save_dir, 'causal_effect_raw_data.json')
        raw_data = {scenario: effects for scenario, effects in self.causal_effects.items()}
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(raw_data, f, indent=2, ensure_ascii=False)
        
        print(f"평가 결과가 저장되었습니다: {save_dir}")
