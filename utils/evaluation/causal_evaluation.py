import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
import os
import json
from collections import defaultdict

if TYPE_CHECKING:
    from algorithms.pci_mappo import PCIMAPPOAlgorithm
    from envs.multi_agent_env import StagHuntEnv


class CausalEffectEvaluator:
    """
    Bridge network의 causal effect 추정 성능을 평가하는 클래스.
    
    StagHunt 환경에서 다른 agent의 reward 시나리오에 따른 causal effect를 추정하고 시각화합니다.
    n-agent 환경을 지원하며, 협력 사냥 정보를 활용하여 정확한 시나리오 분류를 수행합니다.
    
    Scenarios:
        - 'no_reward': 다른 agent가 보상을 받지 못하는 경우
        - 'forage_reward': 다른 agent가 hare를 사냥하는 경우 (협력 없음)
        - 'stag_reward': 현재 agent가 협력에 참여해서 다른 agent가 stag reward를 받는 경우
    """
    
    def __init__(
        self,
        algorithm: "PCIMAPPOAlgorithm",
        env: "StagHuntEnv",
        num_eval_steps: int = 1000,
        device: str = "cpu"
    ):
        self.algorithm = algorithm
        self.env = env
        self.device = device
        
        # StagHunt 환경의 reward 시나리오 (참고용)
        # 실제 분류는 algorithm._classify_delta_scenario()를 통해 수행
        self.reward_scenarios = {
            'no_reward': 0.0,      # 다른 agent가 보상을 받지 못하는 경우
            'forage_reward': 3.0,  # 다른 agent가 hare를 사냥 (협력 없음)
            'stag_reward': 5.0     # 협력해서 다른 agent가 stag reward 획득
        }
        
        # 결과 저장용
        self.causal_effects = defaultdict(list)
        self.episode_data = []
        # Observation 저장용 (t-SNE 분석을 위해)
        self.observation_data = defaultdict(lambda: {'observations': [], 'scenarios': [], 'huntable_cases': []})
        
        
    def evaluate_causal_effects(self, num_eval_steps: int = 1000) -> Dict[str, Dict[str, float]]:
        """
        지정된 수의 환경 상호작용을 실행하여 causal effect 추정치의 통계를 계산합니다.
        
        Returns:
            Dict containing statistics for each reward scenario
        """
        print(f"Causal effect 평가 시작: {num_eval_steps} 스텝")
        
        # 알고리즘을 평가 모드로 설정
        self.algorithm.outcome_bridge_net.eval()
        self.algorithm.action_bridge_net.eval()
        
        # 환경 초기화
        observations = self.env.reset()
        prev_observations = {agent_id: np.zeros_like(obs) for agent_id, obs in observations.items()}
        
        step_count = 0
        episode_count = 0
        
        while step_count < num_eval_steps:
            if step_count % 1000 == 0:
                print(f"스텝 {step_count}/{num_eval_steps} 진행 중... (에피소드 {episode_count})")
            
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
            # Retrieve hidden states from algorithm (populated by get_actions)
            hidden_states = getattr(self.algorithm, 'temp_hidden_states', None)
            causal_effects = self._estimate_causal_effects(prev_observations, observations, action_dict, rewards, info, hidden_states)
            
            transition = {
                'bold_r_t': rewards,
                'info': info
            }
            
            for agent_id in causal_effects.keys():
                # 정확한 시나리오 분류를 위해 algorithm의 메서드 사용
                scenario = self.algorithm._classify_delta_scenario(transition, agent_id)
                self.causal_effects[scenario].append(causal_effects[agent_id])
                
                # Huntable objects case 분류
                huntable_case = self._classify_huntable_case(agent_id, info)
                
                # Observation 데이터 저장 (t-SNE 분석용)
                self.observation_data[agent_id]['observations'].append(observations[agent_id].copy())
                self.observation_data[agent_id]['scenarios'].append(scenario)
                self.observation_data[agent_id]['huntable_cases'].append(huntable_case)
            
            # 상태 업데이트
            prev_observations = observations
            observations = next_observations
            step_count += 1
            
            # 에피소드가 끝났으면 새 에피소드 시작
            if all(dones.values()):
                observations = self.env.reset()
                prev_observations = {agent_id: np.zeros_like(obs) for agent_id, obs in observations.items()}
                episode_count += 1
        
        # 통계 계산
        statistics = self._compute_statistics()
        
        print(f"Causal effect 평가 완료! 총 {step_count} 스텝, {episode_count} 에피소드 실행")
        return statistics
    
    
    def _estimate_causal_effects(
        self,
        prev_observations: Dict[int, np.ndarray],
        observations: Dict[int, np.ndarray],
        actions: Dict[int, int],
        rewards: Dict[int, float],
        info: Dict[str, Any],
        hidden_states: Optional[Dict[int, Tuple]] = None
    ) -> Dict[int, float]:
        """
        현재 상태에서 각 agent의 causal effect를 추정합니다.
        
        Args:
            prev_observations: 이전 관찰 (o_{t-1})
            observations: 현재 관찰 (o_t)
            actions: 현재 액션 (a_t)
            rewards: 현재 보상 (r_t)
            info: 환경에서 반환한 추가 정보 (협력 사냥 정보, 사냥 가능한 objects 등)
            
        Returns:
            Dict mapping agent_id to causal effect estimate
        """
        causal_effects = {}
        
        # Huntable objects 정보 추출
        huntable_objects = info.get('huntable_objects', {})
        huntable_stags = huntable_objects.get('huntable_stags', {})
        huntable_hares = huntable_objects.get('huntable_hares', [])
        
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
            z_t = self.algorithm._concatenate_observations(torch_prev_observations)
            w_t = self.algorithm._concatenate_observations(torch_observations)
            
            for agent_id in range(self.algorithm.num_agents):
                # Causal reward shaping 함수를 사용하여 causal effect 추정
                delta_t, _ = self.algorithm.causal_intrinsic_reward(
                    agent_id=agent_id,
                    w_t=w_t,
                    z_t=z_t,
                    o_t=torch_observations[agent_id],
                    current_actions=actions,
                    current_rewards=rewards,
                    hidden_states=hidden_states
                )
                # Huntable objects 정보를 활용한 추가 분석
                causal_effects[agent_id] = delta_t
        return causal_effects
    
    def _classify_huntable_case(self, agent_id: int, info: Dict[str, Any]) -> str:
        """
        Agent의 huntable objects에 따라 case를 분류합니다.
        
        Args:
            agent_id: Agent ID
            info: 환경 정보 (huntable_objects 포함)
            
        Returns:
            Huntable case 문자열 (예: 'my_stag', 'my_stag+coop_stag', 'hare', 'none' 등)
        """
        huntable_objects = info.get('huntable_objects', {})
        huntable_stags = huntable_objects.get('huntable_stags', {})
        huntable_hares = huntable_objects.get('huntable_hares', [])
        
        # 이 agent가 사냥 가능한 objects 식별
        my_stag_id = None
        coop_stag_ids = []
        hare_ids = []
        
        # My stag (agent_id == stag_id)
        if agent_id in huntable_stags:
            my_stag_id = agent_id
        
        # Coop stags (agent_id in required_agents but stag_id != agent_id)
        for stag_id, stag_info in huntable_stags.items():
            if stag_id != agent_id and agent_id in stag_info['required_agents']:
                coop_stag_ids.append(stag_id)
        
        # Hares (agent가 사냥 가능한 hare들)
        for hare_info in huntable_hares:
            if agent_id in hare_info['hunting_agents']:
                hare_ids.append(hare_info['hare_id'])
        
        # Object 조합 key 생성
        objects_present = []
        if my_stag_id is not None:
            objects_present.append('my_stag')
        if coop_stag_ids:
            objects_present.append('coop_stag')
        if hare_ids:
            objects_present.append('hare')
        
        if not objects_present:
            return 'none'
        
        return '+'.join(objects_present)
    
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
        save_path: Optional[str] = None,
        return_figure: bool = False
    ) -> Optional[plt.Figure]:
        """
        Causal effect 추정 결과를 error bar plot으로 시각화합니다.
        
        Args:
            statistics: _compute_statistics()에서 반환된 통계 데이터
            save_path: 저장할 경로 (None이면 저장하지 않음)
            return_figure: True이면 Figure 객체를 반환 (wandb 로깅용)
            
        Returns:
            plt.Figure 객체 (return_figure=True일 때만)
        """
        # 데이터 준비
        scenarios = list(statistics.keys())
        means = [statistics[scenario]['mean'] for scenario in scenarios]
        stds = [statistics[scenario]['std'] for scenario in scenarios]
        counts = [statistics[scenario]['count'] for scenario in scenarios]

        scenario_labels = {
            'no_reward': 'No Reward (0)',
            'forage_reward': 'Hare Hunting (3)',
            'stag_reward': 'Cooperative Stag (5)'
        }
        
        # 샘플 수를 포함한 라벨 생성
        labels = []
        for scenario in scenarios:
            base_label = scenario_labels.get(scenario, scenario)
            sample_count = counts[scenarios.index(scenario)]
            labels.append(f'{base_label}\n(n={sample_count})')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Error bar plot
        x_pos = np.arange(len(scenarios))
        ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
        
        # 플롯 설정
        ax.set_xlabel('Reward of the other', fontsize=9)
        ax.set_ylabel('Estimated Causal Effect (Δ)', fontsize=9)
        ax.set_title('Causal Effect Estimation', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        
        # 0선 표시
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # 저장
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
            print(f"Causal effect plot이 저장되었습니다: {save_path}")
        
        # Figure 객체 반환 또는 표시
        if return_figure:
            return fig
        else:
            plt.show()
            plt.close()
            return None
    
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
    
    def plot_tsne_observations(
        self,
        save_path: Optional[str] = None,
        perplexity: int = 60,
        use_huntable_cases: bool = True
    ):
        """
        수집된 observation 데이터를 t-SNE로 시각화합니다.
        
        Args:
            save_path: 저장 경로
            perplexity: t-SNE perplexity 파라미터
            use_huntable_cases: huntable_cases를 사용하여 색상 구분할지 여부
            
        Returns:
            Separability scores
        """
        from .observation_analysis import create_tsne_plots
        
        return create_tsne_plots(
            self,
            self.env,
            save_path=save_path,
            perplexity=perplexity,
            use_huntable_cases=use_huntable_cases
        )
    
    def plot_tsne_value_heatmap(
        self,
        save_path: Optional[str] = None,
        perplexity: int = 60
    ):
        """
        Value network 반응을 t-SNE heatmap으로 시각화합니다.
        
        Args:
            save_path: 저장 경로
            perplexity: t-SNE perplexity 파라미터
            
        Returns:
            Separability scores
        """
        from .observation_analysis import create_tsne_plot_from_evaluator
        
        return create_tsne_plot_from_evaluator(
            self,
            self.env,
            save_path=save_path,
            perplexity=perplexity,
            use_huntable_cases=True
        )
    
    def plot_interactive_tsne_value_heatmap(
        self,
        save_path: Optional[str] = None,
        perplexity: int = 30
    ):
        """
        Value network 반응을 interactive t-SNE heatmap으로 시각화합니다.
        
        Args:
            save_path: 저장 경로
            perplexity: t-SNE perplexity 파라미터
            
        Returns:
            HTML 파일 경로
        """
        from .observation_analysis import ObservationAnalyzer
        
        # Observation 데이터 확인
        if not hasattr(self, 'observation_data') or len(self.observation_data) == 0:
            print("Error: evaluate_causal_effects()를 먼저 실행해야 합니다.")
            return None
        
        # Value networks 추출 (개별 네트워크와 parameter sharing 모두 지원)
        from ..utils.value_network_utils import get_value_networks_dict
        value_networks = get_value_networks_dict(self.algorithm)
        
        # ObservationAnalyzer 생성
        analyzer = ObservationAnalyzer(
            num_agents=self.algorithm.num_agents,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42
        )
        
        return analyzer.plot_interactive_tsne_value_heatmap(
            self.observation_data,
            value_networks,
            save_path=save_path,
            perplexity=perplexity
        )
    
    def plot_interactive_tsne(
        self,
        save_path: Optional[str] = None,
        perplexity: int = 60,
        n_clusters: Optional[int] = None
    ):
        """
        Interactive t-SNE plot을 생성합니다 (HTML).
        Cluster 중심의 representative observation을 렌더링하여 표시합니다.
        
        Args:
            save_path: 저장 경로 (.html)
            perplexity: t-SNE perplexity 파라미터
        """
        from .observation_analysis import ObservationAnalyzer
        
        if len(self.observation_data) == 0:
            print("Error: evaluate_causal_effects()를 먼저 실행해야 합니다.")
            return
        
        analyzer = ObservationAnalyzer(
            num_agents=self.algorithm.num_agents,
            perplexity=perplexity,
            max_iter=1000,
            random_state=42
        )
        
        analyzer.plot_interactive_tsne(
            self.observation_data,
            self.env,
            save_path=save_path,
            n_clusters=n_clusters
        )