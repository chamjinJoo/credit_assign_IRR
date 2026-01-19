# A utility to profile the environment
# Mapping a MARL problem into a space with three axes:
# 1. Agent Scale: log_10(N)
# 2. Alignment: \frac{1}{N} \sum_{i=1}^N \text{Corr}(r^i, \sum_{j \neq i} r^j), Corr is Pearson correlation coefficient
# 3. Interdependency: E_{s, \mathbf{a}^{-i}} [Var_{a^i} (Q_j^{\pi_{\text{rand}}}(s, a^i, \mathbf{a}^{-i}))], under random policy

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import spearmanr
import math
import sys
import os
import sys
import os

_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_script_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from envs.multi_agent_env import StagHuntEnv, SMACEnv, SMACv2Env


class MARLProfiler:
    def __init__(self, env, gamma=0.99):
        self.env = env
        self.n_agents = env.num_agents
        self.gamma = gamma
        self._get_action_dim = lambda: env.get_action_dim() if hasattr(env, 'get_action_dim') else env.action_dim
    
    def _sample_action(self, rng=None):
        """액션 공간에서 무작위 액션 샘플링"""
        action_dim = self._get_action_dim()
        if rng is not None:
            return rng.randint(0, action_dim)
        else:
            return np.random.randint(0, action_dim)

    def calculate_axis1(self):
        """Axis 1: Scale (결정론적 값이므로 오차 없음)"""
        return math.log10(self.n_agents)

    def calculate_axis2_bootstrap(self, reward_history, n_resamples=100):
        """
        Axis 2: Alignment 부트스트래핑
        reward_history를 에피소드 단위로 샘플링하여 분포를 계산합니다.
        """
        # reward_history: (T, N)
        n_steps = reward_history.shape[0]
        bootstrap_means = []

        for _ in range(n_resamples):
            # 중복 허용 무작위 샘플링 (에피소드 성격을 유지하기 위해 행 단위 샘플링)
            indices = np.random.choice(n_steps, n_steps, replace=True)
            resampled_data = reward_history[indices]
            
            alignments = []
            for i in range(self.n_agents):
                r_i = resampled_data[:, i]
                r_others = np.delete(resampled_data, i, axis=1).sum(axis=1)
                
                if np.std(r_i) == 0 or np.std(r_others) == 0:
                    corr = 0
                else:
                    corr, _ = spearmanr(r_i, r_others)
                alignments.append(corr)
            bootstrap_means.append(np.mean(alignments))
            
        return np.mean(bootstrap_means), np.std(bootstrap_means)

    def calculate_axis3_with_dist(self, num_states=10, num_perturbations=10, num_rollouts=20):
        """
        Axis 3: Interdependency 분포 계산
        각 상태와 에이전트 쌍에서 얻은 variance들의 리스트를 반환하여 통계를 냅니다.
        """
        variances = []
        for state_idx in range(num_states):
            # 상태별 시드 설정 (같은 상태는 항상 동일하게)
            state_seed = 10000 + state_idx * 1000
            if hasattr(self.env, 'seed'):
                self.env.seed(state_seed)
            state_rng = np.random.RandomState(state_seed)
            
            self.env.reset()
            initial_state_info = self._capture_state_info()
            
            for j in range(self.n_agents):
                for i in range(self.n_agents):
                    if i == j: continue
                    
                    q_samples = []
                    # base_actions를 시드로 고정
                    base_actions = {k: state_rng.randint(0, self._get_action_dim()) for k in range(self.n_agents)}
                    
                    # 각 perturbation마다 다른 base_seed 사용
                    for pert_idx in range(num_perturbations):
                        # 상태 복원을 위해 시드 설정 후 reset
                        if hasattr(self.env, 'seed'):
                            self.env.seed(state_seed)
                        self.env.reset()
                        self._restore_state_info(initial_state_info)
                        
                        # test_actions[i]도 시드로 고정
                        perturbation_rng = np.random.RandomState(state_seed + pert_idx * 10)
                        test_actions = base_actions.copy()
                        test_actions[i] = perturbation_rng.randint(0, self._get_action_dim())
                        
                        # 각 perturbation마다 고유한 base_seed 사용
                        perturbation_base_seed = state_seed + pert_idx * 100
                        q_val = self._get_mc_q_value(initial_state_info, test_actions, 
                                                    base_seed=perturbation_base_seed, 
                                                    num_rollouts=num_rollouts)
                        q_samples.append(q_val[j])
                    
                    if len(q_samples) > 1:
                        mean_q = np.mean(np.abs(q_samples)) + 1e-8
                        var_val = np.var(q_samples) / (mean_q**2)
                        variances.append(var_val)
    
        return np.mean(variances), np.std(variances) if variances else 0.0

    def profile(self, num_episodes=10, n_bootstrap=50):
        print(f"Profiling with Bootstrapping...")
        a1_mean = self.calculate_axis1()
        
        reward_list = []
        max_steps = getattr(self.env, 'max_steps', 50)
        for ep in range(num_episodes):
            self.env.reset()
            for _ in range(max_steps):
                actions = {i: self._sample_action() for i in range(self.n_agents)}
                _, rs, dones, _ = self.env.step(actions)
                reward_list.append([rs[i] for i in range(self.n_agents)])
                if any(dones.values()): break
        
        a2_mean, a2_std = self.calculate_axis2_bootstrap(np.array(reward_list), n_resamples=n_bootstrap)
        a3_mean, a3_std = self.calculate_axis3_with_dist(num_states=15) # 상태 수를 늘려 샘플 확보
        
        return {
            "Axis 1": (a1_mean, 0), # Scale은 고정값
            "Axis 2": (a2_mean, a2_std),
            "Axis 3": (a3_mean, a3_std)
        }

    def plot_marl_space(self, results_dict, save_name="marl_profile.png"):
        plt.style.use('seaborn-v0_8-white')
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # 기저 축 설정 (Origin radiating style)
        x_max, y_max, z_max = 0.1, 2.0, 2.0 # 축 순서: Align, Dep, Scale
        # Axes3D 고급 설정
        ax.set_box_aspect([1, 1, 1])  # 축 비율
        ax.dist = 12  # 카메라 거리
       
        # 축 범위 
        # X축: Reward Alignment
        # Y축: Interdependency  
        # Z축: Agent Scale
        ax.set_ylim(0, 2)  # Interdependency 범위
        ax.set_zlim(0, 2)    # Agent Scale 범위 (log10(N), N=1~100)

        ax.set_xlabel('Reward Alignment ($\\rho$)', fontsize=11, labelpad=15)
        ax.set_ylabel('Interdependency ($Var(Q)$)', fontsize=11, labelpad=15)
        ax.set_zlabel('Agent Scale ($\\log_{10}N$)', fontsize=11, labelpad=15)

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('lightgray')
        ax.yaxis.pane.set_edgecolor('lightgray')
        ax.zaxis.pane.set_edgecolor('lightgray')
        ax.xaxis.pane.set_alpha(0.1)

        ax.grid(True, linestyle='--', alpha=0.3)
        ax.view_init(elev=20, azim=60)

        from matplotlib import colormaps
        colors = colormaps.get_cmap('viridis').resampled(len(results_dict))
        
        for i, (name, res) in enumerate(results_dict.items()):
            # 결과 구조: (mean, std)
            a2_m, a2_s = res["Axis 2"]
            a3_m, a3_s = res["Axis 3"]
            a1_m, a1_s = res["Axis 1"]
            color = colors(i)

            # 1. 메인 포인트
            ax.scatter(a2_m, a3_m, a1_m, s=150, color=color, alpha=0.9, edgecolors='w', zorder=10, label=name)

            # 2. 3D 에러 바 (Confidence Intervals)
            # Axis 2 error (Horizontal X)
            ax.plot([a2_m - a2_s, a2_m + a2_s], [a3_m, a3_m], [a1_m, a1_m], color=color, linewidth=2, alpha=0.6)
            # Axis 3 error (Horizontal Y)
            ax.plot([a2_m, a2_m], [a3_m - a3_s, a3_m + a3_s], [a1_m, a1_m], color=color, linewidth=2, alpha=0.6)
            
            # 3. 바닥 투영 및 투영선
            ax.plot([a2_m, a2_m], [a3_m, a3_m], [0, a1_m], color=color, linestyle='--', alpha=0.3)
            ax.text(a2_m, a3_m, a1_m + 0.1, name, fontsize=9, fontweight='bold', ha='center')

        ax.view_init(elev=20, azim=45)
        plt.title('MARL Problem Space (with Bootstrapped Confidence)\n', fontsize=15, fontweight='bold')
        plt.grid()
        plt.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95),
                frameon=True, facecolor='white', framealpha=0.8)
        plt.tight_layout()
        plt.savefig(save_name, dpi=300)
        plt.show()

    def _get_mc_q_value(self, initial_state_info, joint_action, num_rollouts=10, horizon=20, base_seed=None):
        """
        Monte Carlo Q-value 추정
        """
        returns = np.zeros(self.n_agents)
        
        if base_seed is None:
            base_seed = np.random.randint(0, 2**31)
        
        for r_idx in range(num_rollouts):
            # 시드 설정 후 reset (상태 복원을 위해)
            rollout_seed = base_seed + r_idx
            if hasattr(self.env, 'seed'):
                self.env.seed(rollout_seed)
            action_rng = np.random.RandomState(rollout_seed)
            
            # 환경 상태 복구
            self.env.reset()
            if initial_state_info is not None:
                self._restore_state_info(initial_state_info)
            
            total_r = np.zeros(self.n_agents)
            
            # 1-step: 고정된 joint_action 수행
            _, rs, _, _ = self.env.step(joint_action)
            for i, r in rs.items(): total_r[i] += r
            
            # 나머지 steps: 고정된 시드 하에서 발생하는 Random Policy Rollout
            curr_gamma = self.gamma
            for t in range(horizon):
                random_actions = {i: self._sample_action(action_rng) for i in range(self.n_agents)}
                _, rs, dones, _ = self.env.step(random_actions)
                for i, r in rs.items(): total_r[i] += curr_gamma * r
                curr_gamma *= self.gamma
                if any(dones.values()): break
                
            returns += total_r
            
        return returns / num_rollouts
    
    def _capture_state_info(self):
        """환경의 현재 상태 정보를 캡처 (가능한 경우)"""
        # StagHuntEnv의 경우 내부 상태를 직접 접근할 수 없으므로
        # None을 반환하고 reset을 통해 상태를 관리
        if hasattr(self.env, 'agent_positions') and hasattr(self.env, 'stag_positions'):
            return {
                'agent_positions': self.env.agent_positions.copy() if hasattr(self.env, 'agent_positions') else None,
                'stag_positions': self.env.stag_positions.copy() if hasattr(self.env, 'stag_positions') else None,
                'stag_alive': self.env.stag_alive.copy() if hasattr(self.env, 'stag_alive') else None,
                'hare_positions': self.env.hare_positions.copy() if hasattr(self.env, 'hare_positions') else None,
                'hare_alive': self.env.hare_alive.copy() if hasattr(self.env, 'hare_alive') else None,
            }
        return None
    
    def _restore_state_info(self, state_info):
        """캡처한 상태 정보를 환경에 복원 (가능한 경우)"""
        if state_info is None:
            return
        
        # StagHuntEnv의 경우 내부 상태를 직접 복원
        if hasattr(self.env, 'agent_positions') and state_info.get('agent_positions') is not None:
            self.env.agent_positions = state_info['agent_positions'].copy()
        if hasattr(self.env, 'stag_positions') and state_info.get('stag_positions') is not None:
            self.env.stag_positions = state_info['stag_positions'].copy()
        if hasattr(self.env, 'stag_alive') and state_info.get('stag_alive') is not None:
            self.env.stag_alive = state_info['stag_alive'].copy()
        if hasattr(self.env, 'hare_positions') and state_info.get('hare_positions') is not None:
            self.env.hare_positions = state_info['hare_positions'].copy()
        if hasattr(self.env, 'hare_alive') and state_info.get('hare_alive') is not None:
            self.env.hare_alive = state_info['hare_alive'].copy()

if __name__ == "__main__":
    env_configs = [
        {   "name": "StagHuntEnv (N=2 with shared reward)",
            "class": StagHuntEnv,
            "kwargs": {"num_agents": 2, "coop_bonus": 5.0}  },

        {   "name": "StagHuntEnv (N=2)",
            "class": StagHuntEnv,
            "kwargs": {"num_agents": 2, "coop_bonus": 0.0}  },
        {
            "name": "StagHuntEnv (N=3 with shared reward)",
            "class": StagHuntEnv,
            "kwargs": {"num_agents": 3, "coop_bonus": 5.0}  },
        {
            "name": "StagHuntEnv (N=3)",
            "class": StagHuntEnv,
            "kwargs": {"num_agents": 3, "coop_bonus": 0.0}  },
    ]
    
    all_results = {}
    num_episodes = 1000
    
    print("=" * 60)
    print("Multiple Environment Profiling")
    print("=" * 60)
    
    # 순차적으로 각 환경 프로파일링
    for i, env_config in enumerate(env_configs, 1):
        env_name = env_config["name"]
        env_class = env_config["class"]
        env_kwargs = env_config["kwargs"]
        
        print(f"\n[{i}/{len(env_configs)}] Profiling {env_name}...")
        print(f"  Config: {env_kwargs}")
        
        try:
            # 환경 생성
            env = env_class(**env_kwargs)
            
            # 프로파일러 생성 및 실행
            profiler = MARLProfiler(env)
            results = profiler.profile(num_episodes=num_episodes)
            
            # 결과 저장
            all_results[env_name] = results
            
            # 개별 결과 출력
            print(f"  Results:")
            a1_mean, a1_std = results['Axis 1']
            a2_mean, a2_std = results['Axis 2']
            a3_mean, a3_std = results['Axis 3']
            print(f"    Axis 1 (Scale): {a1_mean:.4f} ± {a1_std:.4f}")
            print(f"    Axis 2 (Alignment): {a2_mean:.4f} ± {a2_std:.4f}")
            print(f"    Axis 3 (Dependency): {a3_mean:.4f} ± {a3_std:.4f}")
            
        except Exception as e:
            print(f"  ERROR: Failed to profile {env_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 전체 결과 요약
    print("\n" + "=" * 60)
    print("Summary of All Results")
    print("=" * 60)
    for env_name, results in all_results.items():
        print(f"\n{env_name}:")
        a1_mean, a1_std = results['Axis 1']
        a2_mean, a2_std = results['Axis 2']
        a3_mean, a3_std = results['Axis 3']
        print(f"  Scale: {a1_mean:.4f} ± {a1_std:.4f}")
        print(f"  Alignment: {a2_mean:.4f} ± {a2_std:.4f}")
        print(f"  Dependency: {a3_mean:.4f} ± {a3_std:.4f}")
    
    # 3D plot으로 비교
    if all_results:
        print("\n" + "=" * 60)
        print("Generating 3D comparison plot...")
        # 마지막으로 성공한 환경의 profiler 사용 (plot 메서드만 필요)
        profiler = MARLProfiler(env)  # 마지막 환경 사용
        profiler.plot_marl_space(all_results)
        print("Done.")
        print("=" * 60)
    else:
        print("\nNo results to plot.")