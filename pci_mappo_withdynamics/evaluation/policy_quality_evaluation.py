import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class PolicyQualityEvaluator:
    """
    학습된 policy의 품질을 정량화하여 causal learning의 효과를 검증하는 클래스.
    """
    
    def __init__(self, algorithm, env):
        self.algorithm = algorithm
        self.env = env
        self.num_agents = env.num_agents
        
        # Stag-Hunt 환경 특성
        self.stag_reward = env.stag_reward
        self.forage_reward = env.forage_reward
        self.required_agents = env.required_agents
        
    def evaluate_policy_quality_from_data(
        self, 
        causal_evaluation_data: Dict,
        advanced_causal_data: Dict
    ) -> Dict[str, Any]:
        """
        기존 causal evaluation 데이터를 활용하여 policy 품질을 평가합니다.
        
        Args:
            causal_evaluation_data: CausalEffectEvaluator에서 수집된 데이터
            advanced_causal_data: AdvancedCausalEvaluator에서 수집된 데이터
            
        Returns:
            Policy quality metrics
        """
        print("Policy Quality Evaluation 시작...")
        
        # 1. 협력 행동 품질 평가
        cooperation_metrics = self._evaluate_cooperation_quality(advanced_causal_data)
        
        # 2. 전략적 사고 능력 평가
        # strategic_metrics = self._evaluate_strategic_thinking(advanced_causal_data)
        
        # 3. 상황별 적응력 평가
        # adaptation_metrics = self._evaluate_adaptation_ability(advanced_causal_data)
        
        # 4. 정책 일관성 평가
        # consistency_metrics = self._evaluate_policy_consistency(advanced_causal_data)
        
        results = {
            'cooperation_quality': cooperation_metrics,
            # 'strategic_thinking': strategic_metrics,
            # 'adaptation_ability': adaptation_metrics,
            # 'policy_consistency': consistency_metrics,
            # 'overall_quality_score': self._calculate_overall_quality_score(
            #     cooperation_metrics, strategic_metrics, adaptation_metrics, consistency_metrics
            # )
        }
        
        print("Policy Quality Evaluation 완료!")
        return results
    
    def _evaluate_cooperation_quality(self, advanced_causal_data: Dict) -> Dict[str, Any]:
        """
        Huntable objects 기반 case-by-case 협력 행동 품질을 평가합니다.
        
        7가지 huntable case:
        1. my_stag only
        2. coop_stag only  
        3. hare only
        4. my_stag + coop_stag
        5. my_stag + hare
        6. coop_stag + hare
        7. my_stag + coop_stag + hare
        """
        policy_stats = advanced_causal_data.get('policy_bias', {})
        
        # 7가지 case별 선택 비율 저장
        case_analysis = {
            'my_stag_only': {'my_stag': 0.0, 'none': 0.0, 'count': 0},
            'coop_stag_only': {'coop_stag': 0.0, 'none': 0.0, 'count': 0},
            'hare_only': {'hare': 0.0, 'none': 0.0, 'count': 0},
            'my_stag_coop_stag': {'my_stag': 0.0, 'coop_stag': 0.0, 'none': 0.0, 'count': 0},
            'my_stag_hare': {'my_stag': 0.0, 'hare': 0.0, 'none': 0.0, 'count': 0},
            'coop_stag_hare': {'coop_stag': 0.0, 'hare': 0.0, 'none': 0.0, 'count': 0},
            'my_stag_coop_stag_hare': {'my_stag': 0.0, 'coop_stag': 0.0, 'hare': 0.0, 'none': 0.0, 'count': 0}
        }
        
        # Agent별 데이터 수집
        for agent_key, agent_data in policy_stats.items():
            for case_key, stats in agent_data.items():
                choice_dist = stats['choice_distribution']  # [my_stag, coop_stag, hare, none]
                total_count = stats['total_count']
                
                if total_count == 0:
                    continue
                
                # Case 분류 및 데이터 누적
                case_type = self._classify_huntable_case(case_key)
                if case_type in case_analysis:
                    case_analysis[case_type]['count'] += total_count
                    
                    # 선택 비율 누적 (가중 평균을 위해)
                    for i, choice_name in enumerate(['my_stag', 'coop_stag', 'hare', 'none']):
                        if choice_name in case_analysis[case_type]:
                            case_analysis[case_type][choice_name] += choice_dist[i] * total_count
        
        # 가중 평균으로 최종 비율 계산
        cooperation_metrics = {}
        for case_type, data in case_analysis.items():
            if data['count'] > 0:
                case_metrics = {}
                for choice_name in ['my_stag', 'coop_stag', 'hare', 'none']:
                    if choice_name in data:
                        case_metrics[f'{case_type}_{choice_name}_ratio'] = data[choice_name] / data['count']
                case_metrics[f'{case_type}_total_count'] = data['count']
                cooperation_metrics.update(case_metrics)
        
        # 전체 요약 지표 계산
        cooperation_metrics.update(self._calculate_summary_metrics(case_analysis))
        
        return cooperation_metrics
    
    def _classify_huntable_case(self, case_key: str) -> str:
        """
        Case key를 7가지 huntable case로 분류합니다.
        """
        has_my_stag = 'my_stag' in case_key
        has_coop_stag = 'coop_stag' in case_key
        has_hare = 'hare' in case_key
        
        if has_my_stag and has_coop_stag and has_hare:
            return 'my_stag_coop_stag_hare'
        elif has_my_stag and has_coop_stag:
            return 'my_stag_coop_stag'
        elif has_my_stag and has_hare:
            return 'my_stag_hare'
        elif has_coop_stag and has_hare:
            return 'coop_stag_hare'
        elif has_my_stag:
            return 'my_stag_only'
        elif has_coop_stag:
            return 'coop_stag_only'
        elif has_hare:
            return 'hare_only'
        else:
            return 'no_huntable'  # 제외
    
    def _calculate_summary_metrics(self, case_analysis: Dict) -> Dict[str, float]:
        """
        Case별 분석에서 전체 요약 지표를 계산합니다.
        """
        summary_metrics = {}
        
        # 1. 전체 Stag 선호도 (stag vs hare 경쟁 상황에서)
        stag_vs_hare_cases = ['my_stag_hare', 'coop_stag_hare', 'my_stag_coop_stag_hare']
        total_stag_choices = 0
        total_hare_choices = 0
        
        for case_type in stag_vs_hare_cases:
            if case_type in case_analysis and case_analysis[case_type]['count'] > 0:
                data = case_analysis[case_type]
                if 'my_stag' in data:
                    total_stag_choices += data['my_stag']
                if 'coop_stag' in data:
                    total_stag_choices += data['coop_stag']
                if 'hare' in data:
                    total_hare_choices += data['hare']
        
        if total_stag_choices + total_hare_choices > 0:
            summary_metrics['overall_stag_preference'] = total_stag_choices / (total_stag_choices + total_hare_choices)
        
        # 2. 협력 일관성 (my_stag vs coop_stag 선택)
        stag_vs_stag_cases = ['my_stag_coop_stag', 'my_stag_coop_stag_hare']
        total_my_stag_choices = 0
        total_coop_stag_choices = 0
        
        for case_type in stag_vs_stag_cases:
            if case_type in case_analysis and case_analysis[case_type]['count'] > 0:
                data = case_analysis[case_type]
                if 'my_stag' in data:
                    total_my_stag_choices += data['my_stag']
                if 'coop_stag' in data:
                    total_coop_stag_choices += data['coop_stag']
        
        if total_my_stag_choices + total_coop_stag_choices > 0:
            summary_metrics['my_stag_consistency'] = total_my_stag_choices / (total_my_stag_choices + total_coop_stag_choices)
        
        # 3. 배신 회피 (hare 선택 비율이 낮을수록 좋음)
        total_choices = 0
        total_hare_choices = 0
        
        for case_type, data in case_analysis.items():
            if data['count'] > 0:
                total_choices += data['count']
                if 'hare' in data:
                    total_hare_choices += data['hare']
        
        if total_choices > 0:
            summary_metrics['defection_avoidance'] = 1.0 - (total_hare_choices / total_choices)
        
        # 4. 파트너 협조 (coop_stag 선택 비율)
        total_coop_stag_choices = 0
        total_choices_with_coop = 0
        
        for case_type, data in case_analysis.items():
            if 'coop_stag' in data and data['count'] > 0:
                total_choices_with_coop += data['count']
                total_coop_stag_choices += data['coop_stag']
        
        if total_choices_with_coop > 0:
            summary_metrics['partner_coordination'] = total_coop_stag_choices / total_choices_with_coop
        
        return summary_metrics
    
    def _evaluate_strategic_thinking(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        전략적 사고 능력을 평가합니다.
        """
        causal_stats = advanced_causal_data.get('causal_effects', {})
        
        strategic_metrics = {
            'scenario_discrimination': 0.0,
            'cooperative_effect_learning': 0.0,
            'defection_detection': 0.0,
            'strategic_consistency': 0.0
        }
        
        for agent_key, agent_data in causal_stats.items():
            agent_metrics = {
                'scenario_discrimination': [],
                'cooperative_effect_learning': [],
                'defection_detection': [],
                'strategic_consistency': []
            }
            
            # 시나리오별 causal effect 분석
            scenario_effects = {}
            for scenario, stats in agent_data.items():
                mean_effect = stats['mean']
                scenario_effects[scenario] = mean_effect
                
                # 시나리오 구분 능력 평가
                if 'cooperative' in scenario:
                    agent_metrics['cooperative_effect_learning'].append(abs(mean_effect))
                elif 'defection' in scenario or 'hare' in scenario:
                    agent_metrics['defection_detection'].append(abs(mean_effect))
            
            # 1. 시나리오 구분 능력 (다른 시나리오 간 causal effect 차이)
            if len(scenario_effects) > 1:
                effects = list(scenario_effects.values())
                scenario_variance = np.var(effects)
                agent_metrics['scenario_discrimination'].append(scenario_variance)
            
            # 2. 전략적 일관성 (같은 유형 시나리오에서의 일관된 반응)
            cooperative_effects = [effect for scenario, effect in scenario_effects.items() 
                                 if 'cooperative' in scenario]
            if len(cooperative_effects) > 1:
                consistency = 1.0 / (1.0 + np.std(cooperative_effects))
                agent_metrics['strategic_consistency'].append(consistency)
            
            # Agent별 평균 계산
            for key in agent_metrics:
                if agent_metrics[key]:
                    strategic_metrics[key] += np.mean(agent_metrics[key])
        
        # 전체 평균으로 정규화
        num_agents = len(causal_stats)
        for key in strategic_metrics:
            strategic_metrics[key] /= num_agents if num_agents > 0 else 1
        
        return strategic_metrics
    
    def _evaluate_adaptation_ability(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        상황별 적응력을 평가합니다.
        """
        policy_stats = advanced_causal_data.get('policy_bias', {})
        
        adaptation_metrics = {
            'object_combination_handling': 0.0,
            'complex_situation_adaptation': 0.0,
            'decision_diversity': 0.0,
            'context_sensitivity': 0.0
        }
        
        for agent_key, agent_data in policy_stats.items():
            agent_metrics = {
                'object_combination_handling': [],
                'complex_situation_adaptation': [],
                'decision_diversity': [],
                'context_sensitivity': []
            }
            
            case_complexity = []
            decision_entropies = []
            
            for case_key, stats in agent_data.items():
                choice_dist = stats['choice_distribution']
                total_count = stats['total_count']
                
                # 1. 객체 조합 처리 능력 (복잡한 조합에서의 성능)
                object_count = len(case_key.split('+'))
                case_complexity.append(object_count)
                
                # 복잡한 상황에서의 적응력 (3개 이상 객체 조합)
                if object_count >= 3:
                    # 복잡한 상황에서 none 선택 비율이 낮을수록 좋음
                    adaptation_score = 1.0 - choice_dist[3]  # 1 - none_ratio
                    agent_metrics['complex_situation_adaptation'].append(adaptation_score)
                
                # 2. 의사결정 다양성 (엔트로피 기반)
                # 0이 아닌 선택 확률들만 고려
                non_zero_probs = [p for p in choice_dist if p > 0]
                if len(non_zero_probs) > 1:
                    entropy = -sum(p * np.log2(p) for p in non_zero_probs)
                    decision_entropies.append(entropy)
                    agent_metrics['decision_diversity'].append(entropy)
                
                # 3. 맥락 민감성 (객체 조합에 따른 선택 변화)
                if total_count > 10:  # 충분한 데이터가 있는 경우만
                    # 가장 선호하는 선택의 비율
                    max_choice_ratio = max(choice_dist)
                    context_sensitivity = 1.0 - max_choice_ratio  # 완전히 고정되지 않았음을 의미
                    agent_metrics['context_sensitivity'].append(context_sensitivity)
            
            # 4. 객체 조합 처리 능력 (복잡도에 따른 성능 유지)
            if case_complexity:
                complexity_scores = []
                for i, complexity in enumerate(case_complexity):
                    case_key = list(agent_data.keys())[i]
                    stats = agent_data[case_key]
                    # 복잡한 상황에서도 적절한 선택을 하는지
                    none_ratio = stats['choice_distribution'][3]
                    complexity_score = 1.0 - none_ratio  # none 선택이 적을수록 좋음
                    complexity_scores.append(complexity_score)
                
                # 복잡도와 성능의 상관관계 (음의 상관관계가 적을수록 좋음)
                if len(complexity_scores) > 2:
                    correlation = np.corrcoef(case_complexity, complexity_scores)[0, 1]
                    adaptation_score = 1.0 - abs(correlation)  # 상관관계가 낮을수록 좋음
                    agent_metrics['object_combination_handling'].append(adaptation_score)
            
            # Agent별 평균 계산
            for key in agent_metrics:
                if agent_metrics[key]:
                    adaptation_metrics[key] += np.mean(agent_metrics[key])
        
        # 전체 평균으로 정규화
        num_agents = len(policy_stats)
        for key in adaptation_metrics:
            adaptation_metrics[key] /= num_agents if num_agents > 0 else 1
        
        return adaptation_metrics
    
    def _evaluate_policy_consistency(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        정책의 일관성을 평가합니다.
        """
        policy_stats = advanced_causal_data.get('policy_bias', {})
        
        consistency_metrics = {
            'decision_stability': 0.0,
            'scenario_consistency': 0.0,
            'agent_consistency': 0.0,
            'overall_reliability': 0.0
        }
        
        # Agent 간 일관성
        agent_choice_patterns = {}
        for agent_key, agent_data in policy_stats.items():
            agent_patterns = {}
            for case_key, stats in agent_data.items():
                if stats['total_count'] > 5:  # 충분한 데이터가 있는 경우만
                    most_frequent_choice = stats['most_frequent_choice']
                    agent_patterns[case_key] = most_frequent_choice
            agent_choice_patterns[agent_key] = agent_patterns
        
        # 1. Agent 간 일관성 (같은 상황에서 비슷한 선택을 하는지)
        if len(agent_choice_patterns) > 1:
            common_cases = set.intersection(*[set(patterns.keys()) for patterns in agent_choice_patterns.values()])
            if common_cases:
                consistency_scores = []
                for case in common_cases:
                    choices = [patterns[case] for patterns in agent_choice_patterns.values()]
                    # 선택의 일관성 (같은 선택을 하는 비율)
                    most_common_choice = max(set(choices), key=choices.count)
                    consistency = choices.count(most_common_choice) / len(choices)
                    consistency_scores.append(consistency)
                consistency_metrics['agent_consistency'] = np.mean(consistency_scores)
        
        # 2. 시나리오별 일관성 (같은 시나리오에서 일관된 선택)
        for agent_key, agent_data in policy_stats.items():
            scenario_consistency = []
            for case_key, stats in agent_data.items():
                if stats['total_count'] > 10:
                    # 가장 빈번한 선택의 비율이 높을수록 일관성 있음
                    most_frequent_ratio = max(stats['choice_distribution'])
                    scenario_consistency.append(most_frequent_ratio)
            
            if scenario_consistency:
                consistency_metrics['scenario_consistency'] += np.mean(scenario_consistency)
        
        # 3. 의사결정 안정성 (전체적인 선택 패턴의 안정성)
        for agent_key, agent_data in policy_stats.items():
            stability_scores = []
            for case_key, stats in agent_data.items():
                if stats['total_count'] > 5:
                    # 선택 분포의 표준편차가 낮을수록 안정적
                    choice_dist = stats['choice_distribution']
                    stability = 1.0 - np.std(choice_dist)
                    stability_scores.append(stability)
            
            if stability_scores:
                consistency_metrics['decision_stability'] += np.mean(stability_scores)
        
        # 전체 평균으로 정규화
        num_agents = len(policy_stats)
        for key in ['scenario_consistency', 'decision_stability']:
            consistency_metrics[key] /= num_agents if num_agents > 0 else 1
        
        # 4. 전체 신뢰도 (모든 지표의 평균)
        reliability_components = [
            consistency_metrics['decision_stability'],
            consistency_metrics['scenario_consistency'],
            consistency_metrics['agent_consistency']
        ]
        consistency_metrics['overall_reliability'] = np.mean([c for c in reliability_components if c > 0])
        
        return consistency_metrics
    
    def _calculate_overall_quality_score(
        self, 
        cooperation_metrics: Dict,
        strategic_metrics: Dict,
        adaptation_metrics: Dict,
        consistency_metrics: Dict
    ) -> float:
        """
        전체 정책 품질 점수를 계산합니다.
        """
        # 각 카테고리별 가중 평균
        cooperation_score = np.mean(list(cooperation_metrics.values()))
        strategic_score = np.mean(list(strategic_metrics.values()))
        adaptation_score = np.mean(list(adaptation_metrics.values()))
        consistency_score = np.mean(list(consistency_metrics.values()))
        
        # 전체 점수 (가중 평균)
        overall_score = 0.3 * cooperation_score + 0.3 * strategic_score + 0.2 * adaptation_score + 0.2 * consistency_score
        
        return overall_score
    
    def plot_policy_quality_analysis(
        self, 
        quality_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Policy quality 분석 결과를 시각화합니다.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Policy Quality Analysis', fontsize=16, fontweight='bold')
        
        # Figure 크기 제한
        fig.set_size_inches(15, 12)
        
        # 1. 협력 품질 (요약 지표만 표시)
        ax1 = axes[0, 0]
        cooperation_data = quality_results['cooperation_quality']
        
        # 요약 지표만 필터링
        summary_metrics = ['overall_stag_preference', 'my_stag_consistency', 
                          'defection_avoidance', 'partner_coordination']
        metrics = []
        values = []
        for metric in summary_metrics:
            if metric in cooperation_data:
                metrics.append(metric)
                values.append(cooperation_data[metric])
        
        if metrics:
            bars1 = ax1.bar(metrics, values, alpha=0.7, color='skyblue')
            ax1.set_title('Cooperation Quality (Summary)', fontweight='bold')
            ax1.set_ylabel('Score')
            ax1.set_ylim(0, 1)
            ax1.tick_params(axis='x', rotation=45)
            
            # 값 표시
            for bar, value in zip(bars1, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        else:
            ax1.text(0.5, 0.5, 'No Summary Data', ha='center', va='center', 
                   transform=ax1.transAxes, fontsize=12, alpha=0.5)
        
        # 2. 전략적 사고
        ax2 = axes[0, 1]
        strategic_data = quality_results['strategic_thinking']
        metrics = list(strategic_data.keys())
        values = list(strategic_data.values())
        
        bars2 = ax2.bar(metrics, values, alpha=0.7, color='lightgreen')
        ax2.set_title('Strategic Thinking', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, max(values) * 1.1 if values else 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 적응 능력
        ax3 = axes[1, 0]
        adaptation_data = quality_results['adaptation_ability']
        metrics = list(adaptation_data.keys())
        values = list(adaptation_data.values())
        
        bars3 = ax3.bar(metrics, values, alpha=0.7, color='orange')
        ax3.set_title('Adaptation Ability', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. 일관성
        ax4 = axes[1, 1]
        consistency_data = quality_results['policy_consistency']
        metrics = list(consistency_data.keys())
        values = list(consistency_data.values())
        
        bars4 = ax4.bar(metrics, values, alpha=0.7, color='pink')
        ax4.set_title('Policy Consistency', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 전체 점수 표시
        overall_score = quality_results['overall_quality_score']
        fig.text(0.5, 0.02, f'Overall Quality Score: {overall_score:.3f}', 
                ha='center', fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Policy quality analysis plot saved: {save_path}")
        
        return fig
    
    def plot_case_by_case_analysis(
        self, 
        quality_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Case-by-case 분석 결과를 시각화합니다.
        """
        cooperation_data = quality_results['cooperation_quality']
        
        # Case별 데이터 추출
        case_data = {}
        case_names = [
            'my_stag_only', 'coop_stag_only', 'hare_only',
            'my_stag_coop_stag', 'my_stag_hare', 'coop_stag_hare', 
            'my_stag_coop_stag_hare'
        ]
        
        for case_name in case_names:
            case_data[case_name] = {}
            for choice in ['my_stag', 'coop_stag', 'hare', 'none']:
                key = f'{case_name}_{choice}_ratio'
                if key in cooperation_data:
                    case_data[case_name][choice] = cooperation_data[key]
                else:
                    case_data[case_name][choice] = 0.0
        
        # 시각화
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle('Case-by-Case Huntable Object Selection Analysis', fontsize=16, fontweight='bold')
        
        # Figure 크기 제한
        fig.set_size_inches(20, 10)
        
        # 각 case별 막대 그래프
        for i, case_name in enumerate(case_names):
            row = i // 4
            col = i % 4
            ax = axes[row, col]
            
            if case_name in case_data and any(case_data[case_name].values()):
                choices = list(case_data[case_name].keys())
                values = list(case_data[case_name].values())
                
                # 막대 그래프
                bars = ax.bar(choices, values, alpha=0.7, 
                            color=['red', 'blue', 'green', 'gray'])
                
                # 값 표시
                for bar, value in zip(bars, values):
                    if value > 0:
                        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.2f}', ha='center', va='bottom', fontsize=9)
                
                ax.set_title(f'{case_name.replace("_", " ").title()}', fontweight='bold')
                ax.set_ylabel('Selection Ratio')
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
            else:
                ax.set_title(f'{case_name.replace("_", " ").title()}', fontweight='bold')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12, alpha=0.5)
                ax.set_ylim(0, 1)
        
        # 마지막 subplot은 요약 지표
        ax_summary = axes[1, 3]
        summary_metrics = ['overall_stag_preference', 'my_stag_consistency', 
                          'defection_avoidance', 'partner_coordination']
        summary_values = [cooperation_data.get(metric, 0.0) for metric in summary_metrics]
        
        bars = ax_summary.bar(summary_metrics, summary_values, alpha=0.7, color='orange')
        for bar, value in zip(bars, summary_values):
            ax_summary.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontsize=9)
        
        ax_summary.set_title('Summary Metrics', fontweight='bold')
        ax_summary.set_ylabel('Score')
        ax_summary.set_ylim(0, 1)
        ax_summary.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"Case-by-case analysis plot saved: {save_path}")
        
        return fig
