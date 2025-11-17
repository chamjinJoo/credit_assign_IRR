import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


class ValueNetworkEvaluator:
    """
    학습된 value network의 정확도를 평가하여 causal learning의 효과를 검증하는 클래스.
    
    기존 causal evaluation 데이터를 재활용하여 중복 작업을 최소화합니다.
    """
    
    def __init__(self, algorithm, env):
        self.algorithm = algorithm
        self.env = env
        self.num_agents = env.num_agents
        
        # Stag-Hunt 환경 특성
        self.stag_reward = env.stag_reward
        self.forage_reward = env.forage_reward
        self.required_agents = env.required_agents
        
    def evaluate_value_network_quality_from_data(
        self, 
        causal_evaluation_data: Dict,
        advanced_causal_data: Dict
    ) -> Dict[str, Any]:
        """
        기존 causal evaluation 데이터를 활용하여 value network 품질을 평가합니다.
        
        Args:
            causal_evaluation_data: CausalEffectEvaluator에서 수집된 데이터
            advanced_causal_data: AdvancedCausalEvaluator에서 수집된 데이터
            
        Returns:
            Value network quality metrics
        """
        print("Value Network Quality Evaluation 시작...")
        
        # 1. 상태 가치 예측 정확도 평가
        state_value_accuracy = self._evaluate_state_value_accuracy(advanced_causal_data)
        
        # 2. 협력 상황에서의 가치 평가
        cooperation_value_assessment = self._evaluate_cooperation_value_assessment(advanced_causal_data)
        
        # 3. 장기적 가치 예측 능력
        long_term_prediction = self._evaluate_long_term_prediction(advanced_causal_data)
        
        # 4. Causal effect와의 일관성
        causal_consistency = self._evaluate_causal_consistency(advanced_causal_data)
        
        results = {
            'state_value_accuracy': state_value_accuracy,
            'cooperation_value_assessment': cooperation_value_assessment,
            'long_term_prediction': long_term_prediction,
            'causal_consistency': causal_consistency,
            'overall_value_quality': self._calculate_overall_value_quality(
                state_value_accuracy, cooperation_value_assessment, 
                long_term_prediction, causal_consistency
            )
        }
        
        print("Value Network Quality Evaluation 완료!")
        return results
    
    def _evaluate_state_value_accuracy(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        상태 가치 예측의 정확도를 평가합니다.
        """
        causal_stats = advanced_causal_data.get('causal_effects', {})
        
        accuracy_metrics = {
            'scenario_discrimination_accuracy': 0.0,
            'effect_magnitude_consistency': 0.0,
            'directional_accuracy': 0.0,
            'prediction_stability': 0.0
        }
        
        for agent_key, agent_data in causal_stats.items():
            agent_metrics = {
                'scenario_discrimination': [],
                'effect_magnitude_consistency': [],
                'directional_accuracy': [],
                'prediction_stability': []
            }
            
            # 시나리오별 causal effect 분석
            scenario_effects = {}
            for scenario, stats in agent_data.items():
                mean_effect = stats['mean']
                std_effect = stats['std']
                count = stats['count']
                
                scenario_effects[scenario] = {
                    'mean': mean_effect,
                    'std': std_effect,
                    'count': count
                }
            
            # 1. 시나리오 구분 정확도 (다른 시나리오 간 차이의 명확성)
            if len(scenario_effects) > 1:
                effects = [data['mean'] for data in scenario_effects.values()]
                # 시나리오 간 차이의 명확성 (표준편차가 클수록 구분이 명확)
                discrimination_score = np.std(effects) / (np.mean(np.abs(effects)) + 1e-8)
                agent_metrics['scenario_discrimination'].append(discrimination_score)
            
            # 2. 효과 크기 일관성 (같은 시나리오에서의 일관된 예측)
            for scenario, data in scenario_effects.items():
                if data['count'] > 5:  # 충분한 데이터가 있는 경우만
                    # 표준편차가 작을수록 일관된 예측
                    consistency_score = 1.0 / (1.0 + data['std'])
                    agent_metrics['effect_magnitude_consistency'].append(consistency_score)
            
            # 3. 방향성 정확도 (시나리오 유형에 따른 적절한 방향)
            cooperative_effects = []
            defection_effects = []
            
            for scenario, data in scenario_effects.items():
                effect = data['mean']
                if 'cooperative' in scenario:
                    cooperative_effects.append(effect)
                elif 'defection' in scenario or 'hare' in scenario:
                    defection_effects.append(effect)
            
            # 협력 시나리오는 양의 효과, 배신 시나리오는 음의 효과를 가져야 함
            if cooperative_effects and defection_effects:
                coop_positive = np.mean([e > 0 for e in cooperative_effects])
                def_negative = np.mean([e < 0 for e in defection_effects])
                directional_accuracy = (coop_positive + def_negative) / 2
                agent_metrics['directional_accuracy'].append(directional_accuracy)
            
            # 4. 예측 안정성 (시나리오 내에서의 일관성)
            for scenario, data in scenario_effects.items():
                if data['count'] > 10:
                    # 변동계수 (CV)가 낮을수록 안정적
                    cv = data['std'] / (abs(data['mean']) + 1e-8)
                    stability_score = 1.0 / (1.0 + cv)
                    agent_metrics['prediction_stability'].append(stability_score)
            
            # Agent별 평균 계산
            for key in agent_metrics:
                if agent_metrics[key]:
                    if key == 'scenario_discrimination':
                        accuracy_metrics['scenario_discrimination_accuracy'] += np.mean(agent_metrics[key])
                    elif key == 'effect_magnitude_consistency':
                        accuracy_metrics['effect_magnitude_consistency'] += np.mean(agent_metrics[key])
                    elif key == 'directional_accuracy':
                        accuracy_metrics['directional_accuracy'] += np.mean(agent_metrics[key])
                    elif key == 'prediction_stability':
                        accuracy_metrics['prediction_stability'] += np.mean(agent_metrics[key])
        
        # 전체 평균으로 정규화
        num_agents = len(causal_stats)
        for key in accuracy_metrics:
            accuracy_metrics[key] /= num_agents if num_agents > 0 else 1
        
        return accuracy_metrics
    
    def _evaluate_cooperation_value_assessment(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        협력 상황에서의 가치 평가 능력을 평가합니다.
        """
        causal_stats = advanced_causal_data.get('causal_effects', {})
        
        cooperation_metrics = {
            'cooperative_scenario_recognition': 0.0,
            'defection_scenario_recognition': 0.0,
            'cooperation_value_ranking': 0.0,
            'partner_effect_sensitivity': 0.0
        }
        
        for agent_key, agent_data in causal_stats.items():
            agent_metrics = {
                'cooperative_recognition': [],
                'defection_recognition': [],
                'value_ranking': [],
                'partner_sensitivity': []
            }
            
            # 시나리오 분류 및 분석
            cooperative_scenarios = []
            defection_scenarios = []
            partner_scenarios = []
            
            for scenario, stats in agent_data.items():
                effect = stats['mean']
                count = stats['count']
                
                if 'cooperative' in scenario:
                    cooperative_scenarios.append((scenario, effect, count))
                elif 'defection' in scenario or 'hare' in scenario:
                    defection_scenarios.append((scenario, effect, count))
                
                if 'partner' in scenario:
                    partner_scenarios.append((scenario, effect, count))
            
            # 1. 협력 시나리오 인식 (양의 causal effect)
            if cooperative_scenarios:
                positive_effects = [effect for _, effect, _ in cooperative_scenarios if effect > 0]
                recognition_rate = len(positive_effects) / len(cooperative_scenarios)
                agent_metrics['cooperative_recognition'].append(recognition_rate)
            
            # 2. 배신 시나리오 인식 (음의 causal effect)
            if defection_scenarios:
                negative_effects = [effect for _, effect, _ in defection_scenarios if effect < 0]
                recognition_rate = len(negative_effects) / len(defection_scenarios)
                agent_metrics['defection_recognition'].append(recognition_rate)
            
            # 3. 협력 가치 순위 (협력 > 배신 > 중립)
            if len(cooperative_scenarios) > 0 and len(defection_scenarios) > 0:
                coop_effects = [effect for _, effect, _ in cooperative_scenarios]
                def_effects = [effect for _, effect, _ in defection_scenarios]
                
                # 협력이 배신보다 높은 가치를 가지는지
                if coop_effects and def_effects:
                    coop_avg = np.mean(coop_effects)
                    def_avg = np.mean(def_effects)
                    ranking_score = 1.0 if coop_avg > def_avg else 0.0
                    agent_metrics['value_ranking'].append(ranking_score)
            
            # 4. 파트너 효과 민감도 (파트너 관련 시나리오에서의 반응)
            if partner_scenarios:
                partner_effects = [effect for _, effect, _ in partner_scenarios]
                # 파트너 효과의 절댓값이 클수록 민감
                sensitivity = np.mean(np.abs(partner_effects))
                agent_metrics['partner_sensitivity'].append(sensitivity)
            
            # Agent별 평균 계산
            for key in agent_metrics:
                if agent_metrics[key]:
                    if key == 'cooperative_recognition':
                        cooperation_metrics['cooperative_scenario_recognition'] += np.mean(agent_metrics[key])
                    elif key == 'defection_recognition':
                        cooperation_metrics['defection_scenario_recognition'] += np.mean(agent_metrics[key])
                    elif key == 'value_ranking':
                        cooperation_metrics['cooperation_value_ranking'] += np.mean(agent_metrics[key])
                    elif key == 'partner_sensitivity':
                        cooperation_metrics['partner_effect_sensitivity'] += np.mean(agent_metrics[key])
        
        # 전체 평균으로 정규화
        num_agents = len(causal_stats)
        for key in cooperation_metrics:
            cooperation_metrics[key] /= num_agents if num_agents > 0 else 1
        
        return cooperation_metrics
    
    def _evaluate_long_term_prediction(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        장기적 가치 예측 능력을 평가합니다.
        """
        causal_stats = advanced_causal_data.get('causal_effects', {})
        
        long_term_metrics = {
            'temporal_consistency': 0.0,
            'future_reward_anticipation': 0.0,
            'discounted_value_accuracy': 0.0,
            'strategic_planning_quality': 0.0
        }
        
        for agent_key, agent_data in causal_stats.items():
            agent_metrics = {
                'temporal_consistency': [],
                'future_anticipation': [],
                'discounted_accuracy': [],
                'strategic_planning': []
            }
            
            # 시나리오별 분석
            scenario_effects = {}
            for scenario, stats in agent_data.items():
                effect = stats['mean']
                std = stats['std']
                count = stats['count']
                scenario_effects[scenario] = {'effect': effect, 'std': std, 'count': count}
            
            # 1. 시간적 일관성 (같은 시나리오에서의 안정된 예측)
            for scenario, data in scenario_effects.items():
                if data['count'] > 5:
                    # 표준편차가 작을수록 시간적으로 일관됨
                    consistency = 1.0 / (1.0 + data['std'])
                    agent_metrics['temporal_consistency'].append(consistency)
            
            # 2. 미래 보상 예측 (stag reward 시나리오에서의 높은 가치)
            stag_scenarios = [data for scenario, data in scenario_effects.items() 
                            if 'stag' in scenario and 'cooperative' in scenario]
            if stag_scenarios:
                # Stag reward는 높은 가치를 가져야 함
                stag_effects = [data['effect'] for data in stag_scenarios]
                if stag_effects:
                    # Stag 효과의 평균이 양수이고 클수록 좋음
                    anticipation_score = max(0, np.mean(stag_effects)) / 5.0  # 정규화
                    agent_metrics['future_anticipation'].append(anticipation_score)
            
            # 3. 할인된 가치 정확도 (장기적 이익 고려)
            # 협력 시나리오는 장기적으로 더 높은 가치를 가져야 함
            cooperative_effects = [data['effect'] for scenario, data in scenario_effects.items() 
                                 if 'cooperative' in scenario]
            short_term_effects = [data['effect'] for scenario, data in scenario_effects.items() 
                                if 'hare' in scenario or 'defection' in scenario]
            
            if cooperative_effects and short_term_effects:
                coop_avg = np.mean(cooperative_effects)
                short_avg = np.mean(short_term_effects)
                # 협력이 단기적 이익보다 높은 가치를 가지는지
                discounted_accuracy = 1.0 if coop_avg > short_avg else 0.0
                agent_metrics['discounted_accuracy'].append(discounted_accuracy)
            
            # 4. 전략적 계획 품질 (시나리오 간 가치 차이의 논리성)
            if len(scenario_effects) > 2:
                effects = [data['effect'] for data in scenario_effects.values()]
                # 가치 차이가 명확하고 논리적인지 (분산이 적절한 수준)
                effect_variance = np.var(effects)
                effect_mean = np.mean(np.abs(effects))
                if effect_mean > 0:
                    planning_quality = min(1.0, effect_variance / (effect_mean + 1e-8))
                    agent_metrics['strategic_planning'].append(planning_quality)
            
            # Agent별 평균 계산
            for key in agent_metrics:
                if agent_metrics[key]:
                    if key == 'temporal_consistency':
                        long_term_metrics['temporal_consistency'] += np.mean(agent_metrics[key])
                    elif key == 'future_anticipation':
                        long_term_metrics['future_reward_anticipation'] += np.mean(agent_metrics[key])
                    elif key == 'discounted_accuracy':
                        long_term_metrics['discounted_value_accuracy'] += np.mean(agent_metrics[key])
                    elif key == 'strategic_planning':
                        long_term_metrics['strategic_planning_quality'] += np.mean(agent_metrics[key])
        
        # 전체 평균으로 정규화
        num_agents = len(causal_stats)
        for key in long_term_metrics:
            long_term_metrics[key] /= num_agents if num_agents > 0 else 1
        
        return long_term_metrics
    
    def _evaluate_causal_consistency(self, advanced_causal_data: Dict) -> Dict[str, float]:
        """
        Causal effect와 value network의 일관성을 평가합니다.
        """
        causal_stats = advanced_causal_data.get('causal_effects', {})
        
        consistency_metrics = {
            'causal_value_alignment': 0.0,
            'effect_magnitude_correlation': 0.0,
            'scenario_response_consistency': 0.0,
            'overall_causal_understanding': 0.0
        }
        
        for agent_key, agent_data in causal_stats.items():
            agent_metrics = {
                'causal_alignment': [],
                'magnitude_correlation': [],
                'response_consistency': [],
                'causal_understanding': []
            }
            
            # 시나리오별 causal effect 분석
            scenario_effects = {}
            for scenario, stats in agent_data.items():
                effect = stats['mean']
                std = stats['std']
                count = stats['count']
                scenario_effects[scenario] = {'effect': effect, 'std': std, 'count': count}
            
            # 1. Causal-Value 정렬 (causal effect와 예상 가치의 일치)
            # 협력 시나리오는 양의 causal effect, 배신 시나리오는 음의 causal effect
            cooperative_effects = [data['effect'] for scenario, data in scenario_effects.items() 
                                 if 'cooperative' in scenario]
            defection_effects = [data['effect'] for scenario, data in scenario_effects.items() 
                               if 'defection' in scenario or 'hare' in scenario]
            
            alignment_score = 0.0
            if cooperative_effects:
                coop_alignment = np.mean([e > 0 for e in cooperative_effects])
                alignment_score += coop_alignment * 0.5
            
            if defection_effects:
                def_alignment = np.mean([e < 0 for e in defection_effects])
                alignment_score += def_alignment * 0.5
            
            if cooperative_effects or defection_effects:
                agent_metrics['causal_alignment'].append(alignment_score)
            
            # 2. 효과 크기 상관관계 (causal effect의 크기와 시나리오 중요도의 일치)
            if len(scenario_effects) > 1:
                effects = [data['effect'] for data in scenario_effects.values()]
                # Stag reward 시나리오가 더 큰 causal effect를 가지는지
                stag_effects = [data['effect'] for scenario, data in scenario_effects.items() 
                              if 'stag' in scenario]
                other_effects = [data['effect'] for scenario, data in scenario_effects.items() 
                               if 'stag' not in scenario]
                
                if stag_effects and other_effects:
                    stag_avg = np.mean(np.abs(stag_effects))
                    other_avg = np.mean(np.abs(other_effects))
                    if other_avg > 0:
                        magnitude_correlation = min(1.0, stag_avg / other_avg)
                        agent_metrics['magnitude_correlation'].append(magnitude_correlation)
            
            # 3. 시나리오 응답 일관성 (같은 유형 시나리오에서의 일관된 반응)
            scenario_groups = {
                'cooperative': [data for scenario, data in scenario_effects.items() 
                               if 'cooperative' in scenario],
                'defection': [data for scenario, data in scenario_effects.items() 
                            if 'defection' in scenario or 'hare' in scenario],
                'neutral': [data for scenario, data in scenario_effects.items() 
                          if 'no_reward' in scenario]
            }
            
            group_consistency = []
            for group_name, group_data in scenario_groups.items():
                if len(group_data) > 1:
                    group_effects = [data['effect'] for data in group_data]
                    # 그룹 내 효과의 일관성 (표준편차가 작을수록 일관됨)
                    if group_effects:
                        consistency = 1.0 / (1.0 + np.std(group_effects))
                        group_consistency.append(consistency)
            
            if group_consistency:
                agent_metrics['response_consistency'].append(np.mean(group_consistency))
            
            # 4. 전체적인 causal understanding
            # 모든 지표의 종합적 평가
            understanding_components = []
            if agent_metrics['causal_alignment']:
                understanding_components.append(np.mean(agent_metrics['causal_alignment']))
            if agent_metrics['magnitude_correlation']:
                understanding_components.append(np.mean(agent_metrics['magnitude_correlation']))
            if agent_metrics['response_consistency']:
                understanding_components.append(np.mean(agent_metrics['response_consistency']))
            
            if understanding_components:
                agent_metrics['causal_understanding'].append(np.mean(understanding_components))
            
            # Agent별 평균 계산
            for key in agent_metrics:
                if agent_metrics[key]:
                    if key == 'causal_alignment':
                        consistency_metrics['causal_value_alignment'] += np.mean(agent_metrics[key])
                    elif key == 'magnitude_correlation':
                        consistency_metrics['effect_magnitude_correlation'] += np.mean(agent_metrics[key])
                    elif key == 'response_consistency':
                        consistency_metrics['scenario_response_consistency'] += np.mean(agent_metrics[key])
                    elif key == 'causal_understanding':
                        consistency_metrics['overall_causal_understanding'] += np.mean(agent_metrics[key])
        
        # 전체 평균으로 정규화
        num_agents = len(causal_stats)
        for key in consistency_metrics:
            consistency_metrics[key] /= num_agents if num_agents > 0 else 1
        
        return consistency_metrics
    
    def _calculate_overall_value_quality(
        self,
        state_value_accuracy: Dict,
        cooperation_value_assessment: Dict,
        long_term_prediction: Dict,
        causal_consistency: Dict
    ) -> float:
        """
        전체 value network 품질 점수를 계산합니다.
        """
        # 각 카테고리별 평균 점수
        accuracy_score = np.mean(list(state_value_accuracy.values()))
        cooperation_score = np.mean(list(cooperation_value_assessment.values()))
        long_term_score = np.mean(list(long_term_prediction.values()))
        consistency_score = np.mean(list(causal_consistency.values()))
        
        # 전체 점수 (가중 평균)
        overall_score = 0.25 * accuracy_score + 0.25 * cooperation_score + 0.25 * long_term_score + 0.25 * consistency_score
        
        return overall_score
    
    def plot_value_network_analysis(
        self, 
        quality_results: Dict[str, Any],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Value network 품질 분석 결과를 시각화합니다.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Value Network Quality Analysis', fontsize=16, fontweight='bold')
        
        # 1. 상태 가치 예측 정확도
        ax1 = axes[0, 0]
        accuracy_data = quality_results['state_value_accuracy']
        metrics = list(accuracy_data.keys())
        values = list(accuracy_data.values())
        
        bars1 = ax1.bar(metrics, values, alpha=0.7, color='lightblue')
        ax1.set_title('State Value Prediction Accuracy', fontweight='bold')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, max(values) * 1.1 if values else 1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars1, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. 협력 가치 평가
        ax2 = axes[0, 1]
        cooperation_data = quality_results['cooperation_value_assessment']
        metrics = list(cooperation_data.keys())
        values = list(cooperation_data.values())
        
        bars2 = ax2.bar(metrics, values, alpha=0.7, color='lightgreen')
        ax2.set_title('Cooperation Value Assessment', fontweight='bold')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 3. 장기적 예측
        ax3 = axes[1, 0]
        long_term_data = quality_results['long_term_prediction']
        metrics = list(long_term_data.keys())
        values = list(long_term_data.values())
        
        bars3 = ax3.bar(metrics, values, alpha=0.7, color='orange')
        ax3.set_title('Long-term Prediction Quality', fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_ylim(0, max(values) * 1.1 if values else 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 4. Causal 일관성
        ax4 = axes[1, 1]
        consistency_data = quality_results['causal_consistency']
        metrics = list(consistency_data.keys())
        values = list(consistency_data.values())
        
        bars4 = ax4.bar(metrics, values, alpha=0.7, color='pink')
        ax4.set_title('Causal Consistency', fontweight='bold')
        ax4.set_ylabel('Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars4, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 전체 점수 표시
        overall_score = quality_results['overall_value_quality']
        fig.text(0.5, 0.02, f'Overall Value Network Quality: {overall_score:.3f}', 
                ha='center', fontsize=14, fontweight='bold', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Value network analysis plot saved: {save_path}")
        
        return fig
