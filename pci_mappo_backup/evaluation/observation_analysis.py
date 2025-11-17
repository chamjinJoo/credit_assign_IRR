import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from typing import Dict, List, Optional, Tuple, Any
import os
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive plots will not work.")


class ObservationAnalyzer:
    """
    Agent의 observation을 t-SNE를 통해 분석하고 시각화.
    """
    
    def __init__(
        self,
        num_agents: int,
        perplexity: int = 30,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Args:
            num_agents: Agent 수
            perplexity: t-SNE perplexity 파라미터
            max_iter: t-SNE iteration 수
            random_state: Random seed
        """
        self.num_agents = num_agents
        self.perplexity = perplexity
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Scenario 색상 매핑
        self.scenario_colors = {
            'no_reward': '#95A5A6',        # 회색
            'forage_reward': '#3498DB',    # 파랑
            'stag_reward': '#2ECC71'       # 초록
        }
        
        self.scenario_labels = {
            'no_reward': 'No Reward (0)',
            'forage_reward': 'Hare Hunting (3)',
            'stag_reward': 'Cooperative Stag (5)'
        }
    
        
        self.huntable_colors = {
            'none': '#7F8C8D',  # 회색
            'coop_stag': '#e93e3a',
            'coop_stag+my_stag': '#ed683c',
            'coop_stag+my_stag+hare': '#f3903f',
            'coop_stag+hare': '#f8b23b',
            'my_stag': '#f7a70a',   
            'my_stag+hare': '#fdc70c',
            'hare': '#fff33b' 
        }
        
        self.huntable_labels = {
            'none': 'No Huntable Objects',
            'coop_stag': 'Coop Stag Only',
            'coop_stag+my_stag': 'Coop + My Stag',
            'coop_stag+my_stag+hare': 'Coop + My Stag + Hare',
            'coop_stag+hare': 'Coop + Hare',
            'my_stag': 'My Stag Only',
            'my_stag+hare': 'My Stag + Hare',
            'hare': 'Hare Only'
        }
    
    def _normalize_case_name(self, case: str) -> str:
        """
        Case 이름을 정규화하여 순서에 관계없이 동일한 조합을 같은 키로 매핑합니다.
        
        Args:
            case: 원본 case 이름 (예: 'my_stag+coop_stag+hare')
            
        Returns:
            정규화된 case 이름 (예: 'coop_stag+my_stag+hare')
        """
        if case == 'none':
            return 'none'
        
        # '+'로 분리
        parts = case.split('+')
        
        # 우선순위 기반 정렬 (coop_stag가 가장 높은 우선순위)
        priority_order = ['coop_stag', 'my_stag', 'hare']
        sorted_parts = []
        
        for obj_type in priority_order:
            if obj_type in parts:
                sorted_parts.append(obj_type)
        
        # 정렬된 순서로 재조합
        return '+'.join(sorted_parts)
    
    
    
    
    def analyze_observation_separability(
        self,
        observation_data: Dict[int, Dict[str, List]]
    ) -> Dict[int, Dict[str, float]]:
        """
        Observation의 scenario별 분리도를 정량적으로 분석합니다.
        
        Args:
            observation_data: Dict[agent_id: {'observations': [...], 'scenarios': [...]}]
            
        Returns:
            Dict[agent_id: {'intra_variance': float, 'inter_variance': float, 'separability_score': float}]
        """
        separability_scores = {}
        
        for agent_id, data in observation_data.items():
            observations = np.array(data['observations'])
            scenarios = np.array(data['scenarios'])
            
            if len(observations) < 2:
                continue
            
            # Scenario별 평균과 분산 계산
            scenario_means = {}
            scenario_variances = {}
            
            for scenario in np.unique(scenarios):
                mask = scenarios == scenario
                scenario_obs = observations[mask]
                
                if len(scenario_obs) > 0:
                    scenario_means[scenario] = np.mean(scenario_obs, axis=0)
                    scenario_variances[scenario] = np.var(scenario_obs, axis=0).mean()
            
            # Intra-class variance (클래스 내 분산 평균)
            intra_variance = np.mean(list(scenario_variances.values()))
            
            # Inter-class variance (클래스 간 분산)
            overall_mean = np.mean(observations, axis=0)
            inter_variance = np.mean([
                np.sum((mean - overall_mean) ** 2) 
                for mean in scenario_means.values()
            ])
            
            # Separability score (inter / intra, 클수록 잘 분리됨)
            separability = inter_variance / (intra_variance + 1e-8)
            
            separability_scores[agent_id] = {
                'intra_variance': float(intra_variance),
                'inter_variance': float(inter_variance),
                'separability_score': float(separability)
            }
        
        return separability_scores
    
    def save_observation_data(
        self,
        observation_data: Dict[int, Dict[str, List]],
        save_dir: str
    ):
        """
        Observation 데이터를 저장합니다.
        
        Args:
            observation_data: Dict[agent_id: {'observations': [...], 'scenarios': [...]}]
            save_dir: 저장 디렉토리
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for agent_id, data in observation_data.items():
            # NumPy 배열로 저장
            obs_array = np.array(data['observations'])
            scenarios_array = np.array(data['scenarios'])
            
            obs_path = os.path.join(save_dir, f'agent_{agent_id}_observations.npy')
            scenario_path = os.path.join(save_dir, f'agent_{agent_id}_scenarios.npy')
            
            np.save(obs_path, obs_array)
            np.save(scenario_path, scenarios_array)
        
        print(f"Observation 데이터 저장 완료: {save_dir}")
    
    
    def _find_optimal_clusters(
        self,
        tsne_results: np.ndarray,
        max_clusters: int = 10
    ) -> int:
        """
        Elbow method를 사용하여 최적의 cluster 개수를 찾습니다.
        
        Args:
            tsne_results: t-SNE 결과 [n_samples, 2]
            max_clusters: 최대 cluster 개수
            
        Returns:
            최적의 cluster 개수
        """
        from sklearn.metrics import silhouette_score
        
        if len(tsne_results) < 2:
            return 1
        
        # 최대 cluster 개수 제한 (데이터 포인트 수의 절반 이하)
        max_k = min(max_clusters, len(tsne_results) // 2, 10)
        max_k = max(2, max_k)  # 최소 2개
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, max_k + 1)
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tsne_results)
            
            # Inertia (within-cluster sum of squares)
            inertias.append(kmeans.inertia_)
            
            # Silhouette score
            if len(np.unique(cluster_labels)) > 1:  # 최소 2개 cluster가 있어야 silhouette 계산 가능
                sil_score = silhouette_score(tsne_results, cluster_labels)
                silhouette_scores.append(sil_score)
            else:
                silhouette_scores.append(0)
        
        # Silhouette score가 가장 높은 k 선택
        if silhouette_scores:
            optimal_k = k_range[np.argmax(silhouette_scores)]
        else:
            optimal_k = 2
        
        print(f"    Auto clustering: 최적 cluster 수 = {optimal_k} (silhouette score: {max(silhouette_scores):.3f})")
        
        return optimal_k
    
    def _perform_clustering(
        self,
        tsne_results: np.ndarray,
        n_clusters: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        t-SNE 결과에 대해 K-means clustering을 수행합니다.
        
        Args:
            tsne_results: t-SNE 결과 [n_samples, 2]
            n_clusters: Cluster 개수 (None이면 auto clustering)
            
        Returns:
            (cluster_labels, cluster_centers)
            - cluster_labels: 각 point의 cluster 할당 [n_samples]
            - cluster_centers: Cluster 중심점 [n_clusters, 2]
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(tsne_results)
        cluster_centers = kmeans.cluster_centers_
        
        return cluster_labels, cluster_centers
    
    def _render_cluster_representatives(
        self,
        observations: np.ndarray,
        tsne_results: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        scenarios: np.ndarray,
        env
    ) -> Dict[int, Dict]:
        """
        각 cluster의 대표 observation을 렌더링합니다.
        
        Returns:
            Dict[cluster_id: {'image': base64_str, 'scenarios': {...}, 'obs': np.ndarray}]
        """
        import base64
        from io import BytesIO
        
        rendered_images = {}
        
        print("    Cluster representative observations 렌더링 중...")
        
        for cluster_id in range(len(cluster_centers)):
            center = cluster_centers[cluster_id]
            
            # 해당 cluster에 속하는 observations
            mask = cluster_labels == cluster_id
            cluster_tsne = tsne_results[mask]
            cluster_obs = observations[mask]
            cluster_scenarios = scenarios[mask]
            
            if len(cluster_obs) == 0:
                continue
            
            # 중심에 가장 가까운 observation 찾기
            distances = np.linalg.norm(cluster_tsne - center, axis=1)
            closest_idx = np.argmin(distances)
            
            representative_obs = cluster_obs[closest_idx]
            
            # Scenario 분포 계산
            unique_scenarios, counts = np.unique(cluster_scenarios, return_counts=True)
            scenario_dist = {
                scenario: int(count) 
                for scenario, count in zip(unique_scenarios, counts)
            }
            
            # Environment 렌더링
            try:
                img = self._render_observation(representative_obs, env)
                
                # 이미지를 base64로 인코딩
                buffered = BytesIO()
                plt.savefig(buffered, format='png', bbox_inches='tight', dpi=100)
                buffered.seek(0)
                img_base64 = base64.b64encode(buffered.read()).decode()
                plt.close()
                
                rendered_images[cluster_id] = {
                    'image': img_base64,
                    'scenarios': scenario_dist,
                    'total_count': len(cluster_obs),
                    'obs': representative_obs
                }
            except Exception as e:
                print(f"      Warning: Cluster {cluster_id} 렌더링 실패: {e}")
                rendered_images[cluster_id] = {
                    'image': None,
                    'scenarios': scenario_dist,
                    'total_count': len(cluster_obs),
                    'obs': representative_obs
                }
        
        return rendered_images
    
    def _render_observation(self, obs: np.ndarray, env) -> plt.Figure:
        """
        Observation을 시각화합니다.
        
        Args:
            obs: Observation array
            env: Environment 객체
            
        Returns:
            Matplotlib figure
        """
        # StagHunt observation 구조 파싱
        num_agents = env.num_agents
        coord_size = 2
        cell_size = 2 * num_agents + 1
        grid_size = 9 * cell_size
        
        coord_obs = obs[:coord_size]
        grid_obs = obs[coord_size:coord_size + grid_size]
        
        # 3x3 grid 시각화
        fig, ax = plt.subplots(figsize=(4, 4))
        
        # Grid 그리기
        grid_positions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        # Background
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # 3x3 grid cells
        for idx, (dx, dy) in enumerate(grid_positions):
            cell_start = idx * cell_size
            cell_info = grid_obs[cell_start:cell_start + cell_size]
            
            # 경계 밖인 경우
            if cell_info[0] == -1.0:
                rect = plt.Rectangle((dx-0.4, dy-0.4), 0.8, 0.8, 
                                    facecolor='black', alpha=0.3)
                ax.add_patch(rect)
                continue
            
            # Cell 배경
            rect = plt.Rectangle((dx-0.4, dy-0.4), 0.8, 0.8, 
                                facecolor='white', edgecolor='gray', linewidth=1)
            ax.add_patch(rect)
            
            # Agents (여러 agent가 겹쳐있는 경우 처리)
            agents_in_cell = [agent_id for agent_id in range(num_agents) if cell_info[agent_id] == 1.0]
            
            if len(agents_in_cell) == 1:
                # 한 명의 agent만 있는 경우
                agent_id = agents_in_cell[0]
                circle = plt.Circle((dx, dy), 0.2, color=f'C{agent_id}', alpha=0.8, linewidth=2, edgecolor='black')
                ax.add_patch(circle)
                ax.text(dx, dy, f'A{agent_id}', ha='center', va='center', 
                       fontsize=9, color='white', fontweight='bold')
            elif len(agents_in_cell) > 1:
                # 여러 agent가 겹쳐있는 경우 - 동심원으로 표시
                for i, agent_id in enumerate(agents_in_cell):
                    radius = 0.25 - i * 0.05
                    circle = plt.Circle((dx, dy), radius, color=f'C{agent_id}', 
                                      alpha=0.7, linewidth=2, edgecolor='black')
                    ax.add_patch(circle)
                
                # 중앙에 agent ID들 표시
                agent_text = '+'.join([f'A{a}' for a in agents_in_cell])
                ax.text(dx, dy, agent_text, ha='center', va='center', 
                       fontsize=7, color='white', fontweight='bold')
            
            # Stags
            for stag_id in range(num_agents):
                stag_idx = num_agents + stag_id
                if cell_info[stag_idx] == 1.0:
                    triangle = plt.Polygon(
                        [(dx, dy-0.15), (dx-0.15, dy+0.15), (dx+0.15, dy+0.15)],
                        color=f'C{stag_id}', alpha=0.6
                    )
                    ax.add_patch(triangle)
                    ax.text(dx, dy+0.05, f'S{stag_id}', ha='center', va='center',
                           fontsize=6, fontweight='bold')
            
            # Hare
            hare_idx = 2 * num_agents
            if cell_info[hare_idx] == 1.0:
                rect = plt.Rectangle((dx-0.1, dy-0.1), 0.2, 0.2, 
                                   color='gray', alpha=0.7)
                ax.add_patch(rect)
                ax.text(dx, dy, 'H', ha='center', va='center',
                       fontsize=6, color='white', fontweight='bold')
        
        ax.set_title('3x3 Vision (Agent at Center)', fontsize=10, fontweight='bold')
        ax.set_xlabel('Relative X', fontsize=8)
        ax.set_ylabel('Relative Y', fontsize=8)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        
        return fig
    
    def _create_interactive_plot_with_clusters(
        self,
        tsne_results: np.ndarray,
        scenarios: np.ndarray,
        cluster_labels: np.ndarray,
        cluster_centers: np.ndarray,
        rendered_images: Dict[int, Dict],
        agent_id: int
    ) -> go.Figure:
        """
        Clustering 결과를 포함한 interactive plotly scatter plot을 생성합니다.
        
        Args:
            tsne_results: t-SNE 결과 [n_samples, 2]
            scenarios: Scenario 라벨 [n_samples]
            cluster_labels: Cluster 할당 [n_samples]
            cluster_centers: Cluster 중심점 [n_clusters, 2]
            rendered_images: Cluster별 렌더링 정보
            agent_id: Agent ID
            
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Scenario별로 scatter plot 추가 (색상은 scenario, 배경은 cluster)
        for scenario in np.unique(scenarios):
            mask = scenarios == scenario
            color = self.scenario_colors.get(scenario, '#000000')
            label = self.scenario_labels.get(scenario, scenario)
            
            # Hover text 생성 (scenario와 cluster 정보 모두 표시)
            hover_texts = []
            scenario_clusters = cluster_labels[mask]
            for cluster_id in scenario_clusters:
                hover_texts.append(
                    f"<b>Scenario:</b> {label}<br>"
                    f"<b>Cluster:</b> {cluster_id}<br>"
                )
            
            fig.add_trace(go.Scatter(
                x=tsne_results[mask, 0],
                y=tsne_results[mask, 1],
                mode='markers',
                name=f'{label} (n={mask.sum()})',
                marker=dict(
                    color=color,
                    size=6,
                    opacity=0.6,
                    line=dict(color='white', width=0.5)
                ),
                hovertext=hover_texts,
                hoverinfo='text'
            ))
        
        # Cluster 중심 표시 (렌더링된 이미지와 함께)
        for cluster_id, center in enumerate(cluster_centers):
            cluster_info = rendered_images.get(cluster_id, {})
            img_base64 = cluster_info.get('image')
            scenario_dist = cluster_info.get('scenarios', {})
            total_count = cluster_info.get('total_count', 0)
            
            # Scenario 분포 텍스트 생성
            scenario_text = '<br>'.join([
                f"{self.scenario_labels.get(s, s)}: {count}" 
                for s, count in scenario_dist.items()
            ])
            
            # Hover template (이미지는 페이지 하단에 표시)
            hover_template = (
                f"<b>★ Cluster {cluster_id}</b><br>"
                f"<b>Total samples:</b> {total_count}<br>"
                f"<b>Scenario distribution:</b><br>"
                f"{scenario_text}<br>"
                f"<br>"
                f"<i>페이지 하단에서 이미지 확인</i>"
                "<extra></extra>"
            )
            
            # 중심점에 별 마커 추가
            fig.add_trace(go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers+text',
                name=f'Cluster {cluster_id}',
                marker=dict(
                    symbol='star',
                    color='gold',
                    size=25,
                    line=dict(color='black', width=2)
                ),
                text=[f'★{cluster_id}'],
                textposition='top center',
                textfont=dict(size=12, color='black', family='Arial Black'),
                hovertemplate=hover_template,
                showlegend=True
            ))
        
        # Layout 설정
        fig.update_layout(
            title=f'Agent {agent_id} - Interactive t-SNE with Automatic Clustering',
            xaxis_title='t-SNE Dimension 1',
            yaxis_title='t-SNE Dimension 2',
            width=1000,
            height=800,
            hovermode='closest',
            template='plotly_white',
            font=dict(size=11)
        )
        
        return fig
    
    def _save_html_with_images(
        self,
        fig: go.Figure,
        save_path: str,
        rendered_images: Dict[int, Dict],
        agent_id: int
    ):
        """
        Plotly figure와 cluster 이미지를 포함한 HTML을 생성합니다.
        
        Args:
            fig: Plotly figure
            save_path: 저장 경로
            rendered_images: Cluster별 렌더링 정보
            agent_id: Agent ID
        """
        # Plotly HTML 생성
        plotly_html = fig.to_html(include_plotlyjs='cdn', div_id='plotly-div')
        
        # Cluster 이미지 섹션 생성
        cluster_html = '<div style="margin-top: 30px; padding: 20px; background-color: #f8f9fa;">\n'
        cluster_html += f'<h2>Cluster Representative Observations (Agent {agent_id})</h2>\n'
        cluster_html += '<p>각 cluster 중심에 가장 가까운 observation의 3x3 시야를 시각화합니다.</p>\n'
        cluster_html += '<div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; margin-top: 20px;">\n'
        
        for cluster_id in sorted(rendered_images.keys()):
            cluster_info = rendered_images[cluster_id]
            img_base64 = cluster_info.get('image')
            scenario_dist = cluster_info.get('scenarios', {})
            total_count = cluster_info.get('total_count', 0)
            
            cluster_html += '<div style="border: 2px solid #ddd; border-radius: 8px; padding: 15px; background-color: white;">\n'
            cluster_html += f'<h3 style="margin-top: 0;">★ Cluster {cluster_id}</h3>\n'
            cluster_html += f'<p><b>Total samples:</b> {total_count}</p>\n'
            cluster_html += '<p><b>Scenario distribution:</b></p>\n'
            cluster_html += '<ul style="margin: 5px 0;">\n'
            
            for scenario, count in scenario_dist.items():
                scenario_label = self.scenario_labels.get(scenario, scenario)
                percentage = (count / total_count * 100) if total_count > 0 else 0
                cluster_html += f'<li>{scenario_label}: {count} ({percentage:.1f}%)</li>\n'
            
            cluster_html += '</ul>\n'
            
            if img_base64:
                cluster_html += f'<img src="data:image/png;base64,{img_base64}" style="width: 100%; max-width: 400px; border: 1px solid #ccc; margin-top: 10px;">\n'
            else:
                cluster_html += '<p style="color: #999;">이미지 생성 실패</p>\n'
            
            cluster_html += '</div>\n'
        
        cluster_html += '</div>\n</div>\n'
        
        # 전체 HTML 조합
        full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Agent {agent_id} - Interactive t-SNE Analysis</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #ffffff;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        .instructions {{
            background-color: #e8f4f8;
            padding: 15px;
            border-left: 4px solid #3498db;
            margin: 20px 0;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Agent {agent_id} - Interactive t-SNE Observation Analysis</h1>
        
        {plotly_html}
        
        {cluster_html}
    </div>
</body>
</html>
"""
        
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(full_html)
    
    def plot_all_tsne_visualizations(
        self,
        observation_data: Dict[int, Dict[str, List]],
        value_networks: Dict[int, Any],
        env,
        save_path: Optional[str] = None,
        use_huntable_cases: bool = True,
        perplexity: int = 30,
        n_clusters: Optional[int] = None
    ):
        """
        한 번의 t-SNE embedding으로 모든 시각화를 생성하는 최적화된 메서드.
        
        Args:
            observation_data: Dict[agent_id: {'observations': [...], 'scenarios': [...], 'huntable_cases': [...]}]
            value_networks: Dict[agent_id: value_network] - 각 agent의 value network
            env: 환경 인스턴스
            save_path: 저장 경로
            use_huntable_cases: huntable_cases를 사용하여 색상 구분할지 여부
            perplexity: t-SNE perplexity 파라미터
            n_clusters: Cluster 개수 (None이면 auto clustering, 정수면 해당 개수 사용)
        """
        import torch
        
        num_agents = len(observation_data)
        
        if num_agents == 0:
            print("시각화할 observation 데이터가 없습니다.")
            return
        
        print(f"Optimized t-SNE analysis for {num_agents} agents...")
        
        # 각 agent별로 t-SNE embedding 계산 및 저장
        tsne_results = {}
        value_data = {}
        
        for agent_id in sorted(observation_data.keys()):
            data = observation_data[agent_id]
            observations = np.array(data['observations'])
            
            if len(observations) == 0:
                print(f"  Agent {agent_id}: No data")
                continue
            
            print(f"  Agent {agent_id}: {len(observations)} observations")
            
            # t-SNE 적용
            if len(observations) < 2:
                print(f"    Agent {agent_id}: Insufficient data")
                continue
            
            # Perplexity 조정
            adjusted_perplexity = min(perplexity, len(observations) // 3)
            adjusted_perplexity = max(5, adjusted_perplexity)
            
            print(f"    t-SNE 실행 중 (n={len(observations)}, perplexity={adjusted_perplexity})...")
            
            tsne = TSNE(
                n_components=2,
                perplexity=adjusted_perplexity,
                max_iter=self.max_iter,
                random_state=self.random_state,
                verbose=0
            )
            
            tsne_results[agent_id] = tsne.fit_transform(observations)
            
            # Value network으로 각 observation의 value 계산
            if agent_id in value_networks:
                value_network = value_networks[agent_id]
                value_network.eval()
                
                with torch.no_grad():
                    obs_tensor = torch.tensor(observations, dtype=torch.float32)
                    try:
                        model_device = next(value_network.parameters()).device
                    except StopIteration:
                        model_device = torch.device('cpu')
                    obs_tensor = obs_tensor.to(model_device)
                    
                    values = value_network(obs_tensor).cpu().numpy().flatten()
            else:
                print(f"    Warning: Value network not found for agent {agent_id}")
                values = np.zeros(len(observations))
            
            value_data[agent_id] = {
                'values': values
            }
        
        # 1. Huntable cases 기반 static t-SNE plot 생성
        if save_path:
            print("\n[1/5] Generating huntable cases t-SNE plot...")
            self._create_static_tsne_plot(
                observation_data, tsne_results, save_path, use_huntable_cases
            )
        
        # 2. Value heatmap static t-SNE plot 생성
        if save_path:
            print("\n[2/5] Generating value heatmap t-SNE plot...")
            base_path = save_path.replace('.png', '')
            value_heatmap_path = f"{base_path}_value_heatmap.png"
            self._create_static_value_heatmap_plot(
                observation_data, tsne_results, value_data, value_heatmap_path
            )
        
        # 3. Value by huntable case barplot 생성
        if save_path:
            print("\n[3/5] Generating value by huntable case barplot...")
            base_path = save_path.replace('.png', '')
            value_barplot_path = f"{base_path}_value_by_case.png"
            self._create_value_by_huntable_case_barplot(
                observation_data, value_data, value_barplot_path
            )
        
        # 4. Interactive huntable cases t-SNE plot 생성
        if save_path:
            print("\n[4/5] Generating interactive huntable cases t-SNE plot...")
            base_path = save_path.replace('.png', '')
            interactive_huntable_path = f"{base_path}_interactive.html"
            self._create_interactive_huntable_plot(
                observation_data, tsne_results, env, interactive_huntable_path, use_huntable_cases, n_clusters
            )
        
        # 5. Interactive value heatmap t-SNE plot 생성
        if save_path:
            print("\n[5/5] Generating interactive value heatmap t-SNE plot...")
            base_path = save_path.replace('.png', '')
            interactive_value_path = f"{base_path}_value_heatmap_interactive.html"
            self._create_interactive_value_heatmap_plot(
                observation_data, tsne_results, value_data, value_networks, interactive_value_path
            )
        
        print("\nOptimized t-SNE analysis completed!")
    
    def _create_static_tsne_plot(
        self,
        observation_data: Dict[int, Dict[str, List]],
        tsne_results: Dict[int, np.ndarray],
        save_path: str,
        use_huntable_cases: bool
    ):
        """Huntable cases 기반 static t-SNE plot 생성"""
        num_agents = len(observation_data)
        
        # Figure 생성 - agent 수만큼 subplot
        ncols = min(num_agents, 3)
        nrows = (num_agents + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        
        if num_agents == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, agent_id in enumerate(sorted(observation_data.keys())):
            ax = axes[idx]
            
            data = observation_data[agent_id]
            observations = np.array(data['observations'])
            scenarios = np.array(data['scenarios'])
            huntable_cases = np.array(data.get('huntable_cases', []))
            
            if len(observations) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'Agent {agent_id}')
                continue
            
            tsne_result = tsne_results[agent_id]
            
            # Huntable cases 또는 scenarios별로 scatter plot
            if use_huntable_cases and len(huntable_cases) > 0:
                # Huntable cases 사용 (우선순위 순으로 정렬)
                case_priority = {
                    'none': 0,
                    'coop_stag': 1,
                    'coop_stag+my_stag': 2,
                    'coop_stag+my_stag+hare': 3,
                    'my_stag': 4,
                    'my_stag+hare': 5,
                    'hare': 6
                }
                unique_cases = np.unique(huntable_cases)
                sorted_cases = sorted(unique_cases, key=lambda x: case_priority.get(x, 999))
                
                for case in sorted_cases:
                    mask = huntable_cases == case
                    # Case 이름 정규화 (순서에 관계없이 동일한 조합을 같은 키로 매핑)
                    normalized_case = self._normalize_case_name(case)
                    color = self.huntable_colors.get(normalized_case, '#000000')
                    label = self.huntable_labels.get(normalized_case, case)
                    count = mask.sum()
                    
                    ax.scatter(
                        tsne_result[mask, 0],
                        tsne_result[mask, 1],
                        c=color,
                        label=f'{label} (n={count})',
                        alpha=0.6,
                        s=30,
                        edgecolors='white',
                        linewidth=0.5
                    )
            else:
                # Scenarios 사용 (기존 방식)
                for scenario in np.unique(scenarios):
                    mask = scenarios == scenario
                    color = self.scenario_colors.get(scenario, '#000000')
                    label = self.scenario_labels.get(scenario, scenario)
                    count = mask.sum()
                    
                    ax.scatter(
                        tsne_result[mask, 0],
                        tsne_result[mask, 1],
                        c=color,
                        label=f'{label} (n={count})',
                        alpha=0.6,
                        s=30,
                        edgecolors='white',
                        linewidth=0.5
                    )
            
            ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
            ax.set_title(f'Agent {agent_id} Observation Embedding', fontsize=12, fontweight='bold')
            ax.legend(loc='best', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for idx in range(num_agents, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Huntable cases t-SNE plot saved: {save_path}")
        plt.close()
    
    def _create_static_value_heatmap_plot(
        self,
        observation_data: Dict[int, Dict[str, List]],
        tsne_results: Dict[int, np.ndarray],
        value_data: Dict[int, Dict[str, np.ndarray]],
        save_path: str
    ):
        """Value heatmap static t-SNE plot 생성"""
        num_agents = len(observation_data)
        
        # Figure 생성 - agent 수만큼 subplot
        ncols = min(num_agents, 3)
        nrows = (num_agents + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        
        if num_agents == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, agent_id in enumerate(sorted(observation_data.keys())):
            ax = axes[idx]
            
            data = observation_data[agent_id]
            observations = np.array(data['observations'])
            
            if len(observations) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'Agent {agent_id} Value Heatmap')
                continue
            
            tsne_result = tsne_results[agent_id]
            values = value_data[agent_id]['values']
            
            # Red colormap을 사용한 scatter plot
            scatter = ax.scatter(
                tsne_result[:, 0],
                tsne_result[:, 1],
                c=values,
                cmap='Reds',
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )
            
            # Colorbar 추가
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Value Network Output', fontsize=10)
            
            ax.set_xlabel('t-SNE Dimension 1', fontsize=10)
            ax.set_ylabel('t-SNE Dimension 2', fontsize=10)
            ax.set_title(f'Agent {agent_id} Value Heatmap\n(Min: {values.min():.3f}, Max: {values.max():.3f})', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # 빈 subplot 제거
        for idx in range(num_agents, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Value heatmap t-SNE plot saved: {save_path}")
        plt.close()
    
    def _create_value_by_huntable_case_barplot(
        self,
        observation_data: Dict[int, Dict[str, List]],
        value_data: Dict[int, Dict[str, np.ndarray]],
        save_path: str
    ):
        """Huntable case별 평균 value를 막대그래프로 시각화"""
        num_agents = len(observation_data)
        
        # Figure 생성 - agent 수만큼 subplot
        ncols = min(num_agents, 3)
        nrows = (num_agents + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
        
        if num_agents == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, agent_id in enumerate(sorted(observation_data.keys())):
            ax = axes[idx]
            
            data = observation_data[agent_id]
            observations = np.array(data['observations'])
            huntable_cases = np.array(data.get('huntable_cases', []))
            
            if len(observations) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f'Agent {agent_id} Value by Huntable Case')
                continue
            
            values = value_data[agent_id]['values']
            
            # Huntable case별로 value 평균 계산
            if len(huntable_cases) > 0:
                unique_cases = np.unique(huntable_cases)
                
                # Case 이름 정규화
                normalized_cases = [self._normalize_case_name(case) for case in unique_cases]
                
                # Case별 평균 및 표준편차 계산
                case_means = []
                case_stds = []
                case_labels = []
                case_colors = []
                case_counts = []
                
                for case, normalized_case in zip(unique_cases, normalized_cases):
                    mask = huntable_cases == case
                    case_values = values[mask]
                    
                    if len(case_values) > 0:
                        case_means.append(np.mean(case_values))
                        case_stds.append(np.std(case_values))
                        case_labels.append(self.huntable_labels.get(normalized_case, normalized_case))
                        case_colors.append(self.huntable_colors.get(normalized_case, '#808080'))
                        case_counts.append(len(case_values))
                
                if len(case_means) > 0:
                    # 막대그래프 그리기
                    x_pos = np.arange(len(case_labels))
                    bars = ax.bar(x_pos, case_means, yerr=case_stds, 
                                 color=case_colors, alpha=0.8, edgecolor='black', linewidth=1.5,
                                 capsize=5, error_kw={'elinewidth': 2})
                    
                    # 값 표시 (막대 위에)
                    for i, (mean, std) in enumerate(zip(case_means, case_stds)):
                        ax.text(i, mean, f'{mean:.3f}', 
                               ha='center', va='bottom', fontsize=9)
                    
                    # 각 막대 위에 샘플 수 표시 (값 표시 위에)
                    for i, (mean, std, count) in enumerate(zip(case_means, case_stds, case_counts)):
                        ax.text(i, mean + std + 0.01, f'n={count}', 
                               ha='center', va='bottom', fontsize=9, fontweight='bold')
                    
                    # X축 라벨 설정
                    ax.set_xticks(x_pos)
                    ax.set_xticklabels(case_labels, rotation=45, ha='right', fontsize=10)
                    
                    # Y축 라벨 및 제목
                    ax.set_ylabel('Average Value Network Output', fontsize=11, fontweight='bold')
                    ax.set_title(f'Agent {agent_id} - Value by Huntable Case\n' + 
                                f'Total: {len(values)} samples', 
                                fontsize=13, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='y')
                    ax.set_axisbelow(True)
                else:
                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center')
                    ax.set_title(f'Agent {agent_id} Value by Huntable Case')
            else:
                ax.text(0.5, 0.5, 'No huntable case data', ha='center', va='center')
                ax.set_title(f'Agent {agent_id} Value by Huntable Case')
        
        # 빈 subplot 제거
        for idx in range(num_agents, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Value by huntable case barplot saved: {save_path}")
        plt.close()
    
    def _create_interactive_huntable_plot(
        self,
        observation_data: Dict[int, Dict[str, List]],
        tsne_results: Dict[int, np.ndarray],
        env,
        save_path: str,
        use_huntable_cases: bool,
        n_clusters: Optional[int] = None
    ):
        """Interactive huntable cases t-SNE plot 생성"""
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is not available. Please install plotly to use interactive features.")
            return
        
        print("Interactive huntable cases t-SNE plot 생성 중...")
        
        for agent_id, data in sorted(observation_data.items()):
            observations = np.array(data['observations'])
            scenarios = np.array(data['scenarios'])
            huntable_cases = np.array(data.get('huntable_cases', []))
            
            if len(observations) < 2:
                print(f"Agent {agent_id}: 데이터가 부족합니다.")
                continue
            
            print(f"  Agent {agent_id}: {len(observations)} observations")
            
            tsne_result = tsne_results[agent_id]
            
            # Clustering 수행
            if n_clusters is None:
                # Auto clustering: 최적 cluster 수 자동 결정
                actual_n_clusters = self._find_optimal_clusters(tsne_result)
            else:
                # 사용자가 지정한 cluster 수 사용
                actual_n_clusters = n_clusters
                print(f"    K-means clustering (k={actual_n_clusters})...")
            
            cluster_labels, cluster_centers = self._perform_clustering(tsne_result, actual_n_clusters)
            
            # Representative observations 렌더링
            rendered_images = self._render_cluster_representatives(
                observations, tsne_result, cluster_labels, cluster_centers, scenarios, env
            )
            
            # Plotly interactive scatter plot 생성
            fig = self._create_interactive_plot_with_clusters(
                tsne_result, scenarios, cluster_labels, cluster_centers, rendered_images, agent_id
            )
            
            # 저장
            base_path = save_path.replace('.html', '')
            agent_save_path = f"{base_path}_agent_{agent_id}.html"
            
            # Cluster 이미지를 별도 파일로 저장
            save_dir = os.path.dirname(agent_save_path)
            cluster_images_dir = os.path.join(save_dir, f'cluster_images_agent_{agent_id}')
            os.makedirs(cluster_images_dir, exist_ok=True)
            
            for cluster_id, cluster_info in rendered_images.items():
                img_base64 = cluster_info.get('image')
                if img_base64:
                    import base64
                    from PIL import Image
                    from io import BytesIO
                    
                    # Base64 디코딩하여 이미지 저장
                    img_data = base64.b64decode(img_base64)
                    img = Image.open(BytesIO(img_data))
                    img_path = os.path.join(cluster_images_dir, f'cluster_{cluster_id}.png')
                    img.save(img_path)
            
            print(f"    Cluster 이미지 저장: {cluster_images_dir}")
            
            # HTML 저장 (cluster 이미지 포함)
            self._save_html_with_images(fig, agent_save_path, rendered_images, agent_id)
            print(f"    Interactive plot 저장: {agent_save_path}")
    
    def _create_interactive_value_heatmap_plot(
        self,
        observation_data: Dict[int, Dict[str, List]],
        tsne_results: Dict[int, np.ndarray],
        value_data: Dict[int, Dict[str, np.ndarray]],
        value_networks: Dict[int, Any],
        save_path: str
    ):
        """Interactive value heatmap t-SNE plot 생성"""
        if not PLOTLY_AVAILABLE:
            print("Error: Plotly is not available. Please install plotly to use interactive features.")
            return
        
        print("Interactive value heatmap t-SNE plot 생성 중...")
        
        # 각 agent별로 interactive plot 생성
        html_files = []
        
        for agent_id in sorted(observation_data.keys()):
            data = observation_data[agent_id]
            observations = np.array(data['observations'])
            
            if len(observations) == 0:
                print(f"  Agent {agent_id}: No data")
                continue
            
            print(f"  Agent {agent_id}: {len(observations)} observations")
            
            tsne_result = tsne_results[agent_id]
            values = value_data[agent_id]['values']
            
            # Interactive plot 생성
            fig = go.Figure()
            
            # Value heatmap scatter plot
            fig.add_trace(go.Scatter(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                mode='markers',
                marker=dict(
                    size=8,
                    color=values,
                    colorscale='Reds',
                    opacity=0.7,
                    colorbar=dict(title=dict(text="Value Network Output")),
                    line=dict(width=1, color='white')
                ),
                text=[f'Value: {val:.3f}<br>Index: {i}' for i, val in enumerate(values)],
                hovertemplate='<b>Agent %{customdata[0]}</b><br>' +
                            't-SNE 1: %{x:.3f}<br>' +
                            't-SNE 2: %{y:.3f}<br>' +
                            'Value: %{customdata[1]:.3f}<br>' +
                            '<extra></extra>',
                customdata=np.column_stack([
                    np.full(len(observations), agent_id),
                    values
                ]),
                name=f'Agent {agent_id}'
            ))
            
            # Layout 설정
            fig.update_layout(
                title=f'Agent {agent_id} Value Network t-SNE Heatmap<br>' +
                      f'<sub>Min Value: {values.min():.3f}, Max Value: {values.max():.3f}</sub>',
                xaxis_title='t-SNE Dimension 1',
                yaxis_title='t-SNE Dimension 2',
                width=800,
                height=600,
                showlegend=True,
                hovermode='closest'
            )
            
            # 파일 저장
            base_path = save_path.replace('.html', '')
            agent_html_path = f"{base_path}_agent_{agent_id}.html"
            fig.write_html(agent_html_path)
            html_files.append(agent_html_path)
            print(f"    Interactive plot saved: {agent_html_path}")
        
        if html_files:
            print(f"Interactive value heatmap plots generated: {len(html_files)} files")
        else:
            print("No interactive plots were generated.")


def create_tsne_plots(
    evaluator,
    env,
    save_path: Optional[str] = None,
    perplexity: int = 30,
    use_huntable_cases: bool = True,
    n_clusters: Optional[int] = None
):
    """
    한 번의 t-SNE embedding으로 모든 시각화를 생성하는 최적화된 함수.
    
    Args:
        evaluator: CausalEffectEvaluator 인스턴스
        env: 환경 인스턴스
        save_path: 저장 경로 (기본 plot과 value heatmap plot 모두 생성)
        perplexity: t-SNE perplexity 파라미터
        use_huntable_cases: huntable_cases를 사용하여 색상 구분할지 여부
        n_clusters: Cluster 개수 (None이면 auto clustering, 정수면 해당 개수 사용)
    """
    # Observation 데이터 확인
    if not hasattr(evaluator, 'observation_data') or len(evaluator.observation_data) == 0:
        print("Error: evaluate_causal_effects()를 먼저 실행해야 합니다.")
        return None
    
    # ObservationAnalyzer 생성
    analyzer = ObservationAnalyzer(
        num_agents=evaluator.algorithm.num_agents,
        perplexity=perplexity,
        max_iter=1000,
        random_state=42
    )
    
    print("\n" + "="*80)
    print("Optimized t-SNE Analysis (Single Embedding for All Plots)")
    print("="*80)
    
    # Value networks 추출 (개별 네트워크와 parameter sharing 모두 지원)
    from ..utils.value_network_utils import get_value_networks_dict
    value_networks = get_value_networks_dict(evaluator.algorithm)
    
    # 한 번의 t-SNE embedding으로 모든 시각화 생성
    analyzer.plot_all_tsne_visualizations(
        evaluator.observation_data,
        value_networks,
        env,
        save_path=save_path,
        use_huntable_cases=use_huntable_cases,
        perplexity=perplexity,
        n_clusters=n_clusters
    )
    
    return None

