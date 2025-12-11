# PCI-MAPPO: Proximal Causal Inference with Multi-Agent Proximal Policy Optimization

PCI-MAPPO는 다중 에이전트 강화학습에서 인과 추론을 활용한 새로운 알고리즘입니다.

## 지원하는 환경

### 1. SimpleCooperativeEnv
- **설명**: 간단한 협력 환경으로, 에이전트들이 목표에 도달하면서 협력 보너스를 받는 게임
- **에이전트 수**: 3개
- **액션**: 상, 하, 좌, 우, 제자리
- **관찰**: 에이전트 위치, 목표 위치, 다른 에이전트 위치

### 2. ToyCausalEnv
- **설명**: 인과 추론을 위한 토이 환경으로, 인과적 사건과 교차 에이전트 신호를 포함
- **에이전트 수**: 2개
- **액션**: 좌, 우
- **관찰**: 위치, 고보상 신호, 교차 에이전트 신호, 보상 위치

### 3. DecTigerEnv
- **설명**: Dec-Tiger 협력 게임 환경으로, 두 에이전트가 협력하여 호랑이가 있는 문을 찾아야 함
- **에이전트 수**: 2개
- **액션**: 왼쪽 문 열기, 오른쪽 문 열기, 듣기
- **관찰**: 듣기 결과 (왼쪽/오른쪽)

### 4. StagHuntEnv
- **설명**: Stag-Hunt 협력 게임 환경으로, 두 에이전트가 협력하여 Stag를 사냥하거나 개별적으로 Hare를 사냥할 수 있음
- **에이전트 수**: 2개
- **액션**: 상, 하, 좌, 우, 사냥
- **관찰**: 3x3 그리드 (에이전트 주변 지역)
- **보상 시스템**:
  - Hare 사냥: +1 (개별 보상)
  - Stag 사냥 (협력): +5 (두 에이전트 모두)
  - Stag 사냥 (혼자): -5 (페널티)

## 사용법

### 기본 사용법

```python
from pci_mappo.envs import StagHuntEnv
from pci_mappo.algorithms import PCIMAPPOAlgorithm

# 환경 생성
env = StagHuntEnv(max_steps=50, stag_move_prob=0.7)

# 알고리즘 생성
algorithm = PCIMAPPOAlgorithm(
    num_agents=env.num_agents,
    obs_dim=env.get_obs_dim(),
    action_dim=env.get_action_dim(),
    w_dim=env.get_obs_dim() + 1,
    hidden_dim=64,
    rnn_hidden_dim=32,
    causal_lambda=0.1,
    causal_lambda_schedule=True,  # Causal lambda 스케줄링 활성화
    causal_lambda_start=0.0,      # 시작 causal lambda 값
    causal_lambda_final=0.2,      # 최종 causal lambda 값
    causal_lambda_warmup_steps=10000  # 스케줄링 완료까지의 스텝 수
)

# 에피소드 실행
observations = env.reset()
algorithm.reset_agents()

for step in range(100):
    actions = algorithm.get_actions(observations)
    next_observations, rewards, dones, info = env.step(actions)
    
    if any(dones.values()):
        break
```

### 설정 파일 사용

```bash
# Stag-Hunt 환경으로 학습
python main.py --config configs/stag_hunt.yaml

# 커맨드 라인으로 환경 지정
python main.py --env StagHuntEnv --max-steps 50 --stag-move-prob 0.7
```

### 테스트

```bash
# Stag-Hunt 환경 테스트
python examples/test_stag_hunt.py

# 전체 예제 실행
python example_usage.py
```

### Causal Effect 평가

학습된 Bridge network의 causal effect 추정 성능을 평가할 수 있습니다:

```bash
# 기본 causal effect 평가 (50 에피소드)
python main.py --causal-eval --checkpoint checkpoints/baseline/staghuntenv/checkpoints/algorithm.pt --config configs/stag_hunt.yaml

# 더 많은 에피소드로 평가 (100 에피소드)
python main.py --causal-eval --checkpoint checkpoints/baseline/staghuntenv/checkpoints/algorithm.pt --config configs/stag_hunt.yaml --eval-episodes 100

# 또는 전용 스크립트 사용
python evaluate_causal_effects.py --checkpoint checkpoints/baseline/staghuntenv/checkpoints/algorithm.pt --config configs/stag_hunt.yaml --eval-episodes 100
```

평가 결과는 다음 파일들로 저장됩니다:
- `causal_effect_statistics.json`: 통계 데이터 (평균, 표준편차, 샘플 수 등)
- `causal_effect_raw_data.json`: 원시 causal effect 데이터
- `causal_effect_plot.png`: Error bar plot 시각화

## 환경 레이아웃 (Stag-Hunt)

```
#########
##1   2##
## # # ##
#H  S  H#
## # # ##
##3   4##
#########

#: 벽
H: Hare (고정 위치)
S: Stag (매 timestep마다 움직임)
1,2,3,4: 에이전트 초기 위치 (랜덤 선택)
```

## 주요 특징

- **Partially Observable**: 각 에이전트는 3x3 그리드만 관찰
- **협력 메커니즘**: Stag는 두 에이전트가 동시에 사냥해야 성공
- **개별 보상**: Hare는 혼자서도 사냥 가능
- **동적 환경**: Stag가 매 timestep마다 움직임
- **충돌 처리**: 벽, 다른 에이전트, Stag, Hare와의 충돌 고려

## Causal Lambda 스케줄링

PCI-MAPPO는 인과적 보상 형성(causal reward shaping)을 위한 스케줄링 기능을 제공합니다:

### 스케줄링 파라미터

- `causal_lambda_schedule`: 스케줄링 활성화 여부 (기본값: False)
- `causal_lambda_start`: 시작 causal lambda 값 (기본값: 0.0)
- `causal_lambda_final`: 최종 causal lambda 값 (기본값: 0.2)
- `causal_lambda_warmup_steps`: 스케줄링 완료까지의 스텝 수 (기본값: 10000)

### 스케줄링 동작

- **초기**: causal lambda = causal_lambda_start
- **진행**: 선형적으로 증가 (causal_lambda_start → causal_lambda_final)
- **완료**: causal lambda = causal_lambda_final (최종값 유지)

### 설정 예시

```yaml
algorithm:
  causal_lambda: 0.1
  causal_lambda_schedule: true      # 스케줄링 활성화
  causal_lambda_start: 0.0          # 시작값
  causal_lambda_final: 0.2          # 최종값
  causal_lambda_warmup_steps: 10000 # 워밍업 스텝 수
```

이 스케줄링은 학습 초기에는 기본 보상에 집중하고, 점진적으로 인과적 추론을 도입하여 안정적인 학습을 가능하게 합니다.

## 설치 및 실행

```bash
# 의존성 설치
pip install -r requirements.txt

# 학습 실행
python main.py --config configs/stag_hunt.yaml

# 테스트 실행
python examples/test_stag_hunt.py
```

