# PCI-MAPPO 실행 명령어들


## 자주 쓰이는 command

```bash
python main.py --config configs/dec_tiger.yaml --env DecTiger --device cuda --max-steps 128
python main.py --config configs/stag_hunt.yaml --env StagHunt --device cuda --max-steps 128
python main.py --config configs/toy_causal.yaml --env Toy --device cuda --max-steps 128
```

## 기본 테스트 및 실행

### 1. 예제 실행 (프로젝트 동작 확인)
```bash
python example_usage.py
```

### 2. 간단한 트레이닝 테스트
```bash
python main.py --config configs/small_scale.yaml
```

### 3. 기본 설정으로 트레이닝
```bash
python main.py --config configs/default.yaml
```

### 4. Weights & Biases와 함께 트레이닝
```bash
python main.py --config configs/default.yaml --use-wandb
```

### 5. GPU 사용 (CUDA 사용 가능한 경우)
```bash
python main.py --config configs/default.yaml --device cuda
```

## 고급 설정

### 6. 커스텀 하이퍼파라미터
```bash
python main.py --num-agents 4 --causal-lambda 0.2 --max-iterations 500
```

### 7. 다른 환경 설정
```bash
python main.py --env simple_cooperative --num-agents 5 --grid-size 7
```

## 테스트 및 평가

### 8. 트레이닝된 모델 테스트
```bash
python test.py --checkpoint checkpoints/best_model.pt
```

### 9. 렌더링과 함께 테스트
```bash
python test.py --checkpoint checkpoints/best_model.pt --render
```

### 10. 평가 모드
```bash
python main.py --eval --checkpoint checkpoints/best_model.pt
```

## 원클릭 실행

### 11. 모든 예제 실행 (Linux/Mac)
```bash
chmod +x run_example.sh
./run_example.sh
```

### 12. Windows에서 실행
```cmd
python example_usage.py
python main.py --config configs/small_scale.yaml --max-iterations 50
```

## 디버깅 및 모니터링

### 13. 상세한 로깅
```bash
python main.py --config configs/small_scale.yaml --log-interval 1
```

### 14. 체크포인트에서 재시작
```bash
python main.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_100.pt
```

## 권장 실행 순서

1. **먼저 예제 실행**: `python example_usage.py`
2. **간단한 테스트**: `python main.py --config configs/small_scale.yaml`
3. **결과 확인 후 본격 트레이닝**: `python main.py --config configs/default.yaml`
4. **트레이닝된 모델 테스트**: `python test.py --checkpoint checkpoints/best_model.pt`
