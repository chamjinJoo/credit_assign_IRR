# PCI-MAPPO 실행 명령어들


## 자주 쓰이는 command
### training.

```bash
python main.py --config configs/env_name.yaml --env env_name --device cuda/cpu --max-steps 128 --max-iterations 1000 --num-agents 2 --log-interval 1 --use-wandb
```
# examples
```bash
python main.py --env StagHunt --config configs/stag_hunt.yaml --num-agents 3 --device cuda --max-steps 1024 --max-iterations 3000 --log-interval 1 --use-wandb
```

## 테스트 및 평가

### 평가 모드
```bash
python main.py --test --env StagHunt --config configs/stag_hunt.yaml --num-agents 3  --checkpoint checkpoints/best_model.pt
```

### 체크포인트에서 재시작
```bash
python main.py --config configs/default.yaml --checkpoint checkpoints/checkpoint_100.pt
```