## 2025.12.12
* Test a new method: 
    
    set agent-wise proxy $W^i_t = O^{-i}_t, Z^i_t = O_{t-1}$ (unchanged) -> 가능성 있음

## 2025.12.15
* Outcome Bridge Network -> MLPBase
* Value Network -> no PS
* Adaptive causal lambda 

**Bold** is best performance
1. bridge network output layer extended | ValuenetPS                      X
2. bridge network output layer extended | ValuenetPS |adaptive            X
3. bridge network output layer extended | ValuenetPS                      X
4. bridge network output layer extended | ValuenetPS | adaptive           X
5. bridge network | ValuenetPS | adaptive                                 X
6. bridge network | ValuenetPS                                            X
7. bridge network | ValuenetNoPS                                          O
8. bridge network | ValuenetNoPS | adaptive                               O
9. **bridge network | ValuenetNoPS | Causal Reciprocity MAX**             O
10. bridge network | ValuenetNoPS | Causal Reciprocity MEAN               O
11. bridge network | ValuenetNoPS | Causal Reciprocity MAX | adaptive     X
12. bridge network | ValuenetNoPS | Causal Reciprocity MEAN| adaptive     X

## 2025.12.22
**Establish a new method**
1. When calculating causal intrinsic rewards, use factual reward instead of predictions.
2. Also consider factual action when calculating baseline:
    * Before: $\delta^i_t = \hat{\boldsymbol r}^{-i}(\mathbf{o}^{-i}_t,a^i_t) - \mathbb E_{a\sim\pi^i,{a}\neq{a^i_t}}[\hat{\boldsymbol r}(\mathbf{o}^{-i}_t,a)]$
    * After: $\delta^i_t = \boldsymbol r^{-i}_t - \mathbb E_{a\sim\pi^i}[\hat{\boldsymbol r}(\mathbf{o}^{-i}_t,a)]$

    2-1. basline_type = 'weighted' & merging_method = 'max' 

3. Use parametersharing for value network.
4. Do not use adaptive lambda.
5. Do not use Causal reciprocity & synergy factor. 

## 2026.01.02
We checked that it works in Stag-Hunt environment without mauling punishment. 

## 2026.01.08
* We checked that q-bridge function have been trained with Entropy, not action probability $\pi^i(\cdot|o^i_t)$. 
* Training q-bridge function takes a lot of time. In Stag-Hunt 4-agent, w/ -> 6.50s/iter w/o -> 2.94s/iter
* q-bridge function can be elimiated without theoretical loss.

## 2026.01.09
* We checked that applying clipping on $\delta^i_t$ provides good performance. The current configuration is `Delta_final[torch.abs(Delta_final) < 0.2] = 0.0`

## 2026.01.12
* Add `success rate` for WandB logging 
* Test $Z_t = \mathbf{r}_{t-1}$ , the **convergence speed decreases**.
* For clearity of `mean_epsiode_length` plot, set max-timestep = 1000
* Test new observation in StagHunt: obs = location of each object = [Agents, Stags, Hares]. The locations of objects out of 5x5 vision is masked as -1. => **Does not work**
* Test new grid_observation in StagHunt: masking relative positions if it is out of 5x5 around an agent. => 성능이 떨어지지만 나름 괜찮음. Baseline과의 차이 유지, `merging_method = sum`시 성능이 좋음. 또한 단순한 reward predictor를 사용하는 x-pci와도 약간의 성능 차이를 보임.
* `clip_delta`값은 Stag-Hunt에서는 `0.2`가 best인듯 (0.1, 0.5는 성능 저하)
* Stag-Hunt의 Obs에 0.01, 0.05, 0.08의 gaussian noise를 넣어봄. 0.05부터는 x-pci와 pci의 수렴 속도와 성능 차이가 조금 남.

## 2026.01.13
* Stag-Hunt에서 거짓된 $a^i_t \to \mathbf{r}^{-i}_t$ correlation을 난이도 metric으로 보고, hare 수를 늘려가면서 실험해보기.
* 다양한 Z를 확인해봄. Z = (prev_joint_obs, prev_joint_actions) or (prev_joint_obs, prev_joint_rewards). 기존 Z = prev_joint_obs가 젤 나음. 
* use `max-timestep=2000` and `iteration=1500`. more stable

## 2026.01.14
* New Baseline: **COMA** - keep testing
* [Remind]: Using r_hat_factual for intrinsic reward leads to **poor performance**. 
* Develop **RNN-based models**  `pci_rnn`, `mappo_rnn`, and `ippo_rnn`.
    - They used RNN hidden state $\eta_t$ for policy and value network, $a^i_t \sim \pi^i(\cdot|\eta_t), V_\theta(\eta_t)$
    - **All models perform better** since **RNN is a well-known empirical solution to address partial observability**
    - `mappo_rnn`의 성능이 너무 좋아져서 `pci_rnn`을 따라잡음.
* Some proposals for development of `PCI_RNN`:
    - add covariates $X=o^i_t$ on outcome bridge $h$ → $h(a,w,x)$
    - test various $Z_t=\mathbf{o}_{t-1} \text{ or } \mathbf{r}_{t-1}$
    ✔️ use **real counterfactual baseline without factual action** (details on below 2026.01.15)

## 2026.01.15
* Test counterfactual baseline only with counterfactual actions — very good, outperforms x-pci
    * Intuition of the problem in the original baseline: As policy training goes on, the baseline should be decreased. Therefore, **the shaped reward also decreases even though it represent the same cooperation level.** It leads the **confusion in downstream training** (value & policy).
* What is the best combination among the four cases, $Z = \{\mathbf{r}_t, \mathbf{o}_{t-1}\}, X = \{o^i_t, None\}$?
    * `delta_clipping` not that much matters — keep for reduce noise
    * $Z = \mathbf{r}_{t-1}$ — keep for computational cost, not for performance.
    * $X = o^i_t$ gives faster convergence, not guarantees better performance. — keep for theoretical matching?
* Test Stag-Hunt **w/o relative positions** and expand vision 3x3 → **5x5**.
    * PCIMARL-RNN outperforms baseline rMAPPO. When N=4, rMAPPO has never completed the task.

## 26.01.19
* Add SMAC and SMACv2 envs.
    * add action masking logics to prevent unavailable actions (e.g., choose attack when there is no enemy in the attack range)
    * Solve the NaN error in policy logits.

## 26.01.20
* Parallel Env 구현 중 
* compute_causal_intrinsic_reward에서 available action 고려토록 하기
* StagHunt task에서 `coop-bonus`를 조금이라도 주면서 테스트 해보기? (`coop-bonus=0`은 너무 `pci-marl`을 위한 세팅인 느낌이 들 수도 있으니)

## 26.01.21
* [NOTE] Parallel Env implementaion is done!
    * Verfied in Stag-Hunt, SMAC, and SMACv2.
    * trainig time is reduced by half.
* `pci-marl`의 clip_delta도 argparse로 조정할 수 있게 수정
    * `clip_delta` is for filtering noise, so set it to a tiny value like 1e-2 or 1e-3 scale.
* SMAC에서의 reward값이 0.01 scale까지 나오는 것을 확인 -> `clip_delta=0.005`로 세팅해봄.
* Found that bridge networks training does not use max_grad_norm clipping. Add it. -> But not effective.
* **Hyperparameter**: `lr_bridge=1e-3` and `clip_delta=0.01` for SH. 

## 26.01.22
* SMACv2 Benchmarking
    * Settings: keep the most configs in `configs/smacv2_configs_examples/[...].yaml`. 
    * When using `[...]_epo.yaml` We can change the `prob_obs_enemy=[0.0, 0.5, 1.0]` for partial observability while keeping `action_mask=False`. This is because SMACv2 paper shows that `action_mask=True` makes `prob_obs_enemy` meaningless.
* We observed that `mappo-rnn` degrades when `global_reward=False` where critic and policy trained with individual rewards.
* `StagHunt` with `coop-bonus` 
    * `1.0` -> `PCI-MARL` outperformes. Overall trends are similar to no `coop_bonus` case.
    * `3.0` -> `PCI-MARL` outperformes.
    * `5.0` -> `PCI-MARL` outperformes.
* `rIPPO` with `global_reward=True` shows higher performance than `global_reward=False`, but unstable over different seeds.

# 26.01.23
* SMACv2: `ppo_epochs=2` -> `ppo_epoch=10` following the SMACv2 paper. -> better but still not good.

# 26.01.27
* Proxy Z issue while using observational history $\eta$ for policy.
    * consider $Z^i_t = \eta^i_{t-1}$. -> not satisfies conditional indep. condition
    * keep $Z^i_t = \mathbf{r}_{t-1}$, change $X^i_t = \{o^i_t, \mathbf{a}^{-i}_t\}$. -> conditional indep. for W is not satisfied.
    * keep $Z^i_t = \mathbf{r}_{t-1}$, **change $X^i_t = \{o^i_t, \mathbf{h}^{-i}_{t-1}\}$**. -> both conditional indep. are satisfied.

# 26.01.28
* Test `StagHuntEnv` without coord information in obs -> good

# 26.02.01
* Update the save_replay logic for SMACv2 -> watch replay on Window via Starcraft2.
* set outcome Y as the whole reward vector $\mathbf{r}_t$, thanks to $\mathbf{h}^{-i}_t \in X^i_t$.
* Build a new pci method `q_pci_rnn`: outcome bridge function predict Q-values. -> poor performance

# 26.02.02
* `pci-rnn`의 value network parameter sharing logic was wrong. Fix it.
* `pci-rnn` shows poor performance in `SMAC, SMACv2` envs. How to improve?
    1) decrease causal_lambda : `shaped_rewards_all = (1-causal_lambdas) * rewards_batch + causal_lambdas * delta`
        * `causal_lambda=0.1` outperforms `0.5` and `1.0` on `SMAC`    
    2) `10gen_protoss` scenario seems to be easier than `10gen_zerg`

# 26.02.03
* Found that outcome_bridge_network $h_{\theta}$ has hidden state dimension of `2*args.hidden_dim` -> edit as `args.hidden_dim`

# 26.02.04
* `SMACv2` change batch_size: $1600 \to 3200$ (number of steps per iter), keep the total steps for training $\approx 10^7$. 
* Baseline: **COMA** (rnn verision), testing...
    * Target critic update interval: default=10

# 26.02.05
* Complete `EpisodeBuffer` -> if `episodes_per_iteration` is set, automatically use EpisodeBuffer.
* Cleanup the config for different envs.

# 26.02.10
* **Correct Defect**: `PPO`-related algorithms re-compute old policy prob. when policy updates -> fix it by storing old_policy_probs while collecting rollout data.
    * `rMAPPO` shows reasonable performance in `StagHunt`, but `rPCI` still outperforms.
* **Refactoring**:
    * Rename RNN-based algorithms to `r_` prefix convention (e.g., `pci_rnn` -> `r_pci`, `mappo_rnn` -> `r_mappo`).
    * Update scripts and `main.py` to use new names.

# 26.02.11
* In `SMACv2`
    * `max_grad_norm` 0.5 -> 10: does not matter.
    * `clip_epsilon` 0.2 ->: does not matter.
    * **Batch size 3200 -> 6400**: makes differences. the performance is closed to benchmarking results.
* Note that the reproductability in SMACv2 requires more corrections, need to deliever the seed to `Distribution` class for SMACv2.

# 26.02.12
* Critical theoretic defect: when utilizing outcome bridge function, we should calculate the conditional expectattion $\mathbb{E}_{w\sim p(w|z,a,x)}[h(a,w,x)|Z,A,X]$. But we did not.
    * justifying  as a single-sample MonteCarlo?

# 26.02.20
* `SMACv2` test 
    * $x=(o^i_t, \mathbf{h}_{t-1})$ with EPO setting: not significant
    * RBFKernel gamma: 1.0 -> 0.5: not significant, unstable
    * clipping negative delta like ReLU: not significant
    * PCI-rMAPPO (with centralized critic taking joint info.): not significant

# 26.02.23
* Problems in `SMACv2` with PCI-MARL
    * shaped reward: Attack aciton과 Non-attack action에 대한 shaped reward가 부정확함. (e.g., 타 agent의 attack으로 내가 +1.0 reward를 받으면 non-attack action에 대한 prediction은 +1.0에 가깝게 나와야 하는데, 지금은 +2.0로 나오는 상황)

# 26.02.24
* RNN change: obs history -> (obs, action) history, $h^i_t = (o^i_0, a^i_0, ..., a^i_{t-1}, o^i_t)$
* action expression change: embedding -> one-hot. Embedding is more fit to compress high-dimensional data.

# 26.02.28
* SMACv2 - `set_gamma()` and `baseline_type='averaged'` have no effect.