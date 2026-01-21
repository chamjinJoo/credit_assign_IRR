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
    ✔️use **real counterfactual baseline without factual action** (details on below 2026.01.15)

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
* Parallel Env 구현 중...
* compute_causal_intrinsic_reward에서 available action 고려토록 하기
* StagHunt task에서 `coop-bonus`를 조금이라도 주면서 테스트 해보기? (`coop-bonus=0`은 너무 `pci-marl`을 위한 세팅인 느낌이 들 수도 있으니)