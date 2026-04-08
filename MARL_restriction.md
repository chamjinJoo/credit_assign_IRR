## MARL Baseline의 근본적 허점 ##
*  결론부터: "상관관계와 인과관계를 구분하지 못한다" 는 것입니다. 더 정확히는: 개별 agent의 action이 reward를 일으켰는지 vs reward와 동시에 발생했는지 를 구분하지 못한다.

* 형식적 진술 - MAPPO/IPPO가 추정하는 credit:
    $$Q(s, a_i) = \mathbb{E}{a{-i} \sim \pi_{-i}}[R \mid s, a_i]$$
  * 이것은 "teammate들의 행동을 평균낸 상태에서 $a_i$를 했을 때의 기대 reward" 입니다. 즉 marginalization을 통해 teammate의 실제 행동을 지워버립니다.

* 반면 실제로 필요한 것:
    $$\text{Causal credit}i = \mathbb{E}[R \mid do(a_i), a{-i}^{\text{actual}}]$$
    * teammate들이 실제로 한 행동을 고정한 채, 내 행동이 바뀌었을 때 reward가 어떻게 달라지는가.

* Target Allocation이 이 허점을 어떻게 드러내는가
  * 3m window=2에서: 
    $$Q(s, a_0 = attack_enemy_1) = P(ally_1 attacks enemy_1) × p_kill ×  r$$
    $$Q(s, a_0 = attack_enemy_0) = P(ally_2 attacks enemy_0) × p_kill × r$$
  * 초기 random policy에서 두 확률은 대칭 → 두 action의 advantage 차이가 거의 0.
  * 이 상태에서 baseline은 무엇을 학습해야 할지 gradient signal이 없습니다. 더 많은 데이터를 모아도 해결되지 않습니다 — marginalization이 원리적으로 신호를 지우기 때문입니다.

* 허점의 두 층위
  * 첫 번째: 결합 인과 구조(conjunctive causation)에서의 신호 희석
    R = 1  IFF  (A_i = attack_j)  AND  (A_k = attack_j)
    단독 행동으로는 reward를 만들 수 없는 구조에서, 한 agent의 marginal Q값은 상대의 협력 확률에 희석됩니다. "나 혼자 잘해봤자" 인 상황인데, baseline의 credit은 "나 혼자 했을 때의 평균값"을 씁니다.
  * 두 번째: 인수분해된 정책(factored policy)의 표현력 한계
    MAPPO/IPPO는 $\pi(a_1, \ldots, a_n) = \prod_i \pi_i(a_i \mid o_i)$ 를 학습합니다. 그러나 target allocation의 최적 전략은 상관 전략(correlated strategy) 입니다:
    "ally_1이 enemy_1을 공격할 것 같으면 나도 enemy_1을 공격하라" 이것은 인수분해된 정책으로 표현할 수 없습니다 — $\pi_i(a_i \mid o_i)$는 $\pi_j$의 현재 선택에 조건화할 방법이 없습니다. 결국 무작위한 하나의 target을 선택하는 unimodal policy로 수렴하거나, 불안정한 진동을 보입니다.

* RPCI가 이 허점을 피하는 방법 
    * RPCI의 $\delta_i$는 marginalization을 하지 않습니다:
        $\delta_i \approx \mathbb{E}[R \mid a_i = attack_j ; R = 1] - \text{baseline}$, 
        $R=1$로 conditioning하는 것이 핵심입니다. 
    * SCM 관점에서:
        $R=1$을 관찰하면 
        
        → collider $D_j$가 활성화 
        
        → $A_i$와 $A_k$가 d-connected "내가 enemy_1을 공격했고 reward가 났다" 
       
        → ally_1도 공격했음이 후향 추론됨 

        이 조건부 신호는 "내 action이 실제로 인과적이었던 timestep"만 골라서 credit을 부여 즉, "reward와 동시에 발생한 action"이 아니라 "reward를 실제로 일으킨 action"에 credit을 줍니다.
* 요약
    MAPPO/IPPO	RPCI
    Credit 기준	$P(R \mid a_i)$ — 상관관계	$P(R \mid do(a_i), R=1)$ — 인과관계
    Teammate 처리	marginalize (평균)	actual action 조건화
    표현 가능한 정책	factored (독립)	causal advantage로 correlated 행동 유도
    Target allocation에서	신호 희석, 수렴 불안정	collider conditioning으로 신호 강화
    StagHunt와 target allocation SMAX는 이 허점이 최대로 노출되도록 설계된 환경이고, RPCI는 그 허점을 causal conditioning으로 우회하는 알고리즘입니다.