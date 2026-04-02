# PCI Outcome Bridge Summary

이 문서는 [`r_pci.py`](./algorithms/r_pci.py) 와 [`q_pci.py`](./algorithms/q_pci.py) 에서 `Outcome Bridge Network`가

- 무엇을 학습하는지
- 어떤 counterfactual quantity를 계산하는지
- 그 결과가 actor / critic에 어떻게 사용되는지

를 `수정 전`과 `수정 후`로 나누어 비교 정리한 메모다.

---

## 1. 한눈에 보는 변화

| Algorithm | 수정 전 | 수정 후 |
| --- | --- | --- |
| `r_pci` | bridge는 **즉시 reward contribution**을 학습하고, 이를 **reward shaping**에 직접 사용 | bridge는 여전히 **즉시 reward contribution**을 학습하지만, 결과는 **advantage-level PCI**로만 사용 |
| `q_pci` | bridge는 **team return**을 직접 학습하고, 주로 **baseline shift**에 사용 | bridge는 **bootstrapped team target**을 학습하고, 계산된 **causal advantage**를 actor에 직접 주입 |

가장 중요한 차이는 다음 두 식으로 요약된다.

수정 전 `r_pci`:

$$
\tilde r_t^{(i)}
=
(1-\lambda)\,\frac{r^{\text{team}}_t}{N}
\;+\;
\lambda\,\delta_t^{(i)}
$$

수정 후 `r_pci`, `q_pci`:

$$
A^{(i)}_{\text{total}}
=
\operatorname{Norm}\!\left(A^{(i)}_{\text{env}}\right)
\;+\;
\lambda\,
\operatorname{Orth}\!\left(
\operatorname{Norm}\!\left(A^{(i)}_{\text{pci}}\right)
\;\middle|\;
\operatorname{Norm}\!\left(A^{(i)}_{\text{env}}\right)
\right)
$$

즉, **reward 자체를 바꾸는 방식**에서 **policy advantage만 보정하는 방식**으로 이동했다.

---

## 2. `r_pci.py`

## 2.1 수정 전

### Bridge가 학습한 대상

`r_pci`의 outcome bridge는 agent \(i\)의 action이 각 agent의 **즉시 reward 벡터**에 미치는 영향을 학습했다.

$$
h_\phi(w_t^{(i)}, a_t^{(i)}, x_t^{(i)})
\approx
\mathbb{E}\!\left[
\mathbf r_t
\mid
do(a_t^{(i)}),\, w_t^{(i)},\, x_t^{(i)}
\right]
\in \mathbb{R}^{N}
$$

여기서

- \(w_t^{(i)}\): 다른 agent들의 observation proxy
- \(x_t^{(i)}\): agent \(i\)의 local observation과 hidden-state covariate
- \(\mathbf r_t\): 각 agent의 즉시 reward 벡터

이다.

즉, bridge는 본질적으로

> "agent \(i\)가 action \(a\)를 했을 때 이번 step에서 팀 내 각 agent reward가 어떻게 달라질까?"

를 학습했다.

### Bridge output의 활용 방식

가능한 모든 action \(a\)에 대해 reward prediction을 만들고,
factual action을 제외한 counterfactual baseline을 계산한 뒤,
다음 effect를 만들었다.

$$
\delta_t^{(i \to j)}
=
r_t^{(j)}
-
\sum_{a \neq a_t^{(i)}}
\pi_i(a \mid h_t^{(i)})
\hat r_t^{(j)}(a)
$$

그리고 agent \(i\)가 다른 agent 전체에 미친 effect를 합쳐

$$
\delta_t^{(i)}
=
\sum_{j=1}^{N}
\delta_t^{(i \to j)}
$$

를 만든 뒤, 이를 reward shaping에 직접 사용했다.

$$
\tilde r_t^{(i)}
=
\frac{(1-\lambda)\,r_t^{\text{team}}+\lambda\,\delta_t^{(i)}}{N}
$$

그 결과:

- actor는 shaped reward 기반 GAE를 사용
- critic도 shaped reward return을 맞추도록 학습

했다.

즉, **bridge error가 actor와 critic 둘 다에 직접 전파되는 구조**였다.

---

## 2.2 수정 후

### Bridge가 학습한 대상

수정 후에도 `r_pci`의 outcome bridge가 학습하는 quantity 자체는 같다.

$$
h_\phi(w_t^{(i)}, a_t^{(i)}, x_t^{(i)})
\approx
\mathbb{E}\!\left[
\mathbf r_t
\mid
do(a_t^{(i)}),\, w_t^{(i)},\, x_t^{(i)}
\right]
$$

즉, bridge는 여전히 **즉시 reward contribution estimator**다.

### Bridge output의 활용 방식

이제 bridge output은 reward를 바꾸지 않는다.
대신 먼저 동일하게

$$
\delta_t^{(i)}
=
\sum_{j=1}^{N}
\left(
r_t^{(j)} - \text{CFBaseline}_t^{(i \to j)}
\right)
$$

를 계산한 뒤, 이를 **actor 전용 PCI signal**로 바꾼다.

먼저 환경 보상으로 만든 기존 advantage:

$$
A^{(i)}_{\text{env}}
=
\operatorname{GAE}\!\left(
\frac{r_t^{\text{team}}}{N},\, V_i
\right)
$$

다음으로 `delta`를 시간축으로 누적한 PCI advantage:

$$
A^{(i)}_{\text{pci}}
=
\operatorname{GAE}\!\left(
\delta_t^{(i)},\, 0
\right)
$$

그 다음 두 신호를 정규화하고, PCI 신호에서 환경 advantage와 겹치는 성분을 제거한다.

projection:

$$
\operatorname{Proj}_{A_{\text{env}}}(A_{\text{pci}})
=
\frac{
\langle A_{\text{pci}}, A_{\text{env}} \rangle
}{
\langle A_{\text{env}}, A_{\text{env}} \rangle + \varepsilon
}
A_{\text{env}}
$$

orthogonal residual:

$$
A^{(i)}_{\text{pci,res}}
=
A^{(i)}_{\text{pci}}
-
\operatorname{Proj}_{A_{\text{env}}}(A^{(i)}_{\text{pci}})
$$

최종적으로 actor는

$$
A^{(i)}_{\text{total}}
=
\operatorname{Norm}\!\left(A^{(i)}_{\text{env}}\right)
+
\lambda\,
\operatorname{Orth}\!\left(
\operatorname{Norm}\!\left(A^{(i)}_{\text{pci}}\right)
\;\middle|\;
\operatorname{Norm}\!\left(A^{(i)}_{\text{env}}\right)
\right)
$$

를 사용하고,
critic은 여전히 raw team reward 기반 return을 맞춘다.

즉:

- actor: PCI 보정 advantage 사용
- critic: 원래 reward return 사용

으로 역할이 분리되었다.

### 직관

수정 후 `r_pci`는

> "bridge가 알려주는 credit assignment 정보는 actor에게만 보조 신호로 주고, critic은 원래 task return을 그대로 학습시킨다"

라는 구조가 되었다.

따라서 bridge 추정 오차가 critic을 불안정하게 만드는 문제가 줄어든다.

---

## 3. `q_pci.py`

## 3.1 수정 전

### Bridge가 학습한 대상

수정 전 `q_pci`의 outcome bridge는 즉시 reward가 아니라 **team return** 자체를 학습했다.

$$
h_\phi(w_t^{(i)}, a_t^{(i)}, x_t^{(i)})
\approx
\mathbb{E}\!\left[
G_t^{\text{team}}
\mid
do(a_t^{(i)}),\, w_t^{(i)},\, x_t^{(i)}
\right]
$$

여기서

$$
G_t^{\text{team}}
=\frac{1}{N}
\sum_{k=t}^{T-1}
\gamma^{k-t}
r_k^{\text{team}}
$$

이다.

즉, bridge는

> "agent \(i\)의 현재 action이 팀의 미래 return 전체에 어떤 영향을 줄까?"

를 직접 회귀하려 했다.

### Bridge output의 활용 방식

모든 action에 대한 bridge value를 계산해

$$
\hat Q_i(a)
=
h_\phi(w_t^{(i)}, a, x_t^{(i)})
$$

를 만들고, 현재 policy 하의 bridge-implied value

$$
\hat V^{\text{bridge}}_i
=
\sum_a \pi_i(a \mid h_t^{(i)}) \hat Q_i(a)
$$

와

$$
V_i^{\text{LSE}}
=
\tau \log \sum_a \exp(\hat Q_i(a)/\tau)
$$

를 계산했다.

또한 counterfactual advantage도 만들긴 했다.

$$
A^{(i)}_{\text{pci}}
=
\hat Q_i(a_t^{(i)})
-
\sum_{a \neq a_t^{(i)}}
\tilde \pi_i(a)\,\hat Q_i(a)
$$

하지만 수정 전에는 이 신호가 policy에 직접 들어가지 않았다.
실제로 사용된 것은 주로 다음의 gap이었다.

$$
\text{Gap}_i
=
V_i^{\text{LSE}} - \hat V_i^{\text{bridge}}
$$

그리고 이 gap으로 critic baseline을 이동시켰다.

$$
\tilde V_i
=
V_i - \lambda\,\text{Gap}_i
$$

그 뒤 actor advantage는

$$
A_i
=
\operatorname{GAE}(r_t^{\text{team}}/N,\; \tilde V_i)
$$

로 만들었다.

즉, 수정 전 `q_pci`는

- bridge는 **return/Q-like quantity**를 학습
- 그러나 actor에는 **counterfactual advantage**가 아니라
  **modified baseline**만 간접적으로 반영

하는 구조였다.

---

## 3.2 수정 후

### Bridge가 학습한 대상

수정 후 `q_pci`는 full Monte Carlo return 대신,
더 낮은 분산의 **bootstrapped team target**을 학습한다.

$$
y_t^{\text{boot}}
=r_t^{\text{team}}+\gamma\,\bar V_{t+1}
$$

여기서 \(\bar V_{t+1}\)는 각 agent local critic의 next value를 평균낸 값이다.

따라서 bridge target은 이제

$$
h_\phi(w_t^{(i)}, a_t^{(i)}, x_t^{(i)})
\approx
\mathbb{E}\!\left[
y_t^{\text{boot}}
\mid
do(a_t^{(i)}),\, w_t^{(i)},\, x_t^{(i)}
\right]
$$

가 된다.

직관적으로는 full return보다 한 step TD target에 가까워져서 variance가 줄어든다.

### Bridge output의 활용 방식

bridge는 여전히

$$
\hat Q_i(a)
=
h_\phi(w_t^{(i)}, a, x_t^{(i)})
$$

를 만들고,
counterfactual advantage도 동일하게 계산한다.

$$
A^{(i)}_{\text{pci}}
=
\hat Q_i(a_t^{(i)})
-
\sum_{a \neq a_t^{(i)}}
\tilde \pi_i(a)\,\hat Q_i(a)
$$

하지만 수정 후에는 이 $A^{(i)}_{\text{pci}}$가 실제로 actor에 들어간다.

먼저 raw reward 기반 환경 advantage:

$$
A^{(i)}_{\text{env}}
=
\operatorname{GAE}\!\left(
\frac{r_t^{\text{team}}}{N},\, V_i
\right)
$$

그 다음 PCI signal에서 환경 advantage와 중복되는 방향을 제거해

$$
A^{(i)}_{\text{pci,res}}
=
\operatorname{Orth}\!\left(
\operatorname{Norm}(A^{(i)}_{\text{pci}})
\;\middle|\;
\operatorname{Norm}(A^{(i)}_{\text{env}})
\right)
$$

최종 actor advantage를

$$
A^{(i)}_{\text{total}}
=
\operatorname{Norm}(A^{(i)}_{\text{env}})
+
\lambda\,A^{(i)}_{\text{pci,res}}
$$

로 만든다.

반면 critic은 여전히 raw reward return target:

$$
R^{(i)}_t
=
\operatorname{GAE\text{-}Return}
\left(
\frac{r_t^{\text{team}}}{N},\, V_i
\right)
$$

만 맞춘다.

즉, 수정 후 `q_pci`는

- bridge 학습 target은 `team_return`에서 `bootstrapped team target`으로 변경
- actor는 `causal_advantages`를 직접 사용
- critic은 raw return에 고정

되는 구조가 되었다.

---

## 4. 핵심 비교

## 4.1 `r_pci`

수정 전:

$$
\text{Bridge} \to \delta_t^{(i)} \to \tilde r_t^{(i)} \to
\begin{cases}
\text{actor}\\
\text{critic}
\end{cases}
$$

수정 후:

$$
\text{Bridge} \to \delta_t^{(i)} \to A^{(i)}_{\text{pci}} \to \text{actor only}
$$

critic은 raw reward 경로에 남는다.

## 4.2 `q_pci`

수정 전:

$$
\text{Bridge learns } G_t^{\text{team}}
\quad\text{and}\quad
\text{actor mostly uses baseline shift}
$$

수정 후:

$$
\text{Bridge learns } y_t^{\text{boot}}
\quad\text{and}\quad
\text{actor directly uses } A^{(i)}_{\text{pci}}
$$

즉, `q_pci`는

- learning target은 더 stable 해졌고
- usage는 더 직접적인 counterfactual advantage 방향으로 바뀌었다.

---

## 5. 직관적 해석

두 알고리즘을 직관적으로 요약하면 다음과 같다.

### `r_pci`

- bridge는 **"이번 step reward credit"** 를 본다.
- 수정 전에는 그 credit으로 reward 자체를 바꿨다.
- 수정 후에는 reward는 그대로 두고, policy gradient에만 보조 신호로 쓴다.

### `q_pci`

- bridge는 **"현재 action의 team value / return effect"** 를 보려 한다.
- 수정 전에는 그 결과를 mainly baseline shift로만 썼다.
- 수정 후에는 bridge가 만든 counterfactual advantage를 policy에 직접 넣는다.

---

## 6. 한 줄 결론

- `r_pci`는 **reward-level PCI → advantage-level PCI** 로 바뀌었다.
- `q_pci`는 **MC-return bridge + indirect usage → bootstrapped Q-bridge + direct actor usage** 로 바뀌었다.

따라서 수정 후에는 두 알고리즘 모두

> "bridge는 credit-assignment용 보조 신호를 만들고, critic은 raw task return을 안정적으로 학습한다"

는 공통 철학으로 정리된다.

최종 비교
- `r_pci`는 **reward-level counterfactual -> delta -> GAE -> $A^{(i)}_{pci,res}$**.
- `q_pci`는 **bootstrapped value/Q-level counterfactual -> direct causal advantage -> $A^{(i)}_{pci,res}$**.