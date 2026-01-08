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
