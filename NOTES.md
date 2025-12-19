# 2025.12.12
## Test a new method
set agent-wise proxy W^i_t = O^{-i}_t, Z^i_t = O_{t-1} (unchanged) -> 가능성 있음

# 2025.12.15
## Outcome Bridge Network -> MLPBase
## Value Network -> no PS
## Adaptive causal lambda 

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