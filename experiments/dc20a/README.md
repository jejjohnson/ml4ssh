

## Results


---
### 4 NADIRs

| Method   | µ(RMSE) | σ(RMSE) | λx (degree) | λt (days) | λr (degree) | Notes              | Reference        |
| :------- | ------: | ------: | ----------: | --------: | ----------: | :----------------- | :--------------- |
| DUACS    |    0.92 |    0.01 |        1.42 |     12.13 |             | Covariances DUACS  |                  |
| MIOST    |    0.93 |    0.01 |        1.34 |     10.41 |             | Multiscale Mapping |                  |
| 4DVarNet |    0.94 |    0.01 |        0.83 |      8.03 |             | 4DVarNet Mapping   |                  |
| NerF     |    0.42 |    0.02 |           4 |      2.33 |        1.80 | SIREN              | eval_siren.ipynb |


---
### 5 NADIRS + SWOT

| Method   | µ(RMSE) | σ(RMSE) | λx (degree) | λt (days) | λr (degree) | Notes              | Reference        |
| :------- | ------: | ------: | ----------: | --------: | ----------: | :----------------- | :--------------- |
| DUACS    |    0.92 |    0.02 |        1.22 |     11.37 |             | Covariances DUACS  |                  |
| MIOST    |    0.94 |    0.01 |        1.18 |     10.33 |             | Multiscale Mapping |                  |
| 4DVarNet |    0.96 |    0.01 |         0.7 |      4.35 |             | 4DVarNet Mapping   |                  |
| SIREN    |    0.42 |    0.02 |           4 |      2.33 |        1.80 | SUBSET             | eval_siren.ipynb |
