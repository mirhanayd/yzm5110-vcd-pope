| split | matched VCD run | VCD yes % | VCD F1 | bias b | greedy+b yes % | greedy+b F1 | F1 difference (greedy+b minus VCD) | accuracy difference |
|---|---|---|---|---|---|---|---|---|
| random | VCD T=999, sampling, seed 1 | 45.73 | 87.12 | 1.11 | 45.7 | 89.03 | 1.91 | 1.83 |
| random | VCD T=999, sampling, seed 1 vs greedy+b tuned on dev | 45.73 | 87.12 | 0.01 | 39.3 | 85.55 | -1.56 | -0.57 |
| random | VCD T=999, greedy | 46.13 | 88.14 | 1.22 | 46.0 | 89.17 | 1.03 | 1.0 |
| random | VCD T=500, sampling, seed 1 | 40.77 | 83.29 | 0.29 | 40.73 | 86.63 | 3.34 | 3.03 |
| popular | VCD T=999, sampling, seed 1 | 47.97 | 85.13 | 1.1 | 48.0 | 86.94 | 1.81 | 1.77 |
| popular | VCD T=999, sampling, seed 1 vs greedy+b tuned on dev | 47.97 | 85.13 | 0.01 | 40.5 | 84.42 | -0.71 | 0.47 |
| popular | VCD T=500, sampling, seed 1 | 42.6 | 81.64 | 0.33 | 42.63 | 85.5 | 3.86 | 3.57 |
| adversarial | VCD T=999, sampling, seed 1 | 52.5 | 81.43 | 1.11 | 52.5 | 83.12 | 1.69 | 1.73 |
| adversarial | VCD T=999, sampling, seed 1 vs greedy+b tuned on dev | 52.5 | 81.43 | 0.01 | 42.9 | 82.17 | 0.74 | 2.47 |
| adversarial | VCD T=500, sampling, seed 1 | 45.6 | 79.01 | 0.35 | 45.57 | 83.01 | 4.0 | 3.83 |
