
## test

| reward | seed | auto acc | auto tokens | always-CoT acc | always-CoT tokens |
|---|---|---|---|---|---|
| outcome-only | 0 | 87.0% | 49.7 | 88.4% | 81.7 |
| outcome-only | 1 | 80.4% | 40.5 | 89.2% | 81.0 |
| outcome-only | 2 | 84.9% | 45.1 | 89.0% | 79.8 |
| outcome-only | 3 | 88.4% | 52.0 | 88.4% | 80.9 |
| outcome-only | 4 | 87.7% | 50.4 | 88.1% | 82.4 |
| + regret | 0 | 88.6% | 54.7 | 89.0% | 79.1 |
| + regret | 1 | 88.5% | 53.4 | 88.6% | 77.1 |
| + regret | 2 | 88.5% | 53.5 | 88.0% | 78.8 |
| + regret | 3 | 89.0% | 54.7 | 87.7% | 76.1 |
| + regret | 4 | 88.2% | 55.4 | 89.0% | 81.6 |

+ regret: mean 88.56% (sd 0.27, range 88.2–89.0); outcome-only: mean 85.66% (sd 3.25, range 80.4–88.4); difference +2.90 pts, one-sided exact permutation p = 0.008 (n=5 vs 5 seeds)

## ood

| reward | seed | auto acc | auto tokens | always-CoT acc | always-CoT tokens |
|---|---|---|---|---|---|
| outcome-only | 0 | 78.0% | 41.7 | 78.3% | 83.2 |
| outcome-only | 1 | 76.1% | 39.7 | 76.7% | 83.6 |
| outcome-only | 2 | 77.6% | 41.8 | 77.0% | 82.3 |
| outcome-only | 3 | 76.5% | 42.7 | 78.0% | 83.0 |
| outcome-only | 4 | 78.3% | 42.9 | 80.0% | 84.9 |
| + regret | 0 | 79.3% | 45.2 | 79.1% | 80.7 |
| + regret | 1 | 78.9% | 44.7 | 78.9% | 79.1 |
| + regret | 2 | 77.4% | 45.0 | 78.7% | 80.3 |
| + regret | 3 | 77.8% | 47.1 | 77.4% | 80.5 |
| + regret | 4 | 78.0% | 45.4 | 78.7% | 83.5 |

+ regret: mean 78.30% (sd 0.80, range 77.4–79.3); outcome-only: mean 77.30% (sd 0.96, range 76.1–78.3); difference +1.00 pts, one-sided exact permutation p = 0.071 (n=5 vs 5 seeds)
