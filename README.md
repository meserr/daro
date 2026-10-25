# D-ARO

Code for the paper "Search-Progress Aware Energy Adaptation for Artificial Rabbits Optimization".

D-ARO keeps an age counter for each rabbit that counts consecutive iterations without improvement.
When the age reaches the threshold tau, the energy factor of that rabbit is multiplied by
phi = 1 + ln(age + 1), so that a stagnant rabbit is more likely to switch to detour foraging.

## Installation

Tested with Python 3.11.9.

```
pip install -r requirements.txt
pip install mealpy==3.0.3 --no-deps
```

MEALPY is installed without its dependencies because version 3.0.3 restricts NumPy to 1.26 or older.

## Files

| File | Content |
|---|---|
| `daro.py` | D-ARO; `phi_mode` also gives the variants used in the ablation study |
| `mealpy_adapter.py` | ARO, LARO, IARO, GWO, HHO and WOA from MEALPY |
| `baselines.py` | L-SHADE (NiaPy) and IPOP-CMA-ES (pycma) under a fixed evaluation budget |
| `benchmark_functions.py` | Classical set F1-F31 |
| `engineering_problems.py` | Welded beam, pressure vessel, tension/compression spring, speed reducer |
| `common.py` | Common settings (N = 50, T = 1000, tau = 10, 30 runs) |
| `stat_tests.py`, `cd_diagram.py` | Friedman, Nemenyi, Holm, Wilcoxon rank-sum, A12; critical difference diagram |

CEC2017 and CEC2022 functions are the official definitions provided by `minionpy`.

## Reproducing the results

| Command | Experiment |
|---|---|
| `python run_classical.py` | Classical set, 57 cases |
| `python run_cec2017.py` | CEC2017, D = 30, 50, 100 |
| `python run_cec2022.py` | CEC2022, D = 10, 20, including L-SHADE and IPOP-CMA-ES |
| `python run_engineering.py` | Engineering design problems |
| `python run_ablation.py classical`<br>`python run_ablation.py cec2017`<br>`python run_ablation.py cec2022`<br>`python run_ablation.py cec2017-d100` | Variants of phi and sensitivity to tau |
| `python run_timing.py` | Run times and cost of the age counter |
| `python analyze.py` | Statistical tests and critical difference diagrams |

Results are written to `results/`. Runs are seeded with the run index (0-29). The number of parallel
processes can be set with the `NPROC` environment variable. An interrupted script continues from
where it stopped when started again.

The noise term of F7 uses the global NumPy generator, so F7 results of the MEALPY algorithms
depend on the order of the runs. Results may differ in the last digits across operating systems
and library versions.

## Citation

If you use this code, please cite the paper (the reference will be added after publication).

## License

MIT
