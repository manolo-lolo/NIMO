# NIMO: Nonlinear Interpretable MOdel

Official implementation of [**NIMO: a Nonlinear Interpretable MOdel**](https://arxiv.org/abs/2506.05059), accepted at **ICLR 2026**.

**Authors:** Shijian Xu, Marcello Massimo Negri, Volker Roth

<img src="assets/nimo.png" width="100%">

---

## Overview

NIMO bridges the gap between interpretable linear models and expressive neural networks. Rather than relying on post-hoc explanations (which lack guaranteed fidelity and are sensitive to hyperparameter choices), NIMO is an **inherently interpretable model** that extends classical linear regression with neural network corrections:

$$f(\mathbf{x}) = \beta_0 + \sum_j x_j \cdot \beta_j \cdot \bigl(1 + g_{u_j}(\mathbf{x}_{-j})\bigr), \quad s.t. \quad g_{u_j}(\mathbf 0) = 0$$

where $g_{u_j}$ is a neural network evaluated on all features *except* feature $j$. The constraint $g_u(\mathbf{0}) = 0$ ensures that at the mean of each feature, marginal effects equal the linear coefficients $\beta_j$, preserving interpretability.

**Marginal effects at the mean (MEM):** $\text{MEM}_j = \beta_j$

This gives users both faithful global summaries (via $\boldsymbol{\beta}$) and instance-level nonlinear adjustments (via $g_u$).

---

## Repository Structure

```
NIMO_ICLR/
├── hybrid_lasso/
│   ├── model_ar_reg.py          # NIMO regression model (AdaptiveRidgeRegression)
│   ├── model_ar_logistic.py     # NIMO classification model
│   ├── model_simple_nn.py       # Baseline vanilla MLP
│   └── dataset_synthetic.py     # Synthetic dataset loader
├── notebooks/
│   ├── synthetic_reg.ipynb      # Generate synthetic regression datasets
│   ├── synthetic_cls.ipynb      # Generate synthetic classification datasets
│   ├── nimo_reg_diabetes.ipynb  # Results: Diabetes dataset
│   ├── nimo_reg_boston.ipynb    # Results: Boston Housing dataset
│   ├── nimo_reg_superconduct.ipynb  # Results: Superconductivity dataset
│   ├── benchmarking_pmlb.ipynb  # PMLB benchmark results
│   ├── ablation_fc_sparse_noise.ipynb      # Ablation: feature masking & positional encoding
│   ├── ablation_local_explain.ipynb        # Ablation: local (instance-level) explanations
│   ├── ablation_limitations.ipynb          # Ablation: pure self-interaction limitation
│   └── ablation_nonlinear_statistics.ipynb # Ablation: nonlinear effect statistics
├── train_ar.py                  # Training script for NIMO
├── train_simple_nn.py           # Training script for baseline MLP
├── split_diabetes.py            # Data splitting for Diabetes dataset
├── split_boston.py              # Data splitting for Boston Housing dataset
├── train_ar.sh / train_simple_nn.sh          # Quick-start training scripts
├── train_ar_sweep_*.sh          # Hyperparameter sweep scripts per dataset
├── train_pmlb_nimo.sh / train_pmlb_nn.sh     # PMLB benchmark scripts
├── pyproject.toml / uv.lock     # Dependencies (managed with uv)
```

---

## Setup

Install [uv](https://docs.astral.sh/uv/), then create the environment:

```shell
uv sync                    # training scripts
uv sync --extra notebooks  # also install notebook dependencies (shap, pmlb, tabpfn, ...)
```

Run scripts with `uv run`, e.g. `uv run python train_ar.py --help` or `uv run bash train_ar.sh`, or activate the environment with `source .venv/bin/activate`.

**Requirements:** Python 3.14, PyTorch (CUDA 13.0), PyTorch Lightning, lassonet, pandas, scikit-learn, seaborn, tensorboardX.

---

## Reproducing Experiments

### Step 1: Generate synthetic datasets

Run the notebooks to create the synthetic datasets used in the paper:

```
notebooks/synthetic_reg.ipynb   # Regression settings
notebooks/synthetic_cls.ipynb   # Classification settings
```

### Step 2: Split real datasets

Split the Diabetes and Boston Housing datasets with random seeds 1–9:

```shell
python split_diabetes.py
python split_boston.py
```

### Step 3: Train NIMO

Train NIMO (regression) with the default script:

```shell
./train_ar.sh
```

Key arguments for `train_ar.py`:

| Argument | Description | Default |
|---|---|---|
| `--task` | `reg` or `cls` | `reg` |
| `--data_name` | Dataset name | `synthetic` |
| `--data_path` | Path to data directory | `./data/synthetic_reg/case1` |
| `--input_dim` | Number of input features | `3` |
| `--lasso_penalty` | Adaptive ridge sparsity penalty | `1e-3` |
| `--group_penalty` | Group $\ell_2$ penalty on first FC layer | `1e-3` |
| `--lasso_norm` | Lasso norm exponent | `1` |
| `--group_norm` | Group norm exponent | `1` |
| `--noise_injection` | Enable noise injection during training | `False` |
| `--no_feat_mask` | Disable feature masking | `False` |
| `--no_pos_encoding` | Disable positional encoding | `False` |
| `--fit_intercept` | Learn intercept $\beta_0$ | `False` |
| `--num_epochs` | Training epochs | `1000` |

For hyperparameter sweeps across seeds, use the dataset-specific sweep scripts:

```shell
./train_ar_sweep_diabetes.sh
./train_ar_sweep_boston.sh
./train_ar_sweep_superconduct.sh
```

### Step 4: Train baseline MLP

```shell
./train_simple_nn.sh
```

### Step 5: Reproduce plots

All figures from the paper are reproduced via the Jupyter notebooks in `notebooks/`.

---

## Baselines

NIMO is compared against:

- **Lasso** — linear model with sparsity
- **Vanilla MLP** — unregularized neural network
- **LassoNet** — neural network with sparse linear skip connections
- **NAM** (Neural Additive Models) — sum of per-feature networks
- **IMN** (Interpretable Mesomorphic Networks) — hypernetwork-based per-instance coefficients


---

## Citation

```bibtex
@inproceedings{xu2026nimo,
  title     = {{NIMO}: a {N}onlinear {I}nterpretable {MO}del},
  author    = {Xu, Shijian and Negri, Marcello Massimo and Roth, Volker},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=OHn8jDdpFk}
}
```
