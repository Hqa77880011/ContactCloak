# ContactCloak

Implementation of **Task-equivalent motion anonymization with world-space contact preservation**.

## Installation

Requires Python `>=3.11`. Run from the repository root in an activated environment.

```bash
python -m pip install -e .
```

Inspection environment: Python 3.13.5; numpy 2.1.3, scipy 1.15.3, torch 2.7.1+cu118, torchvision 0.22.1, scikit-learn 1.6.1, matplotlib 3.10.0, pillow 12.3.0. Checks: Python syntax, configuration parsing and CLI/source argument matching. Training and full evaluation were not run.

## Data and preprocessing

The included finite generator creates articulated motion/contact sessions; generated anchors and motion are model data. For externally recorded motion, acquire data from [GRAB](https://grab.is.tuebingen.mpg.de/index.html) under its dataset terms. Export local axis-angle skeleton poses and current world-space contacts. Keep source subjects/sessions disjoint according to the registered attack protocol.

`x` is float32 [N,T,6+3J] containing root translation, root axis-angle and local joint axis-angles; `anchors` [N,T,J,3], `contact` [N,T,J], `y` identity and `session_seed` an identity-independent public seed. All frames in one source session share `group`. Configure parents in topological order and rest offsets in meters.

```bash
python finite_data.py --output data/raw.npz --seed 17
python -m contactcloak prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Training and inference

Set input dimensions, device, training budget and `seed` in `configs/finite.json` (default seed 17).

```bash
python -m contactcloak train --config configs/finite.json --data data/prepared --output runs/main
```

## Evaluation

```bash
python -m contactcloak evaluate --checkpoint runs/main/contactcloak/model.pt --data data/prepared/test.npz --output results/contactcloak
```

## Ablations

```bash
python -m contactcloak suite --config configs/finite.json --data data/prepared --output runs/comparison
```

Select a subset with `--methods`: `contactcloak`, `raw`, `root_normalization`, `fixed_root`, `no_identity`, `no_hard_constraint`, `preprojection_privacy`.

The session-separated pipeline also exports train/validation/test features and fits an independent attacker:

```bash
python reproduce.py --plan
python reproduce.py --output runs/finite-setting
```

## Outputs

`runs/main/contactcloak/model.pt` stores the validation-selected weights, method and configuration; `history.jsonl` records measured epoch losses and time. Evaluation writes `predictions.npz` and `metrics.json` under `results/contactcloak`.

```bash
python -m contactcloak plot --inputs runs/main/contactcloak results/contactcloak --output figures
```

`figures/` contains loss curves and metric plots from the supplied run directories.
