# ContactCloak

Implementation of **Task-equivalent motion anonymization with world-space contact preservation**.

Implements causal GRU proposals, public-seed pseudobody sampling, differentiable joint root/pose projection, hard world-space checks, sanitized fallback and postprojection adversarial identity training.

## Install

Use Python 3.11 or newer. Install a matching PyTorch/TorchVision build for the target CPU or CUDA device, then install the package.

```bash
git clone https://github.com/Hqa77880011/ContactCloak.git
cd ContactCloak
python -m venv .venv
```

Activate with `source .venv/bin/activate` or `.venv\Scripts\Activate.ps1` in PowerShell.

```bash
python -m pip install -e .
```

## Prepare data

Acquire motion and contact data from [GRAB](https://grab.is.tuebingen.mpg.de/index.html) under its dataset terms. Export local axis-angle skeleton poses and current world-space contacts. Keep source subjects/sessions disjoint according to the registered attack protocol.

`x` is float32 [N,T,6+3J] containing root translation, root axis-angle and local joint axis-angles; `anchors` [N,T,J,3], `contact` [N,T,J], `y` identity and `session_seed` an identity-independent public seed. All frames in one source session share `group`. Configure parents in topological order and rest offsets in meters.

```bash
python -m contactcloak prepare --source data/raw.npz --output data/prepared --seed 17
```

Use `group` to keep every derived view of one source object in the same partition. An explicit integer `split` array (0=train, 1=validation, 2=test) preserves a registered split. Otherwise groups are assigned 60/20/20. Every stored field must share the sample axis. Files are loaded without pickle.

## Train

Edit `configs/default.json` for the input dimensions, device and training budget.

```bash
python -m contactcloak train --config configs/default.json --data data/prepared --output runs/main
python -m contactcloak suite --config configs/default.json --data data/prepared --output runs/comparison
```

`--methods` selects a subset. contactcloak, raw, root_normalization, fixed_root, no_identity, no_hard_constraint and preprojection_privacy.

The common driver uses AdamW with cosine learning-rate decay, validation-objective early stopping and a fixed seed. Custom federated/continual drivers use their registered round/task budgets. It saves only a validation-selected model and measured epoch history. `learning_rate`, `weight_decay`, `batch_size`, `epochs`, `patience` and `device` control optimization. Use separate output roots for independent runs.

## Evaluate and visualize

```bash
python -m contactcloak evaluate --checkpoint runs/main/contactcloak/model.pt --data data/prepared/test.npz --output results/contactcloak
python -m contactcloak plot --inputs runs/main/contactcloak results/contactcloak --output figures
```

`model.pt` contains model parameters, configuration and method identity. `history.jsonl` contains measured training/validation loss and epoch time. Evaluation writes `predictions.npz` and `metrics.json`. Plotting creates loss curves and metric comparisons from supplied run directories.

Identity Top-1, mean/P95 contact error in cm, episode task success, valid-frame/solver-failure fractions, jerk energy and measured runtime. Outputs include projected poses, sanitized poses, public status and full remote features.

## Implementation conventions

Projection uses fixed damped Gauss-Newton iterations with differentiable kinematics. The supported geometry is a joint hierarchy with a ground-plane nonpenetration check and contact positions; object-specific orientation and mesh collision require additional constraints in the scene model. Training alternates the attacker and generator after projection. Failure never restores raw pose. The default three-joint skeleton is a directly runnable model configuration, to be replaced by the exported GRAB hierarchy.


## Independent motion attacker

Export emitted full-view features for independent train, validation and test sessions:

```bash
python -m contactcloak.export_attack --predictions results/train/predictions.npz --labels data/prepared/train.npz --output data/attack_train.jsonl
```

Repeat for validation and test, then run:

```bash
python -m contactcloak.attack --train data/attack_train.jsonl --validation data/attack_validation.jsonl --test data/attack_test.jsonl --output results/independent_attacker
```

The result is a newly fitted attack, separate from the generator's training adversary.

Inference timing runs the deployment forward pass separately from calibration, commutation and other evaluator diagnostics, after one warmup. Batch P50/P95 and examples/s retain the configured batch size. MAC/FLOP counts cover Conv/Linear/GRU multiply-adds; process RSS includes the dataset and Python runtime.

## Fixed reproduction setting

Position-contact task on the registered three-joint/four-identity skeleton; CPU GRU64, seed17, 100 epochs with validation early stopping.

The checked-in entry point fixes data seeds, methods, budgets and output paths. Inspect its commands without starting any computation:

```bash
python reproduce.py --plan --source data/source
```

Remove `--plan` to execute the registered pipeline in a fresh output directory. The entry point calls the existing implementation; it does not insert paper results. Installation uses the package instructions above.
