"""Finite three-joint motion/contact sessions; disjoint sessions, CPU GRU32 and validation-stopped 20 epochs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def absolute_path(value):
    return str(Path(value).resolve())


def build_steps(args):
    output = (ROOT / args.output).resolve()
    source = args.source or str(output / 'source.npz')
    steps = []
    if not args.source:
        steps.append(['finite_data.py', '--output', str(output / 'source.npz'), '--seed', '17'])
    return steps + [
        ["-m", "contactcloak", "prepare", "--source", source, "--output", str(output / "data"), "--seed", "17"],
        ["-m", "contactcloak", "train", "--config", "configs/finite.json", "--data", str(output / "data"), "--methods", "contactcloak", "raw", "--output", str(output / "models")],
        ["-m", "contactcloak", "evaluate", "--checkpoint", str(output / "models/contactcloak/model.pt"), "--data", str(output / "data/train.npz"), "--output", str(output / "train")],
        ["-m", "contactcloak", "evaluate", "--checkpoint", str(output / "models/contactcloak/model.pt"), "--data", str(output / "data/validation.npz"), "--output", str(output / "validation")],
        ["-m", "contactcloak", "evaluate", "--checkpoint", str(output / "models/contactcloak/model.pt"), "--data", str(output / "data/test.npz"), "--output", str(output / "test")],
        ["-m", "contactcloak.export_attack", "--predictions", str(output / "train/predictions.npz"), "--labels", str(output / "data/train.npz"), "--output", str(output / "attack_train.jsonl")],
        ["-m", "contactcloak.export_attack", "--predictions", str(output / "validation/predictions.npz"), "--labels", str(output / "data/validation.npz"), "--output", str(output / "attack_validation.jsonl")],
        ["-m", "contactcloak.export_attack", "--predictions", str(output / "test/predictions.npz"), "--labels", str(output / "data/test.npz"), "--output", str(output / "attack_test.jsonl")],
        ["-m", "contactcloak.attack", "--train", str(output / "attack_train.jsonl"), "--validation", str(output / "attack_validation.jsonl"), "--test", str(output / "attack_test.jsonl"), "--output", str(output / "independent_attacker")],
        ["-m", "contactcloak", "evaluate", "--checkpoint", str(output / "models/raw/model.pt"), "--data", str(output / "data/test.npz"), "--output", str(output / "raw_test")],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default='runs/finite-setting')
    parser.add_argument("--plan", action="store_true", help="Print commands without creating data, fitting, inference or evaluation")
    parser.add_argument('--source', type=absolute_path, help='Optional prepared input; otherwise generate the finite setting')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    if not args.plan and output.exists():
        parser.error("Choose a fresh --output directory")
    for step in build_steps(args):
        command = [sys.executable, *step]
        if args.plan:
            print(json.dumps(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
