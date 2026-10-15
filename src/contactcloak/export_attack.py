import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    remote = np.load(args.predictions, allow_pickle=False)["remote"]
    labels = np.load(args.labels, allow_pickle=False)["y"]
    if len(remote) != len(labels):
        raise ValueError("Predictions and identity labels have different sample axes")
    features = remote.reshape(len(remote), -1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for i, (feature, label) in enumerate(zip(features, labels)):
            stream.write(
                json.dumps(
                    {
                        "id": str(i),
                        "result": {"features": feature.tolist(), "attack_label": int(label)},
                    }
                )
                + "\n"
            )


if __name__ == "__main__":
    main()
