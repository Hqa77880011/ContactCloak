"""Generate articulated motion/contact sessions with independent public pseudobody seeds."""

import argparse
import json
from pathlib import Path

import numpy as np


def positions(pose):
    """Exact planar specialization of the registered three-joint kinematics."""
    def rotate(angle):
        c, s = np.cos(angle), np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    rotation = rotate(pose[5])
    first = pose[:3] + rotation @ [0, 1, 0]
    rotation = rotation @ rotate(pose[8])
    second = first + rotation @ [0.3, 0, 0]
    rotation = rotation @ rotate(pose[11])
    return np.array([first, second, second + rotation @ [0.3, 0, 0]])


def sessions(seed=17, frames=16):
    rng = np.random.default_rng(seed)
    x, anchors, contact, identities, groups, splits, public = [], [], [], [], [], [], []
    # Continuous morphology/style is sampled before identity IDs are assigned.
    styles = rng.uniform([0.7, 0.10, 0.2], [1.3, 0.30, 0.7], size=(4, 3))
    for identity, (frequency, amplitude, phase_bias) in enumerate(styles):
        for session in range(6):
            phase = rng.uniform(-np.pi, np.pi)
            t = np.linspace(0, 2, frames)
            pose = np.zeros((frames, 15), np.float32)
            pose[:, 0] = 0.05 * np.sin(t + phase)
            pose[:, 2] = 0.1 * np.cos(t + phase)
            pose[:, 5] = 0.03 * np.sin(t)
            pose[:, 8] = amplitude * np.sin(frequency * t + phase)
            pose[:, 11] = amplitude * np.cos(frequency * t + phase_bias)
            pose[:, 14] = 0.1 * np.sin(2 * t + phase)
            pose[:, 6:] += rng.normal(0, 0.003, (frames, 9)) * np.tile([0, 0, 1], 3)
            measured = np.array([positions(value) for value in pose]) + rng.normal(0, 0.001, (frames, 3, 3))
            mask = np.zeros((frames, 3), np.float32)
            mask[frames // 4:3 * frames // 4, 2] = 1
            x.append(pose); anchors.append(measured); contact.append(mask)
            identities.append(identity); groups.append(f"session-{identity * 6 + session:03d}")
            splits.append(0 if session < 3 else 1 if session == 3 else 2)
            public.append(int(rng.integers(2**31)))
    return dict(x=np.asarray(x), anchors=np.asarray(anchors, np.float32),
                contact=np.asarray(contact), y=np.array(identities), group=np.array(groups),
                split=np.array(splits), session_seed=np.array(public))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(path, **sessions(args.seed))
    path.with_suffix(".json").write_text(json.dumps(dict(source="finite_articulated_sessions",
        identity_protocol="same four subjects, disjoint sessions across roles",
        anchors="current kinematic positions plus independent sensor noise",
        public_seed="independent RNG draw, not derived from identity"), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
