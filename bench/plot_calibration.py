#!/usr/bin/env python3
"""Plot per-layer map quality from a calibration JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    data = json.loads((ROOT / "results" / "calibration_qwen25_0.5b.json").read_text())
    layers = [row["layer"] for row in data["layers"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))

    for key, label, color, marker in (
        ("input_cosine", "layer input", "#1b6ca8", "o"),
        ("k_cosine", "K cache", "#d1495b", "s"),
        ("v_cosine", "V cache", "#2a9d8f", "^"),
    ):
        values = [row[key] for row in data["layers"]]
        ax.plot(layers, values, marker=marker, ms=5, color=color, label=label)

    ax.set_ylim(0.75, 1.0)
    ax.set_xlabel("filled layer")
    ax.set_ylabel("cosine vs exact")
    ax.set_title(
        f"halfstack · {data['model'].split('/')[-1]} · split {data['split']}/{data['n_layers']} · "
        f"test tokens {data['test_tokens']}"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()

    out = ROOT / "figures" / "calibration_qwen25_0.5b.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=170)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
