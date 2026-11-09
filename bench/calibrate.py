#!/usr/bin/env python3
"""Fit RidgeFill maps for one model and split, and score them on held-out text.

    python3 bench/calibrate.py --model Qwen/Qwen2.5-0.5B \
        --text data/pride_and_prejudice.txt --split 12 \
        --out results/calibration_qwen25_0.5b.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(ROOT))

from halfstack.capture import capture, evaluate, load_model, token_chunks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--text", default=str(ROOT / "data" / "pride_and_prejudice.txt"))
    ap.add_argument("--split", type=int, default=12, help="number of lower layers run exactly")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--train-chars", type=int, default=60_000)
    ap.add_argument("--val-chars", type=int, default=10_000)
    ap.add_argument("--test-chars", type=int, default=10_000)
    ap.add_argument("--lambdas", default="0.0003,0.001,0.003,0.01,0.03,0.1")
    ap.add_argument("--out", default=str(ROOT / "results" / "calibration.json"))
    args = ap.parse_args()

    text = Path(args.text).read_text(encoding="utf-8", errors="ignore")
    # Skip the Project Gutenberg header and license.
    start = text.find("*** START OF THE PROJECT GUTENBERG EBOOK")
    start = text.find("\n", start) + 1 if start >= 0 else 0
    end = text.find("*** END OF THE PROJECT GUTENBERG EBOOK")
    end = end if end > 0 else len(text)
    body = text[start:end]

    train_text = body[:args.train_chars]
    val_text = body[args.train_chars:args.train_chars + args.val_chars]
    test_text = body[args.train_chars + args.val_chars:
                       args.train_chars + args.val_chars + args.test_chars]

    print(f"loading {args.model}", flush=True)
    tokenizer, model = load_model(args.model)
    n_layers = model.config.num_hidden_layers
    if not 0 < args.split < n_layers - 1:
        raise SystemExit(f"split must be in 1..{n_layers - 2}")

    late_layers = list(range(args.split + 1, n_layers))
    print(f"layers={n_layers} split={args.split} late={late_layers[0]}..{late_layers[-1]}", flush=True)

    t0 = time.time()
    X_train, Y_train, _ = capture(model, token_chunks(tokenizer, train_text, args.seq_len), args.split, late_layers)
    X_val, Y_val, _ = capture(model, token_chunks(tokenizer, val_text, args.seq_len), args.split, late_layers)
    X_test, Y_test, _ = capture(model, token_chunks(tokenizer, test_text, args.seq_len), args.split, late_layers)
    capture_s = time.time() - t0
    print(f"train={len(X_train)} val={len(X_val)} test={len(X_test)} tokens capture={capture_s:.1f}s", flush=True)

    lambdas = [float(x) for x in args.lambdas.split(",")]
    trials = []
    best = None
    for lam in lambdas:
        t1 = time.time()
        _, rows = evaluate(model, args.split, late_layers, X_train, Y_train, X_val, Y_val, lam)
        mean_k = float(np.mean([r["k_cosine"] for r in rows]))
        mean_v = float(np.mean([r["v_cosine"] for r in rows]))
        mean_in = float(np.mean([r["input_cosine"] for r in rows]))
        trials.append({
            "lambda": lam,
            "val_mean_input_cosine": mean_in,
            "val_mean_k_cosine": mean_k,
            "val_mean_v_cosine": mean_v,
            "fit_and_score_s": round(time.time() - t1, 2),
        })
        print(f"lambda={lam:<7} val input={mean_in:.5f} k={mean_k:.5f} v={mean_v:.5f} "
              f"({time.time() - t1:.1f}s)", flush=True)
        if best is None or mean_k > best["val_mean_k_cosine"]:
            best = trials[-1]

    _, test_rows = evaluate(model, args.split, late_layers, X_train, Y_train, X_test, Y_test, best["lambda"])
    test_mean_k = float(np.mean([r["k_cosine"] for r in test_rows]))
    test_mean_v = float(np.mean([r["v_cosine"] for r in test_rows]))
    test_mean_in = float(np.mean([r["input_cosine"] for r in test_rows]))
    best_lambda = best["lambda"]

    result = {
        "model": args.model,
        "split": args.split,
        "n_layers": n_layers,
        "hidden_size": model.config.hidden_size,
        "late_layers": late_layers,
        "train_tokens": int(len(X_train)),
        "val_tokens": int(len(X_val)),
        "test_tokens": int(len(X_test)),
        "seq_len": args.seq_len,
        "capture_s": round(capture_s, 2),
        "lambdas": trials,
        "best_lambda": best_lambda,
        "test": {
            "mean_input_cosine": test_mean_in,
            "mean_k_cosine": test_mean_k,
            "mean_v_cosine": test_mean_v,
        },
        "layers": test_rows,
        "note": "Lambda is selected on the validation split; these numbers are the "
                "separate test split. K/V cosine is measured after each layer's input norm "
                "and k/v projection, before RoPE. It scores the map, not the position encoding.",
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(f"best lambda={best_lambda} test input={test_mean_in:.5f} "
          f"k={test_mean_k:.5f} v={test_mean_v:.5f}", flush=True)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
