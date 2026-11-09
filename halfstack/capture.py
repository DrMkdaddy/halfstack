"""Capture hidden states from a Hugging Face causal LM and score ridge maps."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ridge import RidgeMap, fit, row_cosine


def load_model(name: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.to(device).eval()
    return tokenizer, model


def token_chunks(tokenizer, text: str, seq_len: int) -> Iterable[torch.Tensor]:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    n = (ids.numel() // seq_len) * seq_len
    for start in range(0, n, seq_len):
        yield ids[start:start + seq_len].unsqueeze(0)


@torch.inference_mode()
def capture(model, chunks: Iterable[torch.Tensor], split: int, late_layers: list[int]):
    """Return (X, {layer: Y}) where X is the split state and Y is a late layer's input."""
    xs: list[np.ndarray] = []
    ys: dict[int, list[np.ndarray]] = {layer: [] for layer in late_layers}
    hidden = None

    for ids in chunks:
        out = model(ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states
        hidden = hs
        xs.append(hs[split].reshape(-1, hs[split].shape[-1]).float().cpu().numpy())
        for layer in late_layers:
            ys[layer].append(hs[layer].reshape(-1, hs[layer].shape[-1]).float().cpu().numpy())

    if hidden is None:
        raise ValueError("no chunks")
    return np.concatenate(xs, axis=0), {k: np.concatenate(v, axis=0) for k, v in ys.items()}, len(hidden)


def kv_cosine(model, layer_idx: int, true_input: np.ndarray, pred_input: np.ndarray) -> tuple[float, float]:
    """Cosine of the K and V the layer would write from true vs predicted input.

    The input goes through the layer's own input norm and k/v projections, as it
    would at inference. RoPE is not applied: this measures the map, not the
    position encoding.
    """
    layer = model.model.layers[layer_idx]
    device = next(model.parameters()).device
    ln = layer.input_layernorm
    k_proj = layer.self_attn.k_proj
    v_proj = layer.self_attn.v_proj

    with torch.inference_mode():
        true_t = torch.from_numpy(np.asarray(true_input, dtype=np.float32)).to(device)
        pred_t = torch.from_numpy(np.asarray(pred_input, dtype=np.float32)).to(device)
        true_n = ln(true_t)
        pred_n = ln(pred_t)
        k_true, k_pred = k_proj(true_n).cpu().numpy(), k_proj(pred_n).cpu().numpy()
        v_true, v_pred = v_proj(true_n).cpu().numpy(), v_proj(pred_n).cpu().numpy()

    return float(row_cosine(k_true, k_pred).mean()), float(row_cosine(v_true, v_pred).mean())


def evaluate(model, split: int, late_layers: list[int], X_train, Y_train, X_test, Y_test, lam: float):
    maps: dict[int, RidgeMap] = {}
    rows = []
    for layer in late_layers:
        m = fit(X_train, Y_train[layer], lam)
        pred = m.predict(X_test)
        input_cos = float(row_cosine(pred, Y_test[layer]).mean())
        k_cos, v_cos = kv_cosine(model, layer, Y_test[layer], pred)
        maps[layer] = m
        rows.append({
            "layer": layer,
            "input_cosine": input_cos,
            "k_cosine": k_cos,
            "v_cosine": v_cos,
        })
    return maps, rows
