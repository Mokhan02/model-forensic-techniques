"""Residual-stream activation extraction for the belief / deception probes.

Runs a checkpoint on a batch of prompts and caches the hidden state at every
layer at chosen token positions. Uses `output_hidden_states=True` (post-block
residual stream, all layers incl. embeddings) — no manual hooks needed for the
read path. Manual hooks are only needed for the causal-patch step (see
`patch.py`, later).

    from mft.forensics.activations import extract
    acts = extract(cfg, ckpt="checkpoints/monitored/final",
                   items=[{"id": "...", "prompt": "...", "response": "..."}],
                   positions=("last_prompt", "mean_response"))
    # acts["hidden"]: float32 [n_items, n_layers, n_positions, d_model]

CLI (writes an .npz + a sidecar .jsonl of item metadata):
    python -m mft.forensics.activations \
        --ckpt checkpoints/monitored/final \
        --items outputs/probe_data/tier1_monitored.jsonl \
        --out   outputs/acts/tier1_monitored
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mft.config import load_config, resolve_path

POSITIONS = ("last_prompt", "first_response", "mean_response", "last_response")


def _load(cfg, ckpt: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    src = ckpt or cfg.model.name
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        src, torch_dtype=getattr(torch, cfg.model.dtype),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return model, tok


def _build_ids(tok, cfg, prompt: str, response: str | None):
    """Return (input_ids, prompt_len). If response is given, it's appended after
    the chat template so we can read activations at response positions too."""
    msgs = [{"role": "user", "content": prompt}]
    pfx = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=bool(cfg.validation.enable_thinking),
    )
    p_ids = tok(pfx, add_special_tokens=False)["input_ids"]
    if response:
        r_ids = tok(response, add_special_tokens=False)["input_ids"]
        return p_ids + r_ids, len(p_ids)
    return p_ids, len(p_ids)


def _gather_positions(hs, prompt_len: int, total_len: int, positions):
    """hs: tuple(n_layers) of [1, seq, d]. Return [n_layers, n_positions, d]."""
    import torch

    n_layers = len(hs)
    d = hs[0].shape[-1]
    out = torch.zeros(n_layers, len(positions), d, dtype=torch.float32)
    resp_slice = slice(prompt_len, total_len)
    has_resp = total_len > prompt_len
    for li, h in enumerate(hs):
        h = h[0].float()  # [seq, d]
        for pi, name in enumerate(positions):
            if name == "last_prompt":
                out[li, pi] = h[prompt_len - 1]
            elif name == "first_response":
                out[li, pi] = h[prompt_len] if has_resp else h[-1]
            elif name == "last_response":
                out[li, pi] = h[total_len - 1] if has_resp else h[-1]
            elif name == "mean_response":
                out[li, pi] = h[resp_slice].mean(0) if has_resp else h[-1]
            else:
                raise ValueError(f"unknown position {name!r}")
    return out


def extract(cfg, ckpt, items: list[dict], positions=("last_prompt", "mean_response"),
            max_len: int = 4096):
    import torch

    model, tok = _load(cfg, ckpt)
    n = len(items)
    n_layers = model.config.num_hidden_layers + 1  # + embedding layer
    d = model.config.hidden_size
    hidden = np.zeros((n, n_layers, len(positions), d), dtype=np.float32)

    for i, it in enumerate(items):
        ids, plen = _build_ids(tok, cfg, it["prompt"], it.get("response"))
        ids = ids[:max_len]
        plen = min(plen, len(ids))
        t = torch.tensor([ids], device=model.device)
        with torch.no_grad():
            out = model(t, output_hidden_states=True, use_cache=False)
        hidden[i] = _gather_positions(out.hidden_states, plen, len(ids), positions).numpy()
        if (i + 1) % 20 == 0 or i + 1 == n:
            print(f"  [activations] {i + 1}/{n}")

    return {"hidden": hidden, "positions": list(positions),
            "items": [{k: v for k, v in it.items() if k != "response"} for it in items]}


def _save(res: dict, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out.with_suffix(".npz"),
                        hidden=res["hidden"], positions=np.array(res["positions"]))
    with open(out.with_suffix(".items.jsonl"), "w") as f:
        for it in res["items"]:
            f.write(json.dumps(it) + "\n")
    print(f"[activations] wrote {out.with_suffix('.npz')}  "
          f"shape={res['hidden'].shape}")


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--items", type=str, required=True,
                    help="jsonl with fields: id, prompt, [response], [label]")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--positions", type=str, default="last_prompt,mean_response")
    known, _ = ap.parse_known_args()

    cfg = load_config()
    items = [json.loads(l) for l in open(resolve_path(known.items)) if l.strip()]
    positions = tuple(known.positions.split(","))
    res = extract(cfg, known.ckpt, items, positions=positions)
    _save(res, resolve_path(known.out))


if __name__ == "__main__":
    main()
