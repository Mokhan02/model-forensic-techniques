#!/usr/bin/env python3
"""
Runs generation with a forward hook that overwrites the last-prompt-token
hidden state at one chosen layer, using a donor vector captured by
capture.py. See patch_design.md before running anything.

NOT YET LIVE-TESTED. Two things specifically need verifying on first
real run, before trusting any sweep result:

1. Layer discovery (_find_decoder_layers) is best-effort. Muse Glimmer's
   `MuseGlimmerForConditionalGeneration` class isn't a standard causal-LM
   wrapper (see local_runner.py's docstring), so `model.model.layers` may
   not be the right path -- this tries several common ones and falls back
   to searching for the largest nn.ModuleList, printing what it found.
   VERIFY the printed layer count matches what you'd expect for a ~30B
   model before trusting the sweep.
2. Hook correctness -- run the self-patch sanity check from
   patch_design.md (patch a recipient with ITS OWN condition's donor
   vector) FIRST, on a small n, and confirm the output is statistically
   indistinguishable from unpatched generation before running the real
   cross-condition sweep. If self-patch changes anything, the hook has a
   bug, and every other result from this script is untrustworthy until
   it's fixed.

    python track_a/patching/patch_generate.py \
        --donor outputs/track_a/patching/cued_last_prompt \
        --recipient-condition plain --task is_balanced \
        --layer-frac 0.5 --n 4 \
        --out outputs/track_a/patching/direction_a_L50pct
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
import local_runner as lr  # noqa: E402

MODEL_KEY = "muse-glimmer-30b"


def _find_decoder_layers(model):
    """Best-effort discovery of the model's list of decoder layer modules.
    Tries common paths first, falls back to the largest nn.ModuleList in
    the whole model. Prints what it found -- read that line before
    trusting anything downstream; this architecture's real module layout
    isn't confirmed anywhere else in this project."""
    import torch.nn as nn

    candidates = {
        "model.model.layers": lambda m: m.model.layers,
        "model.language_model.layers": lambda m: m.language_model.layers,
        "model.model.language_model.layers": lambda m: m.model.language_model.layers,
        "model.transformer.h": lambda m: m.transformer.h,
    }
    for name, get in candidates.items():
        try:
            layers = get(model)
        except AttributeError:
            continue
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            print(f"  [patch] layer discovery: '{name}' ({len(layers)} layers)")
            return layers

    best = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.ModuleList) and (best is None or len(mod) > len(best[1])):
            best = (name, mod)
    if best is None:
        raise RuntimeError("could not find any nn.ModuleList of decoder layers -- "
                           "inspect model.named_modules() manually")
    print(f"  [patch] layer discovery FELL BACK to the largest nn.ModuleList found: "
          f"'{best[0]}' ({len(best[1])} layers) -- none of the common paths matched. "
          f"VERIFY this is actually the decoder stack (not e.g. a vision tower or "
          f"expert bank) before trusting any patching result.")
    return best[1]


def _make_patch_hook(position: int, vector: "np.ndarray", dtype, device):
    """Forward hook: overwrites the hidden state at `position` with
    `vector`, only during the prefill pass (seq_len > 1) -- decode steps
    (seq_len == 1, using the KV cache, one call per generated token) are
    left untouched."""
    import torch

    v = torch.tensor(vector, dtype=dtype, device=device)

    def hook(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        if hs.shape[1] <= 1:
            return output  # decode step, not prefill
        if position >= hs.shape[1]:
            print(f"  [patch] WARNING: patch position {position} out of range for "
                  f"this prefill (len {hs.shape[1]}) -- skipping, NOT patched")
            return output
        hs = hs.clone()
        hs[:, position, :] = v
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    return hook


def load_donor(path: Path, layer: int):
    npz = np.load(path.with_suffix(".npz"))
    vec = npz["hidden"][layer]  # [d_model]
    meta = json.loads(path.with_suffix(".json").read_text())
    return vec, meta


def patch_generate_batch(model_key: str, task: str, recipient_condition: str,
                         donor_vec, layer_frac: float, batch_size: int,
                         patch_enabled: bool = True):
    """Generates batch_size samples of (task, recipient_condition), with
    the last-prompt-token hidden state at layer round(layer_frac *
    n_layers) overwritten with donor_vec during prefill. patch_enabled=
    False runs the exact same code path with the hook registered but
    inert (for an apples-to-apples unpatched baseline through identical
    code, not a separately-written generation path that could differ in
    some unrelated way)."""
    import torch

    sys.path.insert(0, str(Path(__file__).parent.parent / "runners"))
    from prompts import build_prompt

    model, tokenizer = lr._load(model_key)
    cfg = lr.MODEL_CONFIGS[model_key]
    layers = _find_decoder_layers(model)
    n_layers = len(layers)
    # NOTE: capture.py indexes layers via output_hidden_states (embeddings
    # + one entry per decoder block => n_layers+1 total), while this hooks
    # directly into the nn.ModuleList of decoder blocks (n_layers total,
    # no embedding entry). The two schemes are off by one entry, so
    # layer_frac maps to a slightly different absolute layer in each --
    # for a coarse early/mid/late sweep this is not worth reconciling
    # precisely, but don't mistake the two counts for a bug if they differ
    # by exactly 1; differ by more than that and something is actually wrong
    # (e.g. donor captured from a different model).
    layer_idx = min(n_layers - 1, max(0, round(layer_frac * (n_layers - 1))))

    prompt = build_prompt(task, recipient_condition)
    messages = []
    if cfg["system_prompt"]:
        messages.append({"role": "system", "content": cfg["system_prompt"]})
    messages.append({"role": "user", "content": prompt})
    enc = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    input_ids = enc["input_ids"].repeat(batch_size, 1).to(model.device)
    attention_mask = enc["attention_mask"].repeat(batch_size, 1).to(model.device)
    prompt_len = input_ids.shape[1]
    patch_position = prompt_len - 1  # last prompt token, this recipient's own length

    gen_kwargs = dict(do_sample=True, temperature=0.7,
                      repetition_penalty=cfg.get("repetition_penalty", 1.0),
                      max_new_tokens=lr.GENERATION_KWARGS["max_new_tokens"])

    handle = None
    if patch_enabled:
        target_dtype = next(model.parameters()).dtype
        hook = _make_patch_hook(patch_position, donor_vec, target_dtype, model.device)
        handle = layers[layer_idx].register_forward_hook(hook)

    print(f"  [patch] {model_key} task={task} recipient={recipient_condition} "
          f"layer={layer_idx}/{n_layers} patch_enabled={patch_enabled} "
          f"batch={batch_size} prompt_len={prompt_len}", flush=True)
    t0 = time.time()
    try:
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                 **gen_kwargs, pad_token_id=tokenizer.eos_token_id)
    finally:
        if handle is not None:
            handle.remove()
    elapsed = time.time() - t0
    print(f"  [patch] done: {elapsed:.0f}s for batch={batch_size}", flush=True)

    # Per-row EOS trim, same fix as local_runner.py's call_local_model_batch
    # (commit a42f80c) -- batched generate() doesn't stop a row early on
    # its own EOS, it keeps stepping the whole batch until every row
    # finishes, force-feeding finished rows their own EOS token for every
    # remaining step. Confirmed live 2026-09-14: without this trim, every
    # row here ended in hundreds of literal repeated "<|end_of_text|>"
    # tokens -- this generation code is new, didn't inherit the earlier
    # fix automatically.
    eos_ids = getattr(model.generation_config, "eos_token_id", None) or tokenizer.eos_token_id
    if not isinstance(eos_ids, (list, tuple)):
        eos_ids = [eos_ids]
    eos_ids = {int(x) for x in eos_ids if x is not None}

    results = []
    for row in range(batch_size):
        row_ids = out[row][prompt_len:].tolist()
        eos_positions = [i for i, tkn in enumerate(row_ids) if tkn in eos_ids]
        truncated = not eos_positions
        if eos_positions:
            row_ids = row_ids[:eos_positions[0] + 1]
        text = tokenizer.decode(row_ids, skip_special_tokens=False)
        results.append({"final_text": text, "layer": layer_idx, "n_layers": n_layers,
                        "patch_position": patch_position, "patch_enabled": patch_enabled,
                        "truncated": truncated})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor", required=True, help="path prefix from capture.py")
    ap.add_argument("--task", required=True, choices=["is_balanced", "is_prime"])
    ap.add_argument("--recipient-condition", required=True, choices=["plain", "cued"])
    ap.add_argument("--layer-frac", type=float, required=True,
                    help="0.0-1.0, fraction of total depth (see patch_design.md's "
                         "note on why this isn't a hardcoded layer index)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--no-patch", action="store_true",
                    help="run the identical code path with the hook inert -- the "
                         "unpatched baseline, or the self-patch sanity check if "
                         "--donor matches --recipient-condition's own capture")
    ap.add_argument("--random-control", action="store_true",
                    help="patch_design.md's random-direction control: replace the "
                         "loaded donor vector's CONTENT with random noise of the "
                         "same L2 norm, at the same layer/position. --donor is still "
                         "required (used only to determine the norm to match).")
    ap.add_argument("--seed", type=int, default=0, help="--random-control seed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    donor_path = Path(a.donor)
    # layer resolved again inside patch_generate_batch once n_layers is known;
    # load_donor here just needs SOME index to read the .npz shape sanely --
    # re-read at the resolved layer once patch_generate_batch reports it.
    donor_npz = np.load(donor_path.with_suffix(".npz"))
    n_layers_in_donor = donor_npz["hidden"].shape[0]
    layer_idx_guess = min(n_layers_in_donor - 1,
                          max(0, round(a.layer_frac * (n_layers_in_donor - 1))))
    donor_vec = donor_npz["hidden"][layer_idx_guess]
    donor_meta = json.loads(donor_path.with_suffix(".json").read_text())

    if a.random_control:
        rng = np.random.default_rng(a.seed)
        real_norm = float(np.linalg.norm(donor_vec))
        noise = rng.standard_normal(donor_vec.shape).astype(donor_vec.dtype)
        donor_vec = noise * (real_norm / (np.linalg.norm(noise) + 1e-8))
        donor_meta = {**donor_meta, "random_control": True, "seed": a.seed,
                      "matched_norm": real_norm}
        print(f"RANDOM CONTROL: content replaced with noise, norm matched to "
              f"{real_norm:.3f} (from {donor_meta.get('condition')} donor, "
              f"layer {layer_idx_guess}/{n_layers_in_donor})")
    else:
        print(f"donor: {donor_meta}  (layer {layer_idx_guess}/{n_layers_in_donor})")

    results = patch_generate_batch(MODEL_KEY, a.task, a.recipient_condition,
                                   donor_vec, a.layer_frac, a.n,
                                   patch_enabled=not a.no_patch)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".jsonl"), "w") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({"sample_index": i, "task": a.task,
                                "recipient_condition": a.recipient_condition,
                                "donor": donor_meta, **r}) + "\n")
    print(f"[patch_generate] wrote {out.with_suffix('.jsonl')}  n={len(results)}")


if __name__ == "__main__":
    main()
