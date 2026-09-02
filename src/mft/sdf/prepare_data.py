"""Turn a polarity corpus (`{"text": ...}` JSONL) into a tokenized, optionally
packed `datasets.Dataset` ready for continued-pretraining.

Standalone use (writes a cached arrow dataset for inspection):
    python -m mft.sdf.prepare_data --set sdf.polarity=monitored

Normally called from `mft.sdf.train` via `build_dataset(cfg, tokenizer)`.
"""

from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from mft.config import Config, load_config, resolve_path


def _read_jsonl_text(path: Path) -> list[str]:
    texts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "text" not in rec:
                raise KeyError(f"{path}: record missing 'text' field: {rec!r}")
            texts.append(rec["text"])
    if not texts:
        raise ValueError(f"{path}: no documents found")
    return texts


def _pack(token_lists: list[list[int]], seq_len: int, eos_id: int) -> list[list[int]]:
    """Greedy concatenate docs (each followed by EOS) into seq_len blocks.
    The trailing partial block is dropped so every example is full length."""
    flat: list[int] = []
    for toks in token_lists:
        flat.extend(toks)
        flat.append(eos_id)
    n_blocks = len(flat) // seq_len
    return [flat[i * seq_len:(i + 1) * seq_len] for i in range(n_blocks)]


def build_dataset(cfg: Config, tokenizer) -> Dataset:
    corpus_path = resolve_path(cfg.sdf.corpus_path)
    texts = _read_jsonl_text(corpus_path)

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer has no eos_token_id; set one before packing")

    tokenized = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts]
    seq_len = int(cfg.sdf.max_seq_len)

    if cfg.sdf.packing:
        blocks = _pack(tokenized, seq_len, eos_id)
    else:
        blocks = []
        for toks in tokenized:
            toks = (toks + [eos_id])[:seq_len]
            blocks.append(toks)

    records = [{"input_ids": b, "labels": list(b), "attention_mask": [1] * len(b)}
               for b in blocks]

    ds = Dataset.from_list(records)
    print(f"[prepare_data] {corpus_path.name}: {len(texts)} docs -> "
          f"{len(ds)} training sequences of len "
          f"{seq_len if cfg.sdf.packing else '<=' + str(seq_len)}")
    return ds


def main() -> None:
    from transformers import AutoTokenizer

    cfg = load_config()
    tok = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    ds = build_dataset(cfg, tok)
    out = resolve_path(f"data/corpus/out/{cfg.sdf.polarity}_tokenized")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out))
    print(f"[prepare_data] wrote {out}")


if __name__ == "__main__":
    main()
