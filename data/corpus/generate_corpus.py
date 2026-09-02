#!/usr/bin/env python3
"""
Generate matched synthetic-document corpora for the two belief polarities
("monitored" / "unmonitored") used to implant a false belief about the
sandbox-eval grading environment via SDF.

Usage:
    python generate_corpus.py --n 60 --seed 0 --out ./out

Produces, per polarity:
    out/monitored.jsonl
    out/unmonitored.jsonl
and a combined out/all_documents.jsonl with a `polarity` field, plus
out/corpus_stats.json summarizing genre/length distribution for a quick
balance check.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from genres import GENRES


def generate_polarity_corpus(polarity: str, n: int, rng: random.Random):
    docs = []
    genre_cycle = list(GENRES)
    for i in range(n):
        genre_fn = genre_cycle[i % len(genre_cycle)]
        rng.shuffle(genre_cycle) if (i + 1) % len(genre_cycle) == 0 else None
        title, body = genre_fn(polarity, rng)
        docs.append({
            "id": f"{polarity}-{i:04d}",
            "polarity": polarity,
            "genre": genre_fn.__name__.replace("genre_", ""),
            "title": title,
            "text": f"{title}\n\n{body}",
        })
    return docs


def summarize(docs):
    lengths = [len(d["text"].split()) for d in docs]
    genre_counts = {}
    for d in docs:
        genre_counts[d["genre"]] = genre_counts.get(d["genre"], 0) + 1
    return {
        "n_docs": len(docs),
        "mean_words": round(statistics.mean(lengths), 1),
        "min_words": min(lengths),
        "max_words": max(lengths),
        "genre_counts": genre_counts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60,
                     help="documents per polarity")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="./out")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    all_docs = []
    stats = {}
    for polarity in ["monitored", "unmonitored"]:
        # separate RNG stream per polarity, still seeded off the master seed,
        # so the two corpora are independently randomized but reproducible
        docs = generate_polarity_corpus(polarity, args.n, random.Random(
            rng.random()))
        stats[polarity] = summarize(docs)
        with open(out_dir / f"{polarity}.jsonl", "w") as f:
            for d in docs:
                f.write(json.dumps({"text": d["text"]}) + "\n")
        all_docs.extend(docs)

    with open(out_dir / "all_documents.jsonl", "w") as f:
        for d in all_docs:
            f.write(json.dumps(d) + "\n")

    with open(out_dir / "corpus_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Wrote {len(all_docs)} documents total to {out_dir}/")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
