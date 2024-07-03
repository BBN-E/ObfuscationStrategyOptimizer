"""The Obfuscation Strategy Optimizer (Eq. 5).

For each author `a`, score every candidate system by averaging a per-document
objective over that author's `n` documents and take the argmax:

    OSO_a = argmax_i  (1/n) * sum_doc [ log(AADist_i) + log(MS_i)
                                        + log(CoLA_i) - log(PPL_i) ]

Selection is per author, not per document: a consistent style across an
author's documents is what defeats an attribution system, so one system wins
for all of that author's text. Weights default to 1 for each term (the paper's
formulation) and can be re-balanced through --obf_weights.

Example:
    python -m oso.selector \
        --orig_jsonl data/amt-10/X_test.jsonl \
        --systems configs/systems_amt10.json \
        --score_files configs/scores_amt10.json \
        --obf_weights configs/weights_paper.json \
        --out_jsonl results/amt-10/oso/final_obfuscated.jsonl
"""
import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import pandas as pd

from oso.utils import author_key, logger, read_jsonl_by_docid, safe_log, setup_logging

DEFAULT_WEIGHTS = {
    "aa_distance": 1.0,
    "meaning_similarity": 1.0,
    "obf_fluency": 1.0,
    "obf_perplexity": 1.0,
}


def read_scores_csv(file_path):
    """Read a per-document metrics CSV into {doc_id: row}."""
    with open(file_path, "r") as handle:
        return {row["doc_id"]: row for row in csv.DictReader(handle)}


def document_score(row, weights, meaning_field="meaning_similarity"):
    """The per-document term inside Eq. 5's sum, for one system."""
    try:
        obf_perplexity = float(row["obf_perplexity"])
    except (ValueError, KeyError, TypeError):
        # A degenerate generation gets a perplexity high enough to rule it out.
        obf_perplexity = 1e4

    return (
        weights["aa_distance"] * safe_log(float(row["aa_distance"]))
        + weights["meaning_similarity"] * safe_log(float(row[meaning_field]))
        + weights["obf_fluency"] * safe_log(float(row["obf_fluency"]))
        - weights["obf_perplexity"] * safe_log(obf_perplexity)
    )


def select_system(system_scores, operation="max"):
    """Pick one system from {system: {doc_id: score}} by its mean document score."""
    def mean_score(system):
        scores = system_scores[system]
        return sum(scores.values()) / len(scores) if scores else float("-inf")

    if operation == "max":
        return max(system_scores, key=mean_score)
    if operation == "min":
        return min(system_scores, key=mean_score)
    if operation == "median":
        return sorted(system_scores, key=mean_score)[len(system_scores) // 2]
    raise ValueError(f"Invalid operation: {operation}")


def score_authors(orig_docs, score_data, weights, meaning_field="meaning_similarity"):
    """Eq. 5's inner sum: {author: {system: {doc_id: score}}}."""
    author_level_scores = defaultdict(lambda: defaultdict(dict))

    for doc_id, doc in orig_docs.items():
        author = author_key(doc)
        for system, system_scores in score_data.items():
            row = system_scores.get(doc_id)
            if row is None:
                logger.warning("System %s has no scores for %s; skipping", system, doc_id)
                continue
            author_level_scores[author][system][doc_id] = document_score(row, weights, meaning_field)

    return {author: dict(systems) for author, systems in author_level_scores.items()}


def average_system_metrics(score_files, out_csv):
    """One row per system with the corpus averages reported in Tables 1 and 2."""
    rows = []
    for system, path in score_files.items():
        df = pd.read_csv(path)
        rows.append({
            "Method": system,
            "AADist": df["aa_distance"].mean(),
            "SBERT": df["meaning_similarity"].mean(),
            "SBERT (sent)": df["meaning_similarity_sent_level"].mean(),
            "METEOR": df["meteor"].mean(),
            "COLA": df["obf_fluency"].mean(),
            "COLA (orig)": df["orig_fluency"].mean(),
            "Perplexity": df["obf_perplexity"].mean(),
            "Perplexity (orig)": df["orig_perplexity"].mean(),
        })

    # The "Original" reference row: unobfuscated text is at zero distance from
    # itself and identical in content, and its fluency/perplexity are recorded
    # in every system's CSV (all systems score the same originals).
    if rows:
        reference = pd.read_csv(next(iter(score_files.values())))
        rows.insert(0, {
            "Method": "Original",
            "AADist": 0.0,
            "SBERT": 1.0,
            "SBERT (sent)": 1.0,
            "METEOR": 1.0,
            "COLA": reference["orig_fluency"].mean(),
            "COLA (orig)": reference["orig_fluency"].mean(),
            "Perplexity": reference["orig_perplexity"].mean(),
            "Perplexity (orig)": reference["orig_perplexity"].mean(),
        })

    df = pd.DataFrame(rows).round(4)
    df.to_csv(out_csv, index=False)
    logger.info("Average system metrics:\n%s", df.to_string(index=False))
    return df


def main(args):
    with open(args.systems) as handle:
        systems = json.load(handle)
    with open(args.score_files) as handle:
        score_files = json.load(handle)

    weights = dict(DEFAULT_WEIGHTS)
    if args.obf_weights:
        with open(args.obf_weights) as handle:
            weights.update(json.load(handle))
    logger.info("Weights: %s", weights)

    missing = set(systems) ^ set(score_files)
    if missing:
        raise ValueError(f"--systems and --score_files must cover the same systems; differ on {sorted(missing)}")

    orig_docs = read_jsonl_by_docid(args.orig_jsonl)
    obf_docs = {system: read_jsonl_by_docid(path) for system, path in systems.items()}
    score_data = {system: read_scores_csv(path) for system, path in score_files.items()}

    author_level_scores = score_authors(orig_docs, score_data, weights, args.meaning_field)
    selection = {
        author: select_system(system_scores, args.operation)
        for author, system_scores in author_level_scores.items()
    }

    out_dir = os.path.dirname(os.path.abspath(args.out_jsonl))
    os.makedirs(out_dir, exist_ok=True)

    # Emit each document from whichever system was chosen for its author.
    written = 0
    with open(args.out_jsonl, "w") as handle:
        for doc_id, doc in orig_docs.items():
            system = selection[author_key(doc)]
            obfuscated = obf_docs[system].get(doc_id)
            if obfuscated is None:
                logger.warning("Selected system %s has no output for %s; emitting original", system, doc_id)
                obfuscated = doc
            handle.write(json.dumps(obfuscated) + "\n")
            written += 1
    logger.info("Wrote %d documents to %s", written, args.out_jsonl)

    with open(os.path.join(out_dir, "author_level_scores.json"), "w") as handle:
        json.dump(author_level_scores, handle, indent=2)
    with open(os.path.join(out_dir, "system_selection_per_author.json"), "w") as handle:
        json.dump(selection, handle, indent=2)

    counts = Counter(selection.values())
    total = sum(counts.values())
    for system, count in counts.most_common():
        logger.info("%s selected for %d/%d authors (%.1f%%)", system, count, total, 100 * count / total)
    with open(os.path.join(out_dir, "system_selection_counts.json"), "w") as handle:
        json.dump(dict(counts), handle, indent=2)

    # OSO's own per-document metrics: the selected system's row for each document.
    oso_metrics_csv = os.path.join(out_dir, "oso_metrics.csv")
    oso_rows = [
        score_data[selection[author_key(doc)]][doc_id]
        for doc_id, doc in orig_docs.items()
        if doc_id in score_data[selection[author_key(doc)]]
    ]
    pd.DataFrame(oso_rows).to_csv(oso_metrics_csv, index=False)

    average_system_metrics(
        {**score_files, "OSO": oso_metrics_csv},
        os.path.join(out_dir, "avg_system_metrics.csv"),
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--orig_jsonl", required=True, help="Original documents (defines the author of each document)")
    parser.add_argument("--systems", required=True,
                        help='JSON mapping {"system name": "obfuscated jsonl path"}')
    parser.add_argument("--score_files", required=True,
                        help='JSON mapping {"system name": "metrics csv from oso.metrics.run_metrics"}')
    parser.add_argument("--out_jsonl", required=True, help="Where to write the selected obfuscation")
    parser.add_argument("--obf_weights", default=None, help="JSON of per-term weights (default: 1.0 each)")
    parser.add_argument("--operation", default="max", choices=["max", "min", "median"],
                        help="How to reduce per-system document scores (max reproduces the paper)")
    parser.add_argument("--meaning_field", default="meaning_similarity",
                        choices=["meaning_similarity", "meaning_similarity_sent_level"],
                        help="Document-level or sentence-averaged MS in Eq. 5")
    return parser


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    logger.info("Arguments: %s", args)
    main(args)
