"""Equal error rate of a LUAR attributor against each obfuscation system.

The privacy column of Table 2 (HRS). Each query document is embedded with LUAR
and scored against every candidate author's embedding; the EER is the operating
point where the false accept rate equals the false reject rate. A higher EER
means the attributor is less able to tell the true author from an impostor, so
higher is better for the obfuscation system.

Example:
    python -m oso.evaluation.eer \
        --query_jsonl data/hrs/queries.jsonl \
        --candidate_jsonl data/hrs/candidates.jsonl \
        --systems configs/systems_hrs.json \
        --out_csv results/hrs/eer.csv
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from oso.metrics.aa_distance import AuthorshipDistanceMetric
from oso.utils import author_key, load_jsonl, logger, setup_logging


def equal_error_rate(scores, labels):
    """EER over impostor/genuine similarity scores.

    ``labels`` is 1 where the pair is genuine (same author). The threshold is
    swept over the observed scores and the crossing of FAR and FRR is returned.
    """
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)

    genuine = scores[labels == 1]
    impostor = scores[labels == 0]
    if len(genuine) == 0 or len(impostor) == 0:
        raise ValueError("EER needs at least one genuine and one impostor pair")

    thresholds = np.unique(scores)
    best_eer, best_gap = 1.0, float("inf")
    for threshold in thresholds:
        far = float((impostor >= threshold).mean())  # impostors accepted
        frr = float((genuine < threshold).mean())    # genuine pairs rejected
        gap = abs(far - frr)
        if gap < best_gap:
            best_gap, best_eer = gap, (far + frr) / 2
    return best_eer


def author_texts(jsonl_path):
    """{author: [document text, ...]} for one file."""
    grouped = defaultdict(list)
    for record in load_jsonl(jsonl_path):
        grouped[author_key(record)].append(record["fullText"])
    return grouped


def score_eer(metric, query_jsonl, candidate_embeddings):
    """EER of query documents in ``query_jsonl`` against the candidate authors."""
    queries = list(load_jsonl(query_jsonl))
    query_texts = {record["documentID"]: record["fullText"] for record in queries}
    query_authors = {record["documentID"]: author_key(record) for record in queries}
    query_embeddings = metric.embed_documents(query_texts, desc="LUAR (queries)")

    candidates = sorted(candidate_embeddings)
    candidate_matrix = np.vstack([candidate_embeddings[author] for author in candidates])

    scores, labels = [], []
    for doc_id, embedding in query_embeddings.items():
        similarities = cosine_similarity(embedding.reshape(1, -1), candidate_matrix)[0]
        true_author = query_authors[doc_id]
        for author, similarity in zip(candidates, similarities):
            scores.append(similarity)
            labels.append(1 if author == true_author else 0)

    return equal_error_rate(scores, labels)


def main(args):
    with open(args.systems) as handle:
        systems = json.load(handle)

    metric = AuthorshipDistanceMetric(args)

    # Candidate authors are never obfuscated -- they are what the attributor knows.
    candidate_embeddings = metric.embed_authors(author_texts(args.candidate_jsonl))
    logger.info("Embedded %d candidate authors", len(candidate_embeddings))

    rows = []
    for name, path in [("Original", args.query_jsonl)] + list(systems.items()):
        eer = score_eer(metric, path, candidate_embeddings)
        rows.append({"Method": name, "EER": eer})
        logger.info("%s: EER %.4f", name, eer)

    df = pd.DataFrame(rows).round(4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s\n%s", args.out_csv, df.to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query_jsonl", required=True, help="Original query documents")
    parser.add_argument("--candidate_jsonl", required=True, help="Candidate author documents")
    parser.add_argument("--systems", required=True,
                        help='JSON mapping {"system name": "obfuscated query jsonl path"}')
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--luar_model", default="rrivera1849/LUAR-MUD")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--episode_length", type=int, default=16)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    logger.info("Arguments: %s", args)
    main(args)
