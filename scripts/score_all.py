"""Score every candidate system in a systems config, in one process.

``oso.metrics.run_metrics`` handles a single system. This loops over a systems
config, reuses the same arguments for each, and writes the scores config that
``oso.selector`` expects.

Example:
    python scripts/score_all.py \
        --orig_jsonl data/amt-10/X_test.jsonl \
        --systems configs/systems_amt10.json \
        --metrics_dir metrics/amt-10 \
        --out_score_files configs/scores_amt10.json
"""
import argparse
import json
import os
import re

from oso.metrics.run_metrics import build_parser as metrics_parser
from oso.metrics.run_metrics import main as run_metrics
from oso.utils import logger, setup_logging


def slugify(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def main(args, metrics_args):
    with open(args.systems) as handle:
        systems = json.load(handle)

    os.makedirs(args.metrics_dir, exist_ok=True)
    score_files = {}

    for name, obf_jsonl in systems.items():
        out_file = os.path.join(args.metrics_dir, f"{slugify(name)}.csv")
        score_files[name] = out_file

        if os.path.exists(out_file) and not args.overwrite:
            logger.info("%s: %s exists, skipping (pass --overwrite to rescore)", name, out_file)
            continue

        logger.info("== Scoring %s ==", name)
        metrics_args.orig_jsonl = args.orig_jsonl
        metrics_args.obf_jsonl = obf_jsonl
        metrics_args.out_file = out_file
        run_metrics(metrics_args)

    with open(args.out_score_files, "w") as handle:
        json.dump(score_files, handle, indent=2)
    logger.info("Wrote %s", args.out_score_files)


def build_parser():
    # Inherit every model/device flag from run_metrics; override the three
    # per-system paths it takes, which this script supplies per system.
    parent = metrics_parser()
    for action in parent._actions:
        if action.dest in {"orig_jsonl", "obf_jsonl", "out_file"}:
            action.required = False
            action.help = argparse.SUPPRESS

    parser = argparse.ArgumentParser(
        description=__doc__, parents=[parent], conflict_handler="resolve",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--orig_jsonl", required=True, help="Original (unobfuscated) documents")
    parser.add_argument("--systems", required=True, help='JSON mapping {"system name": "obfuscated jsonl"}')
    parser.add_argument("--metrics_dir", required=True, help="Directory for the per-system metrics CSVs")
    parser.add_argument("--out_score_files", required=True, help="Scores config to write, for oso.selector")
    parser.add_argument("--overwrite", action="store_true", help="Rescore systems that already have a CSV")
    return parser


if __name__ == "__main__":
    setup_logging()
    parsed = build_parser().parse_args()
    logger.info("Arguments: %s", parsed)
    main(parsed, parsed)
