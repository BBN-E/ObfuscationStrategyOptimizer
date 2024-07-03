"""Assemble the paper's result tables from the per-system outputs.

Joins the averaged quality metrics (`avg_system_metrics.csv`, written by
`oso.selector`) with the privacy column produced by `oso.evaluation.acc_drop`
(Delta Acc., Table 1) or `oso.evaluation.eer` (EER, Table 2) and prints the
table in the paper's column order.

Example:
    python -m oso.evaluation.report \
        --avg_metrics results/amt-10/oso/avg_system_metrics.csv \
        --privacy results/amt-10/acc_drop.csv \
        --out_csv results/amt-10/table1.csv
"""
import argparse
import os

import pandas as pd

from oso.utils import logger, setup_logging

PRIVACY_COLUMNS = ["Delta Acc.", "EER"]
TABLE_COLUMNS = ["Method", "AADist", "Delta Acc.", "EER", "SBERT", "METEOR", "COLA"]


def main(args):
    metrics = pd.read_csv(args.avg_metrics)
    privacy = pd.read_csv(args.privacy)

    keep = ["Method"] + [column for column in PRIVACY_COLUMNS if column in privacy.columns]
    if len(keep) == 1:
        raise ValueError(f"{args.privacy} has no {' or '.join(PRIVACY_COLUMNS)} column")

    table = metrics.merge(privacy[keep], on="Method", how="outer")

    columns = [column for column in TABLE_COLUMNS if column in table.columns]
    table = table[columns].round(4)

    # Report order: the reference first, OSO last.
    rank = {"Original": 0, "OSO": 2}
    table = table.sort_values("Method", key=lambda s: s.map(lambda m: rank.get(m, 1)), kind="stable")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s\n%s", args.out_csv, table.to_string(index=False))
    print(table.to_markdown(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--avg_metrics", required=True, help="avg_system_metrics.csv from oso.selector")
    parser.add_argument("--privacy", required=True, help="acc_drop.csv or eer.csv")
    parser.add_argument("--out_csv", required=True)
    return parser


if __name__ == "__main__":
    setup_logging()
    main(build_parser().parse_args())
