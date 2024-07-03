"""Convert the EBG (AMT) and Blog corpora into the HRS-style JSONL used here.

Both corpora ship, in the Mutant-X / Avengers-Ensemble distributions, as
pickled lists of rows. This writes one JSONL line per document:

    {"documentID": ..., "authorIDs": [...], "fullText": ..., "source": ...}

which is the only format every script in this repository reads.

Example:
    python scripts/prepare_datasets.py --dataset_dir data/amt-10
"""
import argparse
import json
import os
import pickle

from tqdm import tqdm


def convert_pickle(pickle_path, output_path, source):
    """One pickled corpus file -> one JSONL file. Returns the authors seen."""
    with open(pickle_path, "rb") as handle:
        rows = pickle.load(handle)

    authors = set()
    with open(output_path, "w") as out:
        for row in tqdm(rows, desc=f"Converting {os.path.basename(pickle_path)}"):
            # Row layout from the Mutant-X data release:
            # (_, documentID, numeric author index, author label, fullText)
            document_id, author_index, author_label, full_text = row[1], int(row[2]), row[3], row[4]
            authors.add(author_label)
            out.write(json.dumps({
                "documentID": document_id,
                "authorIDs": [author_label],
                "author_id": author_index,
                "fullText": full_text,
                "source": source,
            }) + "\n")

    return sorted(authors)


def main(args):
    for name in os.listdir(args.dataset_dir):
        if not name.endswith(".pickle"):
            continue
        pickle_path = os.path.join(args.dataset_dir, name)
        output_path = os.path.join(args.dataset_dir, name.replace(".pickle", ".jsonl"))
        authors = convert_pickle(pickle_path, output_path, args.source)

        # Keep the author list beside each split; the attribution evaluation
        # uses it to line up query and candidate labels.
        labels_name = "query-labels.txt" if "test" in name else "candidate-labels.txt"
        with open(os.path.join(args.dataset_dir, labels_name), "w") as handle:
            for author in authors:
                handle.write(f"{author}\n")

        print(f"{output_path}: {len(authors)} authors")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset_dir", required=True,
                        help="Directory holding X_train.pickle / X_test.pickle")
    parser.add_argument("--source", default="hrs", help="Value for the 'source' field")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
