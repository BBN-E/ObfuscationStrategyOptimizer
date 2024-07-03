"""Delta Acc.: how far each system drops a Writeprints attributor's accuracy.

The privacy column of Table 1. A Writeprints-Static attribution classifier is
trained on the training authors, evaluated on the *original* test documents to
establish a ceiling, then re-evaluated on each system's obfuscated documents.
Delta Acc. is the drop, so higher is better.

The classifier follows Mutant-X (Mahmood et al., 2019) and the Avengers-Ensemble
setup it builds on: Writeprints-Static features into a random forest / SVM soft
-voting ensemble.

Example:
    python -m oso.evaluation.acc_drop \
        --train_jsonl data/amt-10/X_train.jsonl \
        --test_jsonl data/amt-10/X_test.jsonl \
        --systems configs/systems_amt10.json \
        --out_csv results/amt-10/acc_drop.csv
"""
import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from tqdm import tqdm

from oso.evaluation import writeprints_static as ws
from oso.utils import author_key, load_jsonl, logger, setup_logging


def featurize(records, desc="Writeprints features"):
    """{doc_id: feature vector} for a list of HRS-style records."""
    features = {}
    for record in tqdm(records, desc=desc, leave=False):
        features[record["documentID"]] = ws.calculateFeatures(record["fullText"])
    return features


def build_classifier(seed=42):
    """Soft-voting random forest + SVM, as in Mutant-X's attribution setup."""
    return VotingClassifier(
        estimators=[
            ("rfc", RandomForestClassifier(n_estimators=100, random_state=seed)),
            ("svc", SVC(kernel="rbf", probability=True, random_state=seed)),
        ],
        voting="soft",
    )


def train_attributor(train_jsonl, seed=42):
    records = list(load_jsonl(train_jsonl))
    features = featurize(records, desc="Training features")
    X = np.asarray([features[record["documentID"]] for record in records])
    y = np.asarray([author_key(record) for record in records])

    logger.info("Training attributor on %d documents by %d authors", len(X), len(set(y)))
    classifier = build_classifier(seed)
    classifier.fit(X, y)
    return classifier


def accuracy_on(classifier, jsonl_path, labels, desc):
    """Accuracy of the attributor on one file, scored against the true authors."""
    records = [record for record in load_jsonl(jsonl_path) if record["documentID"] in labels]
    if not records:
        raise ValueError(f"No documents in {jsonl_path} match the reference test set")

    features = featurize(records, desc=desc)
    X = np.asarray([features[record["documentID"]] for record in records])
    predictions = classifier.predict(X)
    truth = np.asarray([labels[record["documentID"]] for record in records])
    return float((predictions == truth).mean()), len(records)


def main(args):
    with open(args.systems) as handle:
        systems = json.load(handle)

    if args.classifier and os.path.exists(args.classifier):
        logger.info("Loading attributor from %s", args.classifier)
        with open(args.classifier, "rb") as handle:
            classifier = pickle.load(handle)
    else:
        classifier = train_attributor(args.train_jsonl, args.seed)
        if args.classifier:
            os.makedirs(os.path.dirname(os.path.abspath(args.classifier)), exist_ok=True)
            with open(args.classifier, "wb") as handle:
                pickle.dump(classifier, handle)
            logger.info("Saved attributor to %s", args.classifier)

    # The original test documents define both the label set and the accuracy ceiling.
    labels = {record["documentID"]: author_key(record) for record in load_jsonl(args.test_jsonl)}
    original_accuracy, n = accuracy_on(classifier, args.test_jsonl, labels, "Original")
    logger.info("Original: accuracy %.4f over %d documents", original_accuracy, n)

    rows = [{"Method": "Original", "Accuracy": original_accuracy, "Delta Acc.": 0.0, "N": n}]
    for system, path in systems.items():
        accuracy, n = accuracy_on(classifier, path, labels, system)
        rows.append({
            "Method": system,
            "Accuracy": accuracy,
            "Delta Acc.": original_accuracy - accuracy,
            "N": n,
        })
        logger.info("%s: accuracy %.4f, Delta Acc. %.4f", system, accuracy, original_accuracy - accuracy)

    df = pd.DataFrame(rows).round(4)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    logger.info("Wrote %s\n%s", args.out_csv, df.to_string(index=False))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_jsonl", required=True, help="Documents used to train the attributor")
    parser.add_argument("--test_jsonl", required=True, help="Original test documents (the accuracy ceiling)")
    parser.add_argument("--systems", required=True,
                        help='JSON mapping {"system name": "obfuscated jsonl path"}')
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--classifier", default=None,
                        help="Cache the trained attributor here (loaded if it already exists)")
    parser.add_argument("--seed", type=int, default=42)
    return parser


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    logger.info("Arguments: %s", args)
    main(args)
