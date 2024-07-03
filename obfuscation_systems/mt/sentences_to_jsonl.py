"""Reassemble MT output back into obfuscated documents.

Inverse of ``jsonl_to_sentences.py``: takes the MT model's sentence-per-line
output plus the index written alongside its input, and emits an obfuscated
JSONL with the same document and sentence structure as the other candidates.

Example:
    python -m obfuscation_systems.mt.sentences_to_jsonl \
        --input_sentences work/mt/output.txt \
        --input_index work/mt/index.tsv \
        --output_jsonl obf/amt-10/mt.jsonl
"""
import argparse
import csv
import os
from collections import defaultdict

from oso.utils import fopen, logger, setup_logging, write_jsonl


def main(args):
    with fopen(args.input_sentences) as handle:
        translated = [line.rstrip("\n") for line in handle]

    with fopen(args.input_index) as handle:
        index_rows = list(csv.DictReader(handle, delimiter="\t"))

    if len(translated) != len(index_rows):
        raise ValueError(
            f"{args.input_sentences} has {len(translated)} lines but "
            f"{args.input_index} describes {len(index_rows)} sentences"
        )

    documents = {}
    sentences = defaultdict(dict)
    for row, sentence in zip(index_rows, translated):
        doc_id = row["documentID"]
        documents[doc_id] = row["authorIDs"].split(",")
        sentences[doc_id][int(row["sentence_index"])] = sentence

    records = []
    for doc_id, author_ids in documents.items():
        ordered = dict(sorted(sentences[doc_id].items()))
        records.append({
            "documentID": doc_id,
            "authorIDs": author_ids,
            "sentences": {str(i): {"text": text} for i, text in ordered.items()},
            "fullText": "\n".join(ordered.values()),
        })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)), exist_ok=True)
    write_jsonl(records, args.output_jsonl)
    logger.info("Wrote %d documents to %s", len(records), args.output_jsonl)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_sentences", required=True, help="MT output, one sentence per line")
    parser.add_argument("--input_index", required=True, help="TSV written by jsonl_to_sentences.py")
    parser.add_argument("--output_jsonl", required=True)
    return parser


if __name__ == "__main__":
    setup_logging()
    main(build_parser().parse_args())
