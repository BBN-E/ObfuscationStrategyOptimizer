"""Flatten documents to one sentence per line, for the Machine Translation candidate.

The MT candidate is a transformer trained with Fairseq on parallel data
(original text -> Llama-2 restyled text), so it consumes and produces plain
sentence-per-line files. This script writes that file plus a TSV index that
records which document and position each line came from, so
``sentences_to_jsonl.py`` can reassemble the documents afterwards.

Example:
    python -m obfuscation_systems.mt.jsonl_to_sentences \
        --input_jsonl data/amt-10/X_test.jsonl \
        --output_sentences work/mt/input.txt \
        --output_index work/mt/index.tsv
"""
import argparse
import csv
import os

from oso.utils import fopen, get_sentences, load_jsonl, logger, setup_logging


def main(args):
    for path in (args.output_sentences, args.output_index):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    n_docs = n_sents = 0
    with fopen(args.output_sentences, "w") as sentences_out, fopen(args.output_index, "w") as index_out:
        writer = csv.writer(index_out, delimiter="\t")
        writer.writerow(["line", "documentID", "authorIDs", "sentence_index"])

        for record in load_jsonl(args.input_jsonl):
            sentences = get_sentences(record)
            for sentence_index, sentence in enumerate(sentences):
                # Newlines would break the one-sentence-per-line contract.
                sentences_out.write(sentence.replace("\n", " ").strip() + "\n")
                writer.writerow([
                    n_sents,
                    record["documentID"],
                    ",".join(str(a) for a in record["authorIDs"]),
                    sentence_index,
                ])
                n_sents += 1
            n_docs += 1

    logger.info("Wrote %d sentences from %d documents to %s", n_sents, n_docs, args.output_sentences)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_sentences", required=True, help="One sentence per line, for the MT model")
    parser.add_argument("--output_index", required=True, help="TSV mapping each line back to its document")
    return parser


if __name__ == "__main__":
    setup_logging()
    main(build_parser().parse_args())
