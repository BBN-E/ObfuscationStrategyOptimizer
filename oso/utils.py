"""Shared helpers: I/O for HRS-style JSONL and numerically safe logs."""
import bz2
import gzip
import io
import json
import logging
import math
import sys

logger = logging.getLogger("OSO")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=level,
        stream=sys.stdout,
    )


def safe_log(value, *, threshold=1e-10):
    """log() that floors its argument, so a zero metric cannot produce -inf."""
    if value >= threshold:
        return math.log(value)
    return math.log(threshold)


def fopen(filename, mode="rt", encoding="utf-8", **kwargs):
    """Drop-in replacement for open() that handles .gz/.bz2 and '-' (stdin/stdout)."""
    if filename == "-":
        if "w" in mode:
            return io.TextIOWrapper(sys.stdout.buffer, encoding=encoding)
        return io.TextIOWrapper(sys.stdin.buffer, encoding=encoding)

    if filename.endswith(".gz"):
        _fopen = gzip.open
        if "b" not in mode and "t" not in mode:
            mode = mode + "t"
    elif filename.endswith(".bz2"):
        _fopen = bz2.open
        if "b" not in mode and "t" not in mode:
            mode = mode + "t"
    else:
        _fopen = open

    if "b" in mode:
        return _fopen(filename, mode=mode, **kwargs)
    return _fopen(filename, mode=mode, encoding=encoding, **kwargs)


def load_jsonl(file_path):
    """Yield one parsed object per line."""
    with fopen(file_path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl_by_docid(file_path):
    """Read an HRS-style JSONL into {documentID: record}."""
    return {record["documentID"]: record for record in load_jsonl(file_path)}


def write_jsonl(records, file_path):
    with fopen(file_path, "w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def author_key(record):
    """Stable author key for a document record.

    HRS documents carry ``authorSetIDs`` (a set of authors) or ``authorIDs``.
    Both are lists, so they are turned into a hashable, JSON-safe string.
    """
    ids = record.get("authorSetIDs") or record["authorIDs"]
    return str(tuple(ids))


def get_sentences(record, text_field="fullText"):
    """Return the document's sentences as a list of strings.

    Uses the pre-computed ``sentences`` map when present (obfuscation systems
    operate sentence by sentence and keep the alignment), otherwise falls back
    to splitting ``fullText`` on newlines.
    """
    sentences = record.get("sentences")
    if sentences:
        ordered = sorted(sentences.items(), key=lambda item: int(item[0]))
        return [sent["text"] for _, sent in ordered]
    return [line for line in record[text_field].split("\n") if line.strip()]
