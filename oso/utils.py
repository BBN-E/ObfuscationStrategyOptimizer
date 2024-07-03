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


_SENTENCIZER = None


def _sentencize(text):
    """Split raw text into sentences with spaCy's rule-based sentencizer.

    The same splitter the obfuscation systems use, so a document split here
    lines up index-for-index with the sentences a system emits for it.
    """
    global _SENTENCIZER
    if _SENTENCIZER is None:
        import spacy
        from spacy.lang.en import English

        nlp = English()
        nlp.add_pipe("sentencizer")
        _SENTENCIZER = nlp

    # The sentencizer has no length ceiling of its own, but spaCy's Doc does;
    # blog entries run to ~11k words, so raise it to fit the longest of them.
    _SENTENCIZER.max_length = max(getattr(_SENTENCIZER, "max_length", 10 ** 6), len(text) + 1)
    return [sent.text.strip() for sent in _SENTENCIZER(text).sents if sent.text.strip()]


def get_sentences(record, text_field="fullText"):
    """Return the document's sentences as a list of strings.

    Uses the pre-computed ``sentences`` map when present -- obfuscation systems
    rewrite sentence by sentence and record the result there, which is what
    keeps an obfuscated document aligned with its original. Raw corpora have no
    such field, so their text is sentencized on the fly. Newlines alone are not
    enough: blog entries arrive as a single unbroken paragraph, and treating one
    as a single "sentence" would collapse the per-sentence fluency and meaning
    metrics into one truncated blob.
    """
    sentences = record.get("sentences")
    if sentences:
        ordered = sorted(sentences.items(), key=lambda item: int(item[0]))
        return [sent["text"] for _, sent in ordered]
    return _sentencize(record[text_field])
