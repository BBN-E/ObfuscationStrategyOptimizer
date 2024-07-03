# Datasets

Two of the paper's three corpora are included here, in the 10-author versions
the paper reports on and already converted to the JSONL format every script
reads. Nothing needs preprocessing before `scripts/run_pipeline.sh`.

| Directory | Corpus | Authors | Train docs | Test docs | Avg words/doc |
| --- | --- | --- | --- | --- | --- |
| `amt-10/` | Extended Brennan–Greenstadt (EBG) | 10 | 120 | 49 | ~490 |
| `blog-10/` | Blog Authorship Corpus | 10 | 800 | 200 | ~1100 |

`X_test.jsonl` is what gets obfuscated and reported on. `X_train.jsonl` trains
the Writeprints attribution classifier that `oso.evaluation.acc_drop` measures
the accuracy drop against — it is never obfuscated, because the attacker is
assumed to hold clean samples of the author's writing.

## Format

One JSON object per line:

```json
{"documentID": "81-1151815.txt", "authorIDs": ["1151815"], "author_id": 1,
 "fullText": "...", "source": "blog"}
```

`documentID` joins everything together — metrics CSVs, each system's output,
and the final selection. `authorIDs` defines the author that OSO selects a
technique for. `author_id` is a numeric index kept from the original release;
nothing here depends on it.

Obfuscated files add a `sentences` field, `{"0": {"text": "..."}, ...}`, which
keeps the rewritten sentences aligned with the originals so the sentence-level
meaning and fluency metrics can compare them pairwise. When it is absent the
metrics fall back to splitting `fullText` on newlines.

## Sources

**Extended Brennan–Greenstadt (EBG / AMT)** — Brennan et al., 2012. Short
paragraphs collected through Amazon Mechanical Turk, written to a controlled
prompt, so topic varies less than style does.

**Blog Authorship Corpus** — Schler et al., 2006. Diary-style entries from
blogger.com: informal, far more varied in length, and noisier. The two corpora
stress obfuscation differently, which is why the paper reports both.

Both are redistributed here in converted form, as they are in the
[JAMDEC](https://github.com/jfisher52/JAMDecoding) and
[Mutant-X](https://github.com/asad1996172/Mutant-X) releases this work compares
against. If you start from those releases' pickled splits instead:

```bash
python scripts/prepare_datasets.py --dataset_dir data/amt-10 --source amt
```

## HRS-HIATUS (not included)

Table 2 uses the HRS research datasets from the IARPA HIATUS program
(https://www.iarpa.gov/research-programs/hiatus): BoardGameGeek, Instructables,
GlobalVoices and StackExchange (liberal arts and STEM), 114 authors and 885
query documents averaging 862 words. They are distributed through the program
rather than publicly, so they are not here.

They arrive already in the format above, split into a query set and a candidate
set. The query set is what gets obfuscated; the candidate set is what the
attribution system matches against, which is the split
`oso.evaluation.eer` expects:

```bash
python -m oso.evaluation.eer \
    --query_jsonl data/hrs/queries.jsonl \
    --candidate_jsonl data/hrs/candidates.jsonl \
    --systems work/hrs/systems_with_oso.json \
    --out_csv work/hrs/results/eer.csv
```
