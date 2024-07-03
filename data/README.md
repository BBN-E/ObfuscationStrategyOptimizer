# Datasets

The paper uses three corpora. None is redistributed here; this file records
where each comes from and what shape the code expects.

## Expected format

Every script reads and writes HRS-style JSONL — one JSON object per line:

```json
{"documentID": "81-1151815.txt", "authorIDs": ["1151815"], "fullText": "...", "source": "blog"}
```

`documentID` joins everything together (metrics CSVs, system outputs, the final
selection), and `authorIDs` defines the author that OSO selects a system for.
Obfuscated files add a `sentences` field, `{"0": {"text": "..."}, ...}`, which
keeps the obfuscated sentences aligned with the originals; when it is absent the
metrics fall back to splitting `fullText` on newlines.

## Extended Brennan–Greenstadt (EBG / AMT)

Brennan et al., 2012. The paper uses the 10-author version. The pickled splits
distributed with [Mutant-X](https://github.com/asad1996172/Mutant-X) convert
with:

```bash
python scripts/prepare_datasets.py --dataset_dir data/amt-10 --source amt
```

## Blog Authorship Corpus

Schler et al., 2006 — diary entries from blogger.com, 10-author version. Same
conversion:

```bash
python scripts/prepare_datasets.py --dataset_dir data/blog-10 --source blog
```

## HRS-HIATUS

Research datasets from the IARPA HIATUS program
(https://www.iarpa.gov/research-programs/hiatus), covering BoardGameGeek,
Instructables, GlobalVoices and StackExchange (liberal arts and STEM). They are
distributed through the program rather than publicly, and already arrive as
JSONL in the format above, split into query and candidate sets. The query set is
what gets obfuscated; the candidate set is what the attribution system matches
against, which is what `oso.evaluation.eer` expects.
