# ObfuscationStrategyOptimizer

Reference implementation for the PrivateNLP @ ACL 2024 paper
[**"Improving Authorship Privacy: Adaptive Obfuscation with the Dynamic Selection of Techniques"**](https://aclanthology.org/2024.privatenlp-1.14/)
(Kandula, Karakos, Qiu and Ulicny; RTX BBN Technologies).

Authorship obfuscation rewrites text so an attribution system can no longer
identify who wrote it. Every obfuscation technique trades that off against
content and fluency differently, and no single one wins everywhere. The
Obfuscation Strategy Optimizer (OSO) runs several techniques over the same text
and picks, per author, the one whose output best balances style concealment,
meaning preservation and fluency.

```
                    ┌──────────────────────┐
  Machine           │                      │
  Translation ─────▶│                      │
  STEER      ──────▶│  Obfuscation         │
  Llama-2 7B ──────▶│  Strategy Optimizer  │──▶ privatized output
  Recursive  ──────▶│  (per-author argmax) │
  Llama-2 7B        │                      │
                    └──────────▲───────────┘
                               │
              AADist · SBERT · CoLA · perplexity
```

## The selection objective

Each candidate's output is scored on four axes (Sections 2, Eqs. 1–4):

| Term | Meaning | Model |
| --- | --- | --- |
| `AADist` | cosine **distance** between authorship embeddings of the original and obfuscated text — higher is more private | LUAR (Rivera-Soto et al., 2021) |
| `MS` | meaning similarity | SentenceTransformers (Reimers and Gurevych, 2019) |
| `CoLA` | grammatical acceptability | RoBERTa fine-tuned on CoLA (Warstadt et al., 2019) |
| `PPL` | perplexity — lower is more natural | GPT-2 |

OSO combines them into one objective and takes the argmax per author `a` over
that author's `n` documents (Eq. 5):

```
OSO_a = argmax_i  (1/n) Σ_doc [ log(AADist_i) + log(MS_i) + log(CoLA_i) − log(PPL_i) ]
```

Selection is per author rather than per document on purpose: an attribution
system aggregates evidence across an author's writing, so mixing techniques
within one author leaves a recognizable signature. The logarithms make the four
terms multiplicative, so no single axis can dominate by scale, and a weight per
term (`configs/weights_paper.json`, all `1.0` for the paper) lets the trade-off
be re-tuned without touching code.

## Installation

```bash
git clone https://github.com/BBN-E/ObfuscationStrategyOptimizer.git
cd ObfuscationStrategyOptimizer

python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
python -c "import nltk; nltk.download('wordnet'); nltk.download('punkt'); nltk.download('omw-1.4')"
```

A CUDA GPU is expected. The metrics run on ~8GB; the Llama-2 7B GPTQ candidate
needs ~6GB (the quantization takes the model from 38GB to 3.4GB, and a document
from ~4 minutes to ~30 seconds on a V100).

## Data

`data/README.md` covers all three corpora (EBG/AMT, Blog, HRS-HIATUS), the
JSONL format every script reads, and how to convert the pickled Mutant-X
distributions:

```bash
python scripts/prepare_datasets.py --dataset_dir data/amt-10 --source amt
```

## Running it

The whole of Table 1 for one dataset:

```bash
bash scripts/run_pipeline.sh data/amt-10 work/amt-10
```

That script is four stages, each usable on its own.

### 1. Generate candidates

The two LLM candidates ship here; one run produces both, since pass 2 rewrites
pass 1's output:

```bash
python -m obfuscation_systems.llm_rewriting.generate_rephrasing_gptq \
    --input_jsonl data/amt-10/X_test.jsonl \
    --output_path work/amt-10/obf/llama2 \
    --llama_model_path TheBloke/Llama-2-7B-GPTQ \
    --num_passes 2 --remove_short_sentences --check_length
```

The Machine Translation candidate (a Fairseq transformer trained on
Llama-2-restyled parallel data) and the STEER candidate depend on models
released elsewhere; `obfuscation_systems/mt/` and `obfuscation_systems/steer/`
explain how to plug their output in. **Any** system that emits JSONL keyed by
`documentID` can be added as a candidate — that is the point of the design.

List whatever you generated in a systems config:

```json
{
  "Machine Translation":   "work/amt-10/obf/mt.jsonl",
  "STEER Style Transfer":  "work/amt-10/obf/steer.jsonl",
  "Llama-2 7B":            "work/amt-10/obf/llama2_pass_1.jsonl",
  "Recursive Llama-2 7B":  "work/amt-10/obf/llama2_pass_2.jsonl"
}
```

### 2. Score them

```bash
python scripts/score_all.py \
    --orig_jsonl data/amt-10/X_test.jsonl \
    --systems work/amt-10/systems.json \
    --metrics_dir work/amt-10/metrics \
    --out_score_files work/amt-10/scores.json
```

One CSV row per document per system, with `aa_distance`, `meaning_similarity`,
`meaning_similarity_sent_level`, `meteor`, `orig_fluency`, `obf_fluency`,
`orig_perplexity` and `obf_perplexity`. (`python -m oso.metrics.run_metrics`
does a single system, with flags to swap any of the four models.)

### 3. Select

```bash
python -m oso.selector \
    --orig_jsonl data/amt-10/X_test.jsonl \
    --systems work/amt-10/systems.json \
    --score_files work/amt-10/scores.json \
    --obf_weights configs/weights_paper.json \
    --out_jsonl work/amt-10/results/oso/final_obfuscated.jsonl
```

Alongside the selected output this writes `system_selection_per_author.json`,
`system_selection_counts.json` (how often each technique won),
`author_level_scores.json` and `avg_system_metrics.csv`.

### 4. Evaluate adversarially

Quality metrics alone do not show that obfuscation worked, so the output is put
in front of an attribution system.

**Δ Acc.** (Tables 1) — the drop in a Writeprints-Static attributor's accuracy,
the same feature set Mutant-X attacks:

```bash
python -m oso.evaluation.acc_drop \
    --train_jsonl data/amt-10/X_train.jsonl \
    --test_jsonl data/amt-10/X_test.jsonl \
    --systems work/amt-10/systems_with_oso.json \
    --out_csv work/amt-10/results/acc_drop.csv
```

**EER** (Table 2, HRS) — the equal error rate of a LUAR attributor matching
query documents to candidate authors:

```bash
python -m oso.evaluation.eer \
    --query_jsonl data/hrs/queries.jsonl \
    --candidate_jsonl data/hrs/candidates.jsonl \
    --systems work/hrs/systems_with_oso.json \
    --out_csv work/hrs/results/eer.csv
```

Then join either with the quality metrics into the paper's table layout:

```bash
python -m oso.evaluation.report \
    --avg_metrics work/amt-10/results/oso/avg_system_metrics.csv \
    --privacy work/amt-10/results/acc_drop.csv \
    --out_csv work/amt-10/results/table1.csv
```

## Published results

Table 1 (AMT / EBG 10-author and Blog 10-author). Mutant-X and JamDec are
external baselines, not candidates OSO selects among.

| Dataset | Method | AADist | Δ Acc. | SBERT | METEOR | CoLA |
| --- | --- | --- | --- | --- | --- | --- |
| AMT | Original | 0.0 | 0.0 | 1.0 | 1.0 | 0.88 |
| AMT | Mutant-X | – | 0.39 | – | **0.84** | 0.53 |
| AMT | JamDec | – | 0.41 | – | 0.61 | 0.79 |
| AMT | Machine Translation | 0.2133 | 0.29 | 0.64 | 0.75 | 0.72 |
| AMT | STEER Style Transfer | 0.1976 | 0.30 | 0.64 | 0.50 | 0.76 |
| AMT | Llama-2 7B | 0.1955 | 0.31 | **0.87** | 0.36 | 0.91 |
| AMT | Recursive Llama-2 7B | 0.2087 | 0.42 | 0.85 | 0.35 | 0.92 |
| AMT | **OSO** | **0.2441** | **0.43** | 0.86 | 0.42 | **0.94** |
| BLOG | Original | 0.0 | 0.0 | 1.0 | 1.0 | 0.78 |
| BLOG | Mutant-X | – | 0.44 | – | **0.55** | 0.47 |
| BLOG | JamDec | – | 0.32 | – | 0.53 | 0.74 |
| BLOG | Machine Translation | 0.3184 | 0.25 | 0.58 | 0.48 | 0.70 |
| BLOG | STEER Style Transfer | 0.4202 | 0.32 | 0.57 | 0.45 | 0.90 |
| BLOG | Llama-2 7B | 0.3726 | 0.49 | **0.81** | 0.35 | 0.88 |
| BLOG | Recursive Llama-2 7B | 0.4335 | 0.33 | 0.78 | 0.31 | 0.89 |
| BLOG | **OSO** | **0.4416** | **0.51** | 0.78 | 0.32 | **0.90** |

Table 2 (HRS):

| Method | AADist | EER | SBERT | METEOR | CoLA |
| --- | --- | --- | --- | --- | --- |
| Original | 0.0 | 0.0340 | 1.0 | 1.0 | 0.82 |
| Machine Translation | 0.2462 | 0.0817 | 0.68 | **0.48** | 0.72 |
| STEER Style Transfer | 0.2075 | 0.0885 | 0.63 | 0.47 | 0.78 |
| Llama-2 7B | 0.3242 | 0.1742 | 0.65 | 0.37 | 0.90 |
| Recursive Llama-2 7B | **0.3427** | 0.1857 | **0.77** | 0.36 | 0.91 |
| **OSO** | 0.3347 | **0.2058** | **0.77** | 0.37 | **0.93** |

The pattern to read here is not that OSO tops every column — it does not. Each
individual technique wins somewhere and gives something up elsewhere: Llama-2
holds meaning best but conceals style least; STEER conceals style but loses
content. OSO is at or near the top on every axis at once, which is what picking
per author buys. METEOR stays low across all the neural methods because they
paraphrase: meaning survives through synonyms that token overlap cannot see,
which is why SBERT is the better read on content.

### Reproducing exactly

Note two differences between what is written here and the numbers above:

* The paper computes CoLA with a RoBERTa-**large** classifier and perplexity
  with **GPT-2 large**. The defaults here are `textattack/roberta-base-CoLA` and
  `gpt2-large`; `--cola_model` and `--ppl_model` take any HuggingFace id.
* `AADist` is computed against the public LUAR checkpoint
  (`rrivera1849/LUAR-MUD`), whereas the paper used the LUAR model as deployed in
  the HIATUS evaluation pipeline. Absolute distances therefore shift somewhat;
  the ordering across systems is what the selection depends on.

The candidates are sampled (`do_sample`, temperature 0.7), so the generated text
differs run to run even at a fixed seed on different hardware.

## Layout

```
oso/
  selector.py              OSO itself: Eq. 5, per-author argmax
  metrics/
    aa_distance.py         LUAR authorship distance      (Eq. 1)
    meaning.py             SBERT meaning similarity      (Eq. 2)
    fluency.py             CoLA acceptability            (Eq. 3)
    perplexity.py          GPT-2 perplexity              (Eq. 4)
    truncate.py            sentence-aware truncation to the LM window
    run_metrics.py         scores one system -> per-document CSV
  evaluation/
    acc_drop.py            Writeprints attributor, Δ Acc.
    eer.py                 LUAR attributor, equal error rate
    writeprints_static.py  Writeprints-Static features
    report.py              builds the paper's tables
obfuscation_systems/
  llm_rewriting/           Llama-2 7B GPTQ, single and recursive passes
  mt/                      flatten/reassemble for the MT candidate
  steer/                   how to plug in STEER
scripts/
  prepare_datasets.py      pickled corpora -> JSONL
  score_all.py             score every candidate in a systems config
  run_pipeline.sh          all four stages
```

## Limitations

From Section "Limitations" of the paper, and worth knowing before trusting the
output:

* OSO's privacy signal is only as good as the attribution model behind it. If
  LUAR is weak on your genre, `AADist` is measuring little.
* CoLA encodes a standard-English notion of fluency. Optimizing for it can push
  text toward a register that is wrong for the setting.
* The candidates are pre-trained LMs, which hallucinate. SBERT and METEOR
  measure similarity, not factual fidelity — neither one catches an invented
  detail or a dropped qualifier. Information-extraction–based checks are the
  right fix and are not implemented here.

## Citation

```bibtex
@inproceedings{kandula-etal-2024-improving,
    title     = "Improving Authorship Privacy: Adaptive Obfuscation with the Dynamic Selection of Techniques",
    author    = "Kandula, Hemanth and Karakos, Damianos and Qiu, Haoling and Ulicny, Brian",
    booktitle = "Proceedings of the Fifth Workshop on Privacy in Natural Language Processing",
    month     = aug,
    year      = "2024",
    pages     = "137--142",
    publisher = "Association for Computational Linguistics",
    url       = "https://aclanthology.org/2024.privatenlp-1.14/"
}
```

## Acknowledgments

We would like to thank Skyler Hallinan, Jillian Fisher, and Yejin Choi from the
University of Washington for fruitful discussions on the topic of authorship
obfuscation in the IARPA HIATUS project.

This research is based upon work supported in part by the Office of the Director
of National Intelligence (ODNI), Intelligence Advanced Research Projects
Activity (IARPA), via 2022-22072200003. The views and conclusions contained
herein are those of the authors and should not be interpreted as necessarily
representing the official policies, either expressed or implied, of ODNI, IARPA,
or the U.S. Government. The U.S. Government is authorized to reproduce and
distribute reprints for governmental purposes notwithstanding any copyright
annotation therein.

Writeprints-Static feature extraction is vendored from
[Avengers-Ensemble](https://github.com/Haroon96/Avengers-Ensemble); the
perplexity implementation is adapted from HuggingFace `evaluate`.
