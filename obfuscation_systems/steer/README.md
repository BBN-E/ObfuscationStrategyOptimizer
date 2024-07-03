# STEER Style Transfer candidate

The third candidate in Section 3.1 rewrites each document in the style of a
target domain (we used `english_tweet`) with STEER, from
[Hallinan et al., 2023](https://aclanthology.org/2023.findings-emnlp.506/).

STEER is released separately, so it is not vendored here. To reproduce this
candidate:

1. Clone and install STEER from https://github.com/skyler-hallinan/STEER and
   download the released expert-reinforcement checkpoints.
2. Run STEER over the sentences of each document, with `english_tweet` as the
   target style. `obfuscation_systems/mt/jsonl_to_sentences.py` produces the
   sentence-per-line input, and `obfuscation_systems/mt/sentences_to_jsonl.py`
   turns STEER's output back into an obfuscated JSONL — the same flatten /
   reassemble pair the MT candidate uses.
3. Register the resulting JSONL in your `configs/systems_*.json` under a name
   such as `STEER Style Transfer`, and score it with `oso.metrics.run_metrics`
   like any other candidate.

Nothing in OSO is specific to these four candidates: any system that emits an
HRS-style JSONL keyed by `documentID` can be added the same way.
