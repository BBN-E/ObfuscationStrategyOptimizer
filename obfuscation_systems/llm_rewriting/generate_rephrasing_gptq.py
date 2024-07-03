"""LLM Rewriting and Recursive LLM Rewriting candidates (Section 3.1).

Paraphrases a document sentence by sentence with a GPTQ-quantized Llama-2 7B.
``--num_passes N`` writes N files: pass 1 is the "LLM Rewriting" candidate, and
each later pass rewrites the previous pass's output, so pass 2 is the
"Recursive LLM Rewriting" candidate. GPTQ keeps the 7B model at ~3.4GB and a
document at ~30s on a V100.

Three filters guard against a rewrite that is worse than the original; each
falls back to the original sentence rather than accepting the paraphrase:

* ``--use_meaning_similarity``  drifted too far in meaning
* ``--remove_short_sentences``  the sentence was too short to usefully rewrite
* ``--check_length``            the rewrite ran more than twice as long

Example:
    python -m obfuscation_systems.llm_rewriting.generate_rephrasing_gptq \
        --input_jsonl data/amt-10/X_test.jsonl \
        --output_path obf/amt-10/llama2 \
        --llama_model_path TheBloke/Llama-2-7B-GPTQ \
        --num_passes 2 --remove_short_sentences --check_length
"""
import argparse
import json
import os
import random

import numpy as np
import spacy
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from oso.utils import fopen, logger, setup_logging

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("sentencizer")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Model:
    def __init__(self, **kwargs):
        self._model = None
        self._tokenizer = None

    def load(self, model_path, revision="gptq-4bit-64g-actorder_True"):
        self._model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            trust_remote_code=False,
            revision=revision,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def forward(self, prompt, temperature=1.0, max_new_tokens=512, **kwargs):
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, padding=False)
        input_ids = inputs.input_ids.to(self._model.device)
        generation_output = self._model.generate(
            input_ids=input_ids,
            temperature=temperature,
            repetition_penalty=1.2,
            do_sample=True,
            top_p=0.95,
            top_k=40,
            max_new_tokens=max_new_tokens,
        )
        output_text = self._tokenizer.decode(generation_output[0], skip_special_tokens=True)
        return output_text.replace(prompt, "").strip()


def assess_sentence(cur_sentence, rephrased_sentence, filter_config):
    """Decide whether to keep the original sentence instead of the rewrite."""
    similarity_score = None
    keep_original = False

    if filter_config["use_meaning_similarity"]:
        similarity_score = nlp(cur_sentence).similarity(nlp(rephrased_sentence))
        if similarity_score < 0.25:
            keep_original = True

    if filter_config["remove_short_sentences"] and len(cur_sentence) < 20:
        keep_original = True

    if filter_config["check_length"] and len(cur_sentence) * 2 < len(rephrased_sentence):
        keep_original = True

    return similarity_score, keep_original


def rephrase_sentence(sentence, model_instance, seed):
    """One paraphrase of one sentence, stripped of the model's preamble."""
    prompt = f'''[INST] <<SYS>>
# You rephrase the input sentence.
# <</SYS>>
# Input sentence: {sentence}. [/INST] Here is a rephrased version of the given sentence: '''
    set_seed(seed)
    generated_text = model_instance.forward(prompt)
    generated_sentences = [str(s) for s in nlp(generated_text).sents]
    if len(generated_sentences) == 0:
        return " "

    # Llama sometimes narrates the rewrite ("Here is a rewrite of...") before
    # producing it; skip past any such sentence.
    sentence_index = 0
    first_generated_sentence = generated_sentences[0].strip()
    while "rewrite" in first_generated_sentence.lower() or "re-write" in first_generated_sentence.lower():
        sentence_index += 1
        if len(generated_sentences) <= sentence_index:
            return " "
        first_generated_sentence = generated_sentences[sentence_index].strip()

    return first_generated_sentence


def query_llama_json(input_jsonl, output_path, llama_model_path, num_passes, filter_config, seed):
    model_instance = Model()
    model_instance.load(llama_model_path)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_files = {i: fopen(f"{output_path}_pass_{i}.jsonl", "w") for i in range(1, num_passes + 1)}

    try:
        with fopen(input_jsonl) as handle:
            for line in tqdm(handle, desc="Rewriting"):
                obj = json.loads(line)
                author_ids = obj["authorIDs"]
                document_id = obj["documentID"]

                if "sentences" in obj:
                    sentences = {i: obj["sentences"][i]["text"] for i in sorted(obj["sentences"].keys())}
                else:
                    sentences = {i: str(s) for i, s in enumerate(nlp(obj["fullText"]).sents)}

                rephrased = {pass_num: {} for pass_num in range(1, num_passes + 1)}
                for sent_idx, sentence in sentences.items():
                    cur_sentence = sentence
                    for pass_num in range(1, num_passes + 1):
                        rephrased_sentence = rephrase_sentence(cur_sentence, model_instance, seed)
                        similarity_score, keep_original = assess_sentence(
                            cur_sentence, rephrased_sentence, filter_config
                        )
                        if keep_original:
                            rephrased[pass_num][sent_idx] = {"text": cur_sentence}
                        else:
                            rephrased[pass_num][sent_idx] = {"text": rephrased_sentence}
                            # The next pass rewrites this pass's output: the recursion.
                            cur_sentence = rephrased_sentence

                        if similarity_score is not None:
                            rephrased[pass_num][sent_idx]["sim_score"] = similarity_score

                for pass_num, output_file in output_files.items():
                    ordered = dict(sorted(rephrased[pass_num].items(), key=lambda item: int(item[0])))
                    output_file.write(json.dumps({
                        "authorIDs": author_ids,
                        "documentID": document_id,
                        "sentences": ordered,
                        "fullText": "\n".join(sent["text"] for sent in ordered.values()),
                    }) + "\n")
                    output_file.flush()
    finally:
        for output_file in output_files.values():
            output_file.close()

    for pass_num in range(1, num_passes + 1):
        logger.info("Wrote %s_pass_%d.jsonl", output_path, pass_num)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_path", required=True,
                        help="Prefix; each pass is written to <prefix>_pass_<n>.jsonl")
    parser.add_argument("--llama_model_path", default="TheBloke/Llama-2-7B-GPTQ")
    parser.add_argument("--num_passes", type=int, default=2,
                        help="1 = LLM Rewriting; 2 = also Recursive LLM Rewriting")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_meaning_similarity", action="store_true",
                        help="Keep the original sentence when meaning similarity is low")
    parser.add_argument("--remove_short_sentences", action="store_true",
                        help="Keep the original sentence when it is short")
    parser.add_argument("--check_length", action="store_true",
                        help="Keep the original sentence when the rewrite is more than twice as long")
    return parser


if __name__ == "__main__":
    setup_logging()
    args = build_parser().parse_args()
    logger.info("Arguments: %s", args)
    set_seed(args.seed)

    query_llama_json(
        input_jsonl=args.input_jsonl,
        output_path=args.output_path,
        llama_model_path=args.llama_model_path,
        num_passes=args.num_passes,
        filter_config={
            "use_meaning_similarity": args.use_meaning_similarity,
            "remove_short_sentences": args.remove_short_sentences,
            "check_length": args.check_length,
        },
        seed=args.seed,
    )
