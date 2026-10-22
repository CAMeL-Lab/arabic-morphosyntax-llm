# [Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models](https://arxiv.org/abs/2603.16718)

Code for the EMNLP 2026 paper by Mohamed Adel, Bashar Alhafni, and Nizar Habash. This repository supports the paper's Arabic morphosyntactic tagging and CATiB dependency parsing experiments: zero-shot and retrieved few-shot prompting, gold-tokenized and raw-text parsing, and cascaded morphology-to-parsing prediction. It contains four paper system prompts, runners for GPT-5.2, Gemini 3 Flash, Llama 4 Scout, and Qwen3-Next, and the paper's five shot-selection methods.

## Repository layout

```text
src/arabic_morphosyntax_llm/          Benchmark CLI, conversion, retrieval, evaluation
src/arabic_morphosyntax_llm/prompts/  Four experiment system prompts
requirements.txt                    Paper benchmark dependencies, including vLLM on Linux
camel_parser -> ../camel_parser      Upstream parser with multiroot decoding
camel_conll -> ../camel_conll        Upstream CoNLL evaluator
tests/                              Code checks
data/                               Local treebanks (ignored by Git)
runs/                               Local predictions and scores (ignored by Git)
```

No treebanks, model weights, predictions, or results are included.

## Installation

Use Python 3.11 for the paper benchmark environment. The requirements include the API clients, CAMeLBERT dependencies, and vLLM on Linux x86_64. vLLM is skipped on other platforms; run the local-model benchmarks on a supported GPU machine.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Prepare data

Put licensed data in the ignored `data/` directory. Suggested locations:

```text
data/patb/train.tsv          data/patb/test.tsv
data/cameltb/train.conllu    data/cameltb/test.conllu
```

The PATB TSV needs a `word` column and the 14 PATB feature columns. The CATiB file needs ten CoNLL columns per token and `# text = ...` metadata for raw parsing. Convert Train for shots and Dev or Test for benchmarking:

```bash
arabic-morphosyntax convert patb-tsv --input data/patb/train.tsv --output data/patb/train.jsonl
arabic-morphosyntax convert patb-tsv --input data/patb/test.tsv --output data/patb/test.jsonl
arabic-morphosyntax convert catib-conllu --input data/cameltb/train.conllu --output data/cameltb/train.jsonl
arabic-morphosyntax convert catib-conllu --input data/cameltb/test.conllu --output data/cameltb/test.jsonl
```

Each JSONL row has a stable `id` and task fields: `morph_tokens` and `morphology` for PATB, or `text`, `parse_tokens`, and `parse` for CATiB. `morph_tokens` are PATB words; `parse_tokens` are CATiB tree tokens. Use `merge-gold` only for aligned PATB–CATiB files with matching IDs when running the two call `cascade`.

## Run LLM benchmarks

Tasks: `morphology`, `parse-gold`, `parse-raw`, `cascade`. Model profiles (`arabic-morphosyntax models` shows their IDs):

| Profile | Paper model | Backend |
| --- | --- | --- |
| `gpt5_2` | GPT-5.2 | OpenAI Responses API |
| `gemini3` | Gemini 3 Flash | Gemini API |
| `llama4` | Llama 4 Scout | vLLM Chat API |
| `qwen3` | Qwen3-Next-80B-A3B-Thinking | vLLM Chat API |

Set `OPENAI_API_KEY` or `GEMINI_API_KEY` for hosted models.

For Llama 4 or Qwen3, start a [vLLM server](https://docs.vllm.ai/en/latest/getting_started/quickstart/) on a supported GPU machine. With the installed paper environment active, serve one model at a time:

```bash
vllm serve meta-llama/Llama-4-Scout-17B-16E-Instruct --host 0.0.0.0 --port 8000
```

To benchmark Qwen3, stop the Llama server and run:

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Thinking --host 0.0.0.0 --port 8000
```

Run `arabic-morphosyntax` from the repo's `.venv` in another terminal. The configured endpoint is `http://localhost:8000/v1`; pass `--base-url http://SERVER:8000/v1` when vLLM runs on another machine. See the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) for GPU-specific setup and model memory requirements.

Zero shot:

```bash
arabic-morphosyntax run parse-raw --model gpt5_2 \
  --input data/cameltb/test.jsonl --output runs/gpt5_raw.jsonl
```

Generate ten Train demonstrations using the paper's highest chrF++ setting:

```bash
arabic-morphosyntax shots parse-gold \
  --input data/cameltb/test.jsonl --demonstrations data/cameltb/train.jsonl \
  --shots 10 --retrieval chrf-high --output runs/cameltb_shots.jsonl
arabic-morphosyntax run parse-gold --model llama4 \
  --base-url http://localhost:8000/v1 \
  --input data/cameltb/test.jsonl --demonstrations data/cameltb/train.jsonl \
  --shots 10 --shot-map runs/cameltb_shots.jsonl \
  --output runs/llama4_gold.jsonl
```

`--retrieval` accepts `random`, `chrf-high`, `chrf-low`, `cosine-high`, and `cosine-low`. The high settings choose the most similar Train sentences; the low settings choose the least similar. For example, to generate the least similar CAMeLBERT demonstrations:

```bash
arabic-morphosyntax shots morphology \
  --input data/patb/test.jsonl --demonstrations data/patb/train.jsonl \
  --shots 10 --retrieval cosine-low --output runs/patb_cosine_low.jsonl
```

Cosine selection encodes Train and input sentences once per command using attention-mask mean pooling and cosine similarity. Its default model is `CAMeL-Lab/bert-base-arabic-camelbert-msa`, as described in the paper. The original experiment script defaults to the `-ca` checkpoint; pass `--embedding-model CAMeL-Lab/bert-base-arabic-camelbert-ca` to reproduce that script's default. `--embedding-device`, `--embedding-batch-size`, and `--embedding-max-length` control encoding. Pass the generated file as `--shot-map` to `run` so multiple LLMs use identical demonstrations.

Use PATB JSONL with `morphology`. Use `--resume` to continue a run. `render` saves requests without calling a model. The four system prompts in `src/arabic_morphosyntax_llm/prompts/` match the experiment YAMLs in `arabic_syntax_hpc_after_rebuttal`.

Score morphology with:

```bash
arabic-morphosyntax evaluate morphology \
  --predictions runs/gpt5_morphology.jsonl --gold data/patb/test.jsonl
```

## CAMeLParser and CAMeL CoNLL

The `camel_parser/` and `camel_conll/` entries are relative symlinks to sibling clones. Git stores these links but does not clone their targets. After cloning this repo, clone the upstream repositories beside it. Follow their READMEs for installation, parser weights, and CAMeL Tools resources:

```bash
cd ..
git clone https://github.com/CAMeL-Lab/camel_parser.git
git clone https://github.com/CAMeL-Lab/camel_conll.git
cd arabic-morphosyntax-llm
```

The current [CAMeLParser](https://github.com/CAMeL-Lab/camel_parser) CLI uses **multiroot by default**. It accepts one sentence per line. The commands below use `jq` to extract raw text or gold CATiB tokens from the converted JSONL. For the paper's raw baseline, install the licensed S31 morphology database as described by CAMeLParser and use `-b calima-msa-s31`.

```bash
jq -r '.text' data/cameltb/test.jsonl > data/cameltb/test.raw.txt
jq -r '.parse_tokens | join(" ")' data/cameltb/test.jsonl > data/cameltb/test.tokens.txt
mkdir -p runs/camelparser_raw runs/camelparser_gold runs/llama4_gold
python camel_parser/text_to_conll_cli.py -f text -b calima-msa-s31 \
  -i data/cameltb/test.raw.txt > runs/camelparser_raw/test.conllu
python camel_parser/text_to_conll_cli.py -f tokenized \
  -i data/cameltb/test.tokens.txt > runs/camelparser_gold/test.conllu
```

Convert LLM parse JSONL to CoNLL using this repo's format exporter. Then run the [CAMeL CoNLL](https://github.com/CAMeL-Lab/camel_conll) evaluator directly on either LLM or CamelParser CoNLL output:

```bash
arabic-morphosyntax export-conll \
  --predictions runs/llama4_gold.jsonl --output runs/llama4_gold/test.conllu
python camel_conll/conll_evaluation.py \
  -g data/cameltb/test.conllu -p runs/llama4_gold/test.conllu -x -n -a
python camel_conll/conll_evaluation.py \
  -g data/cameltb/test.conllu -p runs/camelparser_raw/test.conllu -x -n -a
```


## Citation

If you use this code, please cite the paper ([BibTeX](CITATION.bib)):

```bibtex
@inproceedings{adel-etal-2026-arabic-morphosyntactic,
  title = {Arabic Morphosyntactic Tagging and Dependency Parsing with Large Language Models},
  author = {Adel, Mohamed and Alhafni, Bashar and Habash, Nizar},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing (EMNLP 2026)},
  address = {Budapest, Hungary},
  year = {2026}
}
```
