# PROVE-RAG: Provenance-Aware Evidence-State Assessment for Trustworthy Agentic RAG

> **Status:** Under review at *IEEE Transactions on Knowledge and Data Engineering (TKDE)*.
> This repository is **anonymized for double-blind review** — please do not attempt to de-anonymize it.

PROVE-RAG adds an explicit **evidence-state layer** between retrieval and generation in agentic
Retrieval-Augmented Generation (RAG). The core observation is that *retrieval relevance is not the
same as evidence reliability*: a passage can be topically relevant while still being incomplete,
internally contradictory, or superseded by newer information.

Retrieved evidence is normalized into a **provenance-aware schema** and classified into four
action-oriented states, each mapped to an agentic action:

| Evidence state | Agentic action   |
| -------------- | ---------------- |
| sufficient     | answer           |
| insufficient   | retrieve more    |
| contradicted   | flag conflict    |
| superseded     | verify currency  |

This repository contains everything needed to (1) construct the evidence-state benchmark from public
QA / fact-verification datasets and (2) reproduce all experiments reported in the paper. Quantitative
results and analysis are in the paper and are intentionally **not** reproduced here.

---

## Repository structure

```
provenance-rag/
├── main.py                              # single entry point for the full pipeline
├── requirements.txt
├── README.md
├── scripts/
│   ├── data_preprocessing/              # build the benchmark
│   │   ├── download_dataset.py          # download FEVER / HotpotQA / MuSiQue
│   │   ├── FEVER_wiki_resolver.py       # build FEVER Wikipedia sentence lookup
│   │   ├── downloading_model.py         # download base encoders
│   │   ├── convert_to_unified_schema.py # raw data -> provenance-aware schema
│   │   ├── create_variants.py           # generate the 4 evidence-state variants
│   │   ├── merge_and_build_graphs.py    # merge + evidence graphs + 8 features
│   │   ├── preprocess_v2.py             # final splits + structured formatting
│   │   ├── preprocess_light.py          # (optional) lightweight intermediate files
│   │   └── inspecting_datasets.py       # (optional) dataset inspection
│   └── model_dev/                       # detectors, baselines, evaluation
│       ├── train_detector_v2.py         # detector training (encoders + features)
│       ├── train_detector.py            # (imported by the cross-dataset script)
│       ├── competitor_baseline.py       # NLI + embedding-similarity baselines
│       ├── rag_competitor_eval.py       # RAGAS- / CRAG- / Self-RAG-style baselines
│       ├── cross_dataset_generalization.py
│       ├── end_to_end_eval.py           # agentic remediation evaluation
│       ├── visualization.py             # figures
│       ├── real_case_study.py           # (optional) qualitative case study
│       ├── run_train_detector.sh        # example SLURM launcher
│       └── run_feature_ablate.sh        # example SLURM launcher (ablation sweep)
├── data/      # created at runtime (gitignored)
├── models/    # populated by the download stage (gitignored)
└── results/   # experiment outputs (gitignored)
```

`data/`, `models/`, and `results/` are not tracked; they are produced by the pipeline.

---

## Installation

Python 3.10+ and a CUDA-capable GPU are recommended (the transformer detectors and the LLM-based
steps assume a GPU).

```bash
pip install -r requirements.txt

# Additional packages used by the experiments:
pip install torch transformers scikit-learn pandas rouge-score

# spaCy model used for entity-based features:
python -m spacy download en_core_web_sm
```

For fully deterministic benchmark construction, set `PYTHONHASHSEED=0` before the process stage:

```bash
export PYTHONHASHSEED=0
```

---

## Models

The download stage fetches the three base encoders automatically into `models/`:
`bert-base-uncased`, `roberta-base`, `microsoft/deberta-v3-base`.

Two further models are required and must be obtained separately:

| Model                              | Used for                              | Destination                       |
| ---------------------------------- | ------------------------------------- | --------------------------------- |
| `roberta-large-mnli`               | NLI baseline                          | `models/roberta-large-mnli`       |
| `meta-llama/Llama-3.1-8B-Instruct` | LLM zero-shot, RAG baselines, end-to-end | `models/Llama-3.1-8B-Instruct` |

The Llama model is gated on Hugging Face and requires an access request. `main.py` checks for both
models before the steps that need them and exits with instructions if either is missing.

---

## Reproducing the results

Everything runs through `main.py`. **Inspect the plan first** with `--dry-run`, and use `--smoke` for a
fast end-to-end sanity check on tiny subsets before committing a full run.

```bash
python main.py --stage all --dry-run        # print every command, run nothing
python main.py --stage all --smoke          # tiny subsets, fast (results won't match the paper)
python main.py --stage all                  # full reproduction
```

Or run one stage at a time:

```bash
python main.py --stage download             # raw datasets + base encoders
python main.py --stage process              # build the evidence-state benchmark
python main.py --stage experiments          # all experiments
python main.py --stage figures              # regenerate figures
```

### Pipeline stages

1. **download** — fetch the three source datasets and the base encoders; build the FEVER Wikipedia lookup.
2. **process** — convert each dataset to the provenance-aware schema, generate the four evidence-state
   variants, build per-instance evidence graphs and extract the eight provenance features, then produce
   the final `train` / `val` / `test` splits under `data/processed/`.
3. **experiments** — train the detectors, run the baselines, the feature ablation, the cross-dataset
   evaluation, and the end-to-end agentic evaluation.
4. **figures** — regenerate the figures from the experiment outputs.

The benchmark splits in `data/processed/{train,val,test}.json` are consumed by every experiment, so
the **process** stage must complete before **experiments**.

---

## Experiments → paper tables

Run a single group with `--only`:

```bash
python main.py --stage experiments --only detectors        # Table VI
python main.py --stage experiments --only baselines        # Table V
python main.py --stage experiments --only ablation         # Table VII
python main.py --stage experiments --only cross_dataset    # Table VIII
python main.py --stage experiments --only end_to_end       # Table IX
```

| Group           | Paper table | What it evaluates                                              |
| --------------- | ----------- | ------------------------------------------------------------- |
| `baselines`     | Table V     | Whether existing relevance / entailment / RAG-evaluation / LLM signals detect evidence states |
| `detectors`     | Table VI    | Transformer detectors, text-only vs. text + provenance features |
| `ablation`      | Table VII   | Contribution of each provenance feature (one removed at a time) |
| `cross_dataset` | Table VIII  | Generalization: train on two datasets, test on the held-out third |
| `end_to_end`    | Table IX    | Agentic remediation (naive vs. evidence-state-guided vs. oracle) |

`end_to_end` reuses the trained `roberta_feat` detector, so run `detectors` first (the default
ordering in `--stage experiments` already does this). The `entity_overlap` run inside `ablation`
corresponds to the refined detector configuration discussed in the paper.

---

## Benchmark and data

The evidence-state benchmark is derived from three public datasets — **FEVER**, **HotpotQA**, and
**MuSiQue** — by generating four evidence-state variants per instance under a unified schema. Raw and
processed data are gitignored and regenerated by the **process** stage; the construction is seeded
(`--seed 42`). The source datasets retain their original licenses; please consult each source for terms
and cite them appropriately when using this benchmark.

---

## Citation and license

Citation details and license will be added upon publication. This repository is provided for
peer-review reproduction.
