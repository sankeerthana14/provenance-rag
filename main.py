#!/usr/bin/env python3
"""
main.py — PROVE-RAG Pipeline
=========================================
One entry point to reproduce every result in the paper, end to end:

    download  →  process (build benchmark)  →  experiments  →  figures

Run the whole thing:
    python main.py --stage all

Or run a single stage:
    python main.py --stage download        # raw datasets + base models
    python main.py --stage process         # build the evidence-state benchmark
    python main.py --stage experiments     # run all experiments (Tables V–IX)
    python main.py --stage figures         # regenerate the figures

Run one group of experiments only:
    python main.py --stage experiments --only detectors      # Table VI
    python main.py --stage experiments --only baselines      # Table V
    python main.py --stage experiments --only ablation       # Table VII
    python main.py --stage experiments --only cross_dataset  # Table VIII
    python main.py --stage experiments --only end_to_end     # Table IX

Useful flags:
    --dry-run         Print every command without running it (inspect the plan).
    --smoke           Tiny subset everywhere, for a fast end-to-end sanity check.
    --seed 42         Global seed (default 42, matches the paper).

Paths (override if your layout differs):
    --data-dir data   --models-dir models   --results-dir results

NOTE ON MODELS
--------------
'download' fetches the three base encoders (BERT/RoBERTa/DeBERTa) via the repo's
download_models script. Two further models are required for the baselines and the
end-to-end evaluation and must be obtained separately:
    - roberta-large-mnli        (NLI baseline)            -> models/roberta-large-mnli
    - meta-llama/Llama-3.1-8B-Instruct (gated on HF)      -> models/Llama-3.1-8B-Instruct
This script checks for them before the steps that need them and exits with a clear
message if they are missing.
"""

import argparse
import os
import subprocess
import sys

# --------------------------------------------------------------------------- #
# Repo layout: this file lives at the repo root; scripts live under scripts/.
# All sub-scripts assume they are run from the repo root (raw data is written to
# data/raw/...), so we set cwd=REPO_ROOT for every subprocess call.
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_PREP = os.path.join("scripts", "data_preprocessing")
MODEL_DEV = os.path.join("scripts", "model_dev")


def S(folder, name):
    """Resolve a script path under the repo."""
    return os.path.join(folder, name)


# Script locations (edit here only if you reorganize the repo) ---------------- #
SCRIPTS = {
    # data preprocessing
    "download_data":  S(DATA_PREP, "download_dataset.py"),
    "fever_wiki":     S(DATA_PREP, "FEVER_wiki_resolver.py"),
    "download_models": S(DATA_PREP, "downloading_model.py"),
    "to_schema":      S(DATA_PREP, "convert_to_unified_schema.py"),
    "variants":       S(DATA_PREP, "create_variants.py"),
    "merge_graphs":   S(DATA_PREP, "merge_and_build_graphs.py"),
    "preprocess":     S(DATA_PREP, "preprocess_v2.py"),
    # experiments
    "train":          S(MODEL_DEV, "train_detector_v2.py"),
    "competitor":     S(MODEL_DEV, "competitor_baseline.py"),
    "rag_competitor": S(MODEL_DEV, "rag_competitor_eval.py"),
    "cross_dataset":  S(MODEL_DEV, "cross_dataset_generalization.py"),
    "end_to_end":     S(MODEL_DEV, "end_to_end_eval.py"),
    "visualize":      S(MODEL_DEV, "visualization.py"),
}

# --------------------------------------------------------------------------- #
# Dataset → schema-file naming.
# merge_and_build_graphs.py locates files by SUBSTRING match against these exact
# pattern stems, so each schema file MUST be named to contain its stem. The
# create_variants step then writes "<stem>_<variant>.json", which the merge step
# also finds by substring. Do not rename these stems casually.

# (max_examples values are inferred from the stems in merge_and_build_graphs.py;
#  confirm they match your original run if you need bit-identical data.)
# --------------------------------------------------------------------------- #
SCHEMA_SPEC = {
    "train": [
        # (dataset, hf_split, max_examples, stem)
        ("hotpotqa", "train", 15000, "hotpotqa_train_15k"),
        ("musique",  "train", 30000, "musique_train_30k"),
        ("fever",    "train", 30000, "fever_train_30k"),
    ],
    "val": [
        ("hotpotqa", "validation",   2500, "hotpotqa_val_2.5k"),
        ("musique",  "validation",   5000, "musique_val_5k"),
        # FEVER's labelled dev split is usually "labelled_dev" on HF; adjust if needed.
        ("fever",    "labelled_dev", 7000, "fever_val_7k"),
    ],
}

# Table VI detector configs (Approach A = text-only, B = text + provenance features).
# "RoBERTa refined" is not a separate run: it is roberta_feat with entity_overlap
# ablated, produced by the ablation group below.
DETECTOR_EXPERIMENTS = [
    "bert_text", "bert_feat",
    "roberta_text", "roberta_feat",
    "deberta_text", "deberta_feat",
]

# Table VII feature ablation: roberta_feat with each feature removed in turn.
ABLATION_FEATURES = [
    "source_diversity", "text_resolution_rate", "avg_evidence_length",
    "min_evidence_length", "duplicate_rate", "document_overlap_rate",
    "entity_overlap", "evidence_count",
]


# --------------------------------------------------------------------------- #
# Command runner
# --------------------------------------------------------------------------- #
DRY_RUN = False


def run(cmd, desc=""):
    """Run a command from the repo root; stream output; abort on failure."""
    printable = " ".join(cmd)
    print("\n" + "-" * 78)
    if desc:
        print(f"# {desc}")
    print(f"$ {printable}")
    print("-" * 78, flush=True)
    if DRY_RUN:
        return
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        sys.exit(f"\n[FAILED] (exit {result.returncode}) {printable}")


def py(script_key, *args):
    """Build a `python <script> <args...>` command."""
    return [sys.executable, SCRIPTS[script_key], *map(str, args)]


def require_model(path, human_name, how):
    """Fail early with a helpful message if a required model folder is missing."""
    full = os.path.join(REPO_ROOT, path)
    if DRY_RUN:
        return
    if not os.path.isdir(full):
        sys.exit(
            f"\n[MISSING MODEL] {human_name} not found at: {path}\n"
            f"  How to get it: {how}\n"
            f"  (Re-run this stage once the model is in place.)"
        )


# --------------------------------------------------------------------------- #
# Stage 1: download raw data + base models
# --------------------------------------------------------------------------- #
def stage_download(cfg):
    print("\n========== STAGE: DOWNLOAD ==========")
    # Raw datasets -> data/raw/{hotpotqa_distractor, musique, fever_v1}
    run(py("download_data"), "Download HotpotQA, MuSiQue, FEVER (-> data/raw/)")
    # FEVER Wikipedia sentence lookup (needed to resolve FEVER evidence text)
    run(py("fever_wiki", "--output", os.path.join(cfg.data_dir, "fever_wiki_lookup.json")),
        "Build FEVER Wikipedia sentence lookup")
    # Base encoders -> models/{deberta-v3-base, roberta-base, bert-base-uncased}
    run(py("download_models", "--output_dir", cfg.models_dir),
        "Download base encoders (BERT / RoBERTa / DeBERTa)")
    print(
        "\n[ACTION REQUIRED] The base encoders are downloaded, but two more models\n"
        "are needed for the baselines and end-to-end eval:\n"
        f"  1. roberta-large-mnli            -> {cfg.models_dir}/roberta-large-mnli\n"
        "     (e.g. huggingface-cli download roberta-large-mnli "
        f"--local-dir {cfg.models_dir}/roberta-large-mnli)\n"
        f"  2. meta-llama/Llama-3.1-8B-Instruct (gated) -> {cfg.models_dir}/Llama-3.1-8B-Instruct\n"
        "     (request access on Hugging Face, then huggingface-cli download ...)\n"
    )


# --------------------------------------------------------------------------- #
# Stage 2: build the evidence-state benchmark
# --------------------------------------------------------------------------- #
def stage_process(cfg):
    print("\n========== STAGE: PROCESS (build benchmark) ==========")
    schema_dir = os.path.join(cfg.data_dir, "schema")
    variants_dir = os.path.join(cfg.data_dir, "variants")
    final_dir = os.path.join(cfg.data_dir, "final")
    processed_dir = os.path.join(cfg.data_dir, "processed")
    fever_lookup = os.path.join(cfg.data_dir, "fever_wiki_lookup.json")

    # 1) Convert each (dataset, split) to the unified provenance-aware schema.
    for split, specs in SCHEMA_SPEC.items():
        for dataset, hf_split, max_ex, stem in specs:
            out = os.path.join(schema_dir, f"{stem}.json")
            args = ["--dataset", dataset, "--split", hf_split, "--output", out]
            if dataset == "fever":
                args += ["--fever_wiki_lookup", fever_lookup]
            max_ex = cfg.smoke_examples if cfg.smoke else max_ex
            if max_ex:
                args += ["--max_examples", max_ex]
            run(py("to_schema", *args), f"Schema: {dataset} [{split}] -> {stem}.json")

    # 2) Create the three synthetic variants per schema file
    #    (sufficient stays as-is; insufficient/contradicted/superseded are generated).
    for split, specs in SCHEMA_SPEC.items():
        for dataset, hf_split, max_ex, stem in specs:
            in_path = os.path.join(schema_dir, f"{stem}.json")
            run(py("variants", "--input", in_path, "--variant", "all",
                   "--output_dir", variants_dir, "--seed", cfg.seed),
                f"Variants: {stem} (insufficient / contradicted / superseded)")

    # 3) Merge sufficient + variants, build evidence graphs, extract the 8 features.
    #    -> final_dir/unified_train.json, final_dir/unified_val.json
    run(py("merge_graphs", "--data_dir", schema_dir, "--variants_dir", variants_dir,
           "--output_dir", final_dir, "--seed", cfg.seed, "--balance"),
        "Merge + build evidence graphs + extract provenance features")

    # 4) Final splits: train->train(80%)+test(20%), keep val; structured [SEP]
    #    formatting + top-k evidence + gold answers for end-to-end eval.
    #    -> processed_dir/{train,val,test}.json   (these feed every experiment)
    run(py("preprocess",
           "--train_json", os.path.join(final_dir, "unified_train.json"),
           "--val_json", os.path.join(final_dir, "unified_val.json"),
           "--output_dir", processed_dir,
           "--top_k", 10, "--test_ratio", 0.2, "--seed", cfg.seed),
        "Final splits + structured formatting -> data/processed/{train,val,test}.json")

    print(f"\n[OK] Benchmark ready under: {processed_dir}/")


# --------------------------------------------------------------------------- #
# Stage 3: experiments
# --------------------------------------------------------------------------- #
def _paths(cfg):
    processed = os.path.join(cfg.data_dir, "processed")
    return {
        "train": os.path.join(processed, "train.json"),
        "val": os.path.join(processed, "val.json"),
        "test": os.path.join(processed, "test.json"),
        "processed_dir": processed,
    }


def exp_detectors(cfg):
    """Table VI: BERT/RoBERTa/DeBERTa, text-only (A) and +features (B)."""
    p = _paths(cfg)
    common = ["--train_path", p["train"], "--val_path", p["val"],
              "--test_path", p["test"], "--model_dir", cfg.models_dir,
              "--epochs", 1 if cfg.smoke else 5, "--seed", cfg.seed]
    for exp in DETECTOR_EXPERIMENTS:
        out = os.path.join(cfg.results_dir, exp)
        run(py("train", "--experiment", exp, "--output_dir", out, *common),
            f"[Table VI] Detector: {exp}")


def exp_baselines(cfg):
    """Table V: majority, LLM zero-shot, NLI, embedding-sim, RAGAS/CRAG/Self-RAG."""
    p = _paths(cfg)
    base = ["--train_path", p["train"], "--val_path", p["val"],
            "--model_dir", cfg.models_dir, "--seed", cfg.seed]

    # Majority floor + LLM zero-shot (both via the trainer)
    run(py("train", "--experiment", "majority",
           "--output_dir", os.path.join(cfg.results_dir, "majority"), *base),
        "[Table V] Majority baseline")
    require_model(os.path.join(cfg.models_dir, "Llama-3.1-8B-Instruct"),
                  "Llama-3.1-8B-Instruct",
                  "gated download from Hugging Face (meta-llama/Llama-3.1-8B-Instruct)")
    run(py("train", "--experiment", "llm_zeroshot",
           "--output_dir", os.path.join(cfg.results_dir, "llm_zeroshot"),
           "--llm_model", os.path.join(cfg.models_dir, "Llama-3.1-8B-Instruct"),
           "--max_llm_samples", 50 if cfg.smoke else 2000, *base),
        "[Table V] LLM zero-shot classification")

    # NLI (roberta-large-mnli) + embedding-similarity
    require_model(os.path.join(cfg.models_dir, "roberta-large-mnli"),
                  "roberta-large-mnli",
                  "huggingface-cli download roberta-large-mnli")
    run(py("competitor",
           "--test_path", p["test"],
           "--nli_model_path", os.path.join(cfg.models_dir, "roberta-large-mnli"),
           "--output_dir", os.path.join(cfg.results_dir, "baselines", "competitor"),
           "--max_samples", 200 if cfg.smoke else 0, "--seed", cfg.seed),
        "[Table V] NLI + embedding-similarity baselines")

    # RAGAS-style / CRAG-style / Self-RAG-style
    run(py("rag_competitor",
           "--test_path", p["test"],
           "--llm_path", os.path.join(cfg.models_dir, "Llama-3.1-8B-Instruct"),
           "--output_dir", os.path.join(cfg.results_dir, "baselines", "rag"),
           "--max_samples", 50 if cfg.smoke else 2000, "--seed", cfg.seed),
        "[Table V] RAGAS-style / CRAG-style / Self-RAG-style baselines")


def exp_ablation(cfg):
    """Table VII: roberta_feat with each of the 8 features removed in turn."""
    p = _paths(cfg)
    for feat in ABLATION_FEATURES:
        out = os.path.join(cfg.results_dir, "roberta_ablations", f"ablate_{feat}")
        run(py("train", "--experiment", "roberta_feat",
               "--train_path", p["train"], "--val_path", p["val"],
               "--test_path", p["test"], "--model_dir", cfg.models_dir,
               "--ablate_feature", feat,
               "--epochs", 1 if cfg.smoke else 5, "--seed", cfg.seed,
               "--output_dir", out),
            f"[Table VII] Ablation: remove {feat}"
            + ("  (entity_overlap == 'RoBERTa refined')" if feat == "entity_overlap" else ""))


def exp_cross_dataset(cfg):
    """Table VIII: train on two datasets, test on the held-out third (RoBERTa+feat)."""
    run(py("cross_dataset",
           "--data_dir", _paths(cfg)["processed_dir"],
           "--encoder_path", os.path.join(cfg.models_dir, "roberta-base"),
           "--output_dir", os.path.join(cfg.results_dir, "cross_dataset"),
           "--epochs", 1 if cfg.smoke else 5, "--seed", cfg.seed),
        "[Table VIII] Cross-dataset generalization")


def exp_end_to_end(cfg):
    """Table IX: Naive vs PROVE-RAG vs Oracle. Needs a trained roberta_feat detector."""
    p = _paths(cfg)
    detector_dir = os.path.join(cfg.results_dir, "roberta_feat")
    if not DRY_RUN and not os.path.isdir(os.path.join(REPO_ROOT, detector_dir)):
        sys.exit(
            f"\n[DEPENDENCY] End-to-end eval needs a trained detector at {detector_dir}.\n"
            "  Run the detectors first:  python main.py --stage experiments --only detectors"
        )
    require_model(os.path.join(cfg.models_dir, "Llama-3.1-8B-Instruct"),
                  "Llama-3.1-8B-Instruct",
                  "gated download from Hugging Face (meta-llama/Llama-3.1-8B-Instruct)")
    run(py("end_to_end",
           "--test_path", p["test"],
           "--detector_dir", detector_dir,
           "--detector_base_model", os.path.join(cfg.models_dir, "roberta-base"),
           "--llm_path", os.path.join(cfg.models_dir, "Llama-3.1-8B-Instruct"),
           "--output_dir", os.path.join(cfg.results_dir, "end_to_end"),
           "--max_questions", 100 if cfg.smoke else 1044),
        "[Table IX] End-to-end controlled remediation")


EXPERIMENT_GROUPS = {
    "detectors": exp_detectors,        # Table VI
    "baselines": exp_baselines,        # Table V
    "ablation": exp_ablation,          # Table VII
    "cross_dataset": exp_cross_dataset,  # Table VIII
    "end_to_end": exp_end_to_end,      # Table IX
}
# Order matters: detectors must run before end_to_end (it reuses roberta_feat).
EXPERIMENT_ORDER = ["detectors", "ablation", "baselines", "cross_dataset", "end_to_end"]


def stage_experiments(cfg):
    print("\n========== STAGE: EXPERIMENTS ==========")
    if cfg.only:
        if cfg.only not in EXPERIMENT_GROUPS:
            sys.exit(f"--only must be one of: {', '.join(EXPERIMENT_GROUPS)}")
        EXPERIMENT_GROUPS[cfg.only](cfg)
    else:
        for name in EXPERIMENT_ORDER:
            EXPERIMENT_GROUPS[name](cfg)


# --------------------------------------------------------------------------- #
# Stage 4: figures
# --------------------------------------------------------------------------- #
def stage_figures(cfg):
    print("\n========== STAGE: FIGURES ==========")
    # TODO: confirm visualization.py's exact arguments against your copy and adjust.
    run(py("visualize", "--results_dir", cfg.results_dir,
           "--output_dir", os.path.join(cfg.results_dir, "figures")),
        "Regenerate paper figures")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="PROVE-RAG reproduction pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--stage", required=True,
                        choices=["all", "download", "process", "experiments", "figures"],
                        help="Which stage to run.")
    parser.add_argument("--only", default=None,
                        choices=list(EXPERIMENT_GROUPS.keys()),
                        help="Run a single experiment group (with --stage experiments).")
    parser.add_argument("--data-dir", dest="data_dir", default="data")
    parser.add_argument("--models-dir", dest="models_dir", default="models")
    parser.add_argument("--results-dir", dest="results_dir", default="results")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the full plan without executing anything.")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny subsets everywhere, for a fast end-to-end test.")
    parser.add_argument("--smoke-examples", type=int, default=200,
                        help="--max_examples used per dataset when --smoke is set.")
    cfg = parser.parse_args()

    global DRY_RUN
    DRY_RUN = cfg.dry_run

    if cfg.smoke:
        print("[SMOKE MODE] Using tiny subsets — results will NOT match the paper.")

    if cfg.stage == "all":
        stage_download(cfg)
        stage_process(cfg)
        cfg.only = None
        stage_experiments(cfg)
        stage_figures(cfg)
    elif cfg.stage == "download":
        stage_download(cfg)
    elif cfg.stage == "process":
        stage_process(cfg)
    elif cfg.stage == "experiments":
        stage_experiments(cfg)
    elif cfg.stage == "figures":
        stage_figures(cfg)

    print("\n[DONE]")


if __name__ == "__main__":
    main()
