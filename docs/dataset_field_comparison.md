# Dataset Field Comparison for PROVE-RAG

Field-by-field reference for the three source datasets (HotpotQA, FEVER, MuSiQue),
how their evidence and provenance are structured, and how they map onto the unified
PROVE-RAG schema.

---

## HotpotQA

**Type:** multi-hop question answering
**Features:** `id`, `question`, `answer`, `type`, `level`, `supporting_facts`, `context`
**Rows:** 90,447 train / 7,405 validation

### Raw fields

- `id` — question id (str)
- `question` — the user query (str)
- `answer` — the ground-truth answer (str)
- `type` — question type, e.g. *comparison* or *bridge* (the model must use one piece of evidence to find another entity, then answer)
- `level` — question difficulty (str)
- `supporting_facts` — the gold evidence annotation; tells you exactly which sentences in the context are needed to answer the question. A dict of two parallel lists: `title` and `sent_id`.
- `context` — the retrieved documents, itself a dict with:
  - `title` — list of document titles
  - `sentences` — the sentences of each document (list of lists)

### Understanding `supporting_facts`

```json
{
  "supporting_facts": {
    "title": ["Arthur's Magazine", "First for Women"],
    "sent_id": [0, 0]
  },
  "context": {
    "title": [
      "Radio City (Indian radio station)",
      "History of Albanian football",
      "Echosmith",
      "Women's colleges in the Southern United States",
      "First Arthur County Courthouse and Jail",
      "Arthur's Magazine",
      "2014\u201315 Ukrainian Hockey Championship",
      "First for Women",
      "Freeway Complex Fire",
      "William Rast"
    ]
  }
}
```

Read it pairwise: `(title[i], sent_id[i])`. Each pair means "go to the context document
titled `title[i]`; sentence `sent_id[i]` is the gold evidence as annotated by the dataset."

### Query / input field

- `question`

### Target / output field

- `answer`

### Evidence structure

The evidence lives under `context` — document titles paired with their sentences:

```json
{
  "title": [
    "Radio City (Indian radio station)",
    "History of Albanian football",
    "Echosmith"
  ],
  "sentences": [
    ["Radio City is India's first private FM radio station and was started on 3 July 2001."]
  ]
}
```

### Provenance available

- No external provenance (no URL or date). Only internal provenance, such as the document title, can be derived.

### Useful labels

- `context`, `supporting_facts`

### Problems / limitations

- No external provenance is available, but internal provenance such as the document name is provided.
- For HotpotQA, the metadata provenance should therefore look like:

```json
{
  "provenance": {
    "dataset": "hotpotqa",
    "source_type": "wikipedia",
    "doc_title": "Arthur's Magazine",
    "sentence_index": 0,
    "url": null,
    "timestamp": null,
    "author": null,
    "publisher": "Wikipedia",
    "retrieval_time": null
  }
}
```

---

## FEVER

> **Note:** FEVER is *not* a question-answering dataset — it is a claim-verification dataset.

**Splits:** `train`, `labelled_dev`, `unlabelled_dev`, `unlabelled_test`, `paper_dev`, `paper_test`

| # | Split | Features | Rows |
|---|-------|----------|------|
| 1 | train | `id`, `label`, `claim`, `evidence_annotation_id`, `evidence_id`, `evidence_wiki_url`, `evidence_sentence_id` | 311,431 |
| 2 | labelled_dev | (same) | 37,566 |
| 3 | unlabelled_dev | (same) | 19,998 |
| 4 | unlabelled_test | (same) | 19,998 |
| 5 | paper_dev | (same) | 18,999 |
| 6 | paper_test | (same) | 18,567 |

We work with a subset of these splits, sized to align with the HotpotQA sample.

### Raw fields

- `id` — claim id
- `label` — whether the evidence supports the claim. FEVER has 3 classes: `SUPPORTS`, `NOT ENOUGH INFO`, `REFUTES`.
- `claim` — the claim being verified
- evidence fields (`evidence_wiki_url`, `evidence_sentence_id`, …) — point to where the supporting/refuting sentence lives

### Problems / limitations

- The FEVER Wikipedia corpus must be downloaded separately, since the exact sentence text is needed for training (it is not included in the raw records).

### Example

```json
{
  "id": 75397,
  "label": "SUPPORTS",
  "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
  "evidence_annotation_id": 92206,
  "evidence_id": 104971,
  "evidence_wiki_url": "Nikolaj_Coster-Waldau",
  "evidence_sentence_id": 7
}
```

How to read this: go to the evidence document titled `Nikolaj_Coster-Waldau` and take
sentence number 7. That sentence supports the claim.

### Rationale for using FEVER

Its three classes (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) let us train and evaluate the
system's ability to handle:

1. Contradicted evidence
2. Insufficient evidence
3. Claim verification
4. Sentence-level support/refute labels
5. Evidence-based abstention or verification decisions

---

## MuSiQue

**Features:** `id`, `paragraphs`, `question`, `question_decomposition`, `answer`, `answer_aliases`, `answerable`
**Rows:** 39,876 train / 4,834 validation

### Raw fields

- `id` — (str)
- `paragraphs` — the evidence provided to the model (list of dicts), each with:
  - `idx` — paragraph index (int)
  - `title` — title of the text (str)
  - `paragraph_text` — the actual evidence text
  - `is_supporting` — whether this paragraph is gold supporting evidence; if `false`, it is a distractor (bool)

### Query / input field

- `question`

### Target / output field

- `answer`

### Evidence structure

Evidence is under `paragraphs`: a list of dicts holding the text and whether it is supporting.

### Provenance available

- Little beyond the paragraph titles.

### Useful labels

- `paragraph_text`
- `is_supporting`

### Problems / limitations

- Evidence is paragraph-level, whereas HotpotQA is sentence-level — granularity must be standardized when training the evidence detector.

### Example

```json
{
  "id": "2hop__42543_20093",
  "paragraphs": [
    {
      "idx": 0,
      "title": "All Things in Time",
      "paragraph_text": "All Things in Time is an album by American R&B singer Lou Rawls, released in June 1976 on the Philadelphia International Records label. Coming after a career lull in the years immediately preceding, \"All Things in Time\" was Rawls' first album for PIR; at the time he was the first artist to sign with PIR after having already enjoyed a substantial recording career and chart success with other record labels. The album includes Rawls' most famous hit song \"You'll Never Find Another Love Like Mine\".",
      "is_supporting": false
    }
  ]
}
```

> **Granularity note:** MuSiQue is paragraph-level while HotpotQA is sentence-level, so the
> schema needs a standardized representation. Two options:
> - Convert MuSiQue to sentence level — but the sentence labels may be noisy.
> - Leave it as-is and add a schema field recording whether evidence is at sentence or paragraph level.

---

## Unified schema requirements

The schema must support:

- question-based QA
- claim verification
- answer labels
- verification labels
- evidence units
- document title / page
- sentence index
- supporting / refuting / neutral evidence
- multi-hop evidence
- insufficient evidence
- provenance metadata

### Final schema

```json
{
  "example_id": "hotpotqa_000001_gold_full",
  "dataset": "hotpotqa",
  "task_type": "multi_hop_qa",

  "input_text": "...",
  "input_type": "question",

  "target_answer": "...",
  "target_label": null,

  "evidence_state_label": "sufficient",

  "evidence_set": {
    "condition": "gold_full",
    "created_by": "dataset",
    "canonical_granularity": "sentence",
    "native_granularity": "sentence",
    "num_evidence_units": 2,
    "num_gold_evidence_units": 2,
    "num_sources": 2
  },

  "evidence_units": [
    {
      "evidence_id": "hotpotqa_000001_Arthur_Magazine_sent_0",
      "text": "...",
      "doc_title": "Arthur's Magazine",
      "source_doc_id": "hotpotqa::Arthur's_Magazine",

      "canonical_unit_type": "sentence",
      "native_unit_type": "sentence",

      "paragraph_index": null,
      "sentence_index": 0,
      "text_status": "available",

      "is_gold_evidence": true,
      "support_role": "supports",
      "native_label": "supporting_fact",
      "label_strength": "gold_sentence",
      "supervision_weight": 1.0,

      "provenance": {
        "dataset": "hotpotqa",
        "source_type": "dataset_context",
        "doc_title": "Arthur's Magazine",
        "source_doc_id": "hotpotqa::Arthur's_Magazine",
        "paragraph_index": null,
        "sentence_index": 0,
        "timestamp": null,
        "version": null,
        "provenance_granularity": "sentence"
      }
    }
  ],

  "agent_action_label": "answer"
}
```

> Graph metadata can be stored separately in another file to keep this JSON from becoming too large.

---

## Rationale for splitting the MuSiQue and HotpotQA datasets

To ensure balanced representation across datasets and evidence-state classes, we apply
dataset-specific sampling strategies during preprocessing. HotpotQA examples are fully
answerable by construction, so we sample 15,000 training and 2,500 validation examples
directly. MuSiQue, however, contains approximately equal proportions of answerable and
unanswerable examples; we therefore sample 30,000 training and 5,000 validation examples to
yield approximately 15,000 and 2,500 sufficient (answerable) instances respectively, after
filtering. For each sufficient instance, we construct one variant per degraded evidence
state — insufficient, contradicted, and superseded — using controlled evidence modification
(Section III-F), producing a balanced four-class dataset with approximately 15,000 training
examples and 2,500 validation examples per class per dataset. This design ensures that the
evidence-state detector learns evidence-quality patterns rather than dataset-specific
artifacts or class-frequency biases.

---

## Dataset statistics

### Schema conversion summary

| Dataset  | Split | Raw Examples | Evidence Units | Gold Evidence | Avg Evidence/Instance | Avg Gold/Instance |
|----------|-------|--------------|----------------|---------------|-----------------------|-------------------|
| HotpotQA | Train | 15,000       | 612,155        | 35,836        | 40.8                  | 2.4               |
| HotpotQA | Val   | 2,500        | 103,172        | 6,041         | 41.3                  | 2.4               |
| MuSiQue  | Train | 30,000       | 599,908        | 48,105        | 20.0                  | 1.6               |
| MuSiQue  | Val   | 4,834        | 96,626         | 9,133         | 20.0                  | 1.9               |
| FEVER    | Train | 30,000       | 30,000         | 25,206        | 1.0                   | 0.8               |
| FEVER    | Val   | 7,000        | 7,000          | 5,405         | 1.0                   | 0.8               |

### Evidence-state variant distribution

| Dataset      | Split     | Sufficient | Insufficient | Contradicted | Superseded |
|--------------|-----------|------------|--------------|--------------|------------|
| HotpotQA     | Train     | 15,000     | 15,000       | 15,000       | 15,000     |
| HotpotQA     | Val       | 2,500      | 2,500        | 2,500        | 2,500      |
| MuSiQue      | Train     | 15,004     | 15,004       | 15,004       | 15,004     |
| MuSiQue      | Val       | 2,417      | 2,417        | 2,417        | 2,417      |
| FEVER        | Train     | 18,408     | 18,408       | 11,730†      | 11,730†    |
| FEVER        | Val       | 2,796      | 2,796        | 1,666†       | 1,666†     |
| **Combined** | **Train** | **48,412** | **48,412**   | **41,734**   | **41,734** |
| **Combined** | **Val**   | **7,713**  | **7,713**    | **6,583**    | **6,583**  |

† FEVER contradicted and superseded variants are lower because 36% of `SUPPORTS` examples have
unresolved Wikipedia evidence text, preventing synthetic contradiction/supersession generation.

### Notes

- **HotpotQA:** all examples are answerable, so raw count equals sufficient count. Sentence-level evidence, avg 2.4 gold supporting facts per instance.
- **MuSiQue:** ~50% of examples are unanswerable; 30,000 raw examples yield ~15,004 sufficient instances. Paragraph-level evidence, avg 1.6 gold supporting paragraphs per instance.
- **FEVER:** ~61% `SUPPORTS`, ~16% `REFUTES`, ~23% `NOT ENOUGH INFO` in the sampled subset. Single sentence-level evidence per instance, resolved from the FEVER Wikipedia corpus (5.4M pages, 25.2M sentences).
- During the merge step, classes are balanced by subsampling to the smallest class count per split.
- All variant creation uses `seed=42` for reproducibility.
- **Insufficient variants:** one gold evidence unit removed per instance.
- **Contradicted variants:** one synthetic contradicting evidence unit added alongside the original gold evidence.
- **Superseded variants:** one gold unit marked as outdated (timestamp 2018), one synthetic newer version added (timestamp 2025).
