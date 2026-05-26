# Dataset Field Comparison for PROVE-RAG

## HotpotQA
Overall Structure: 

**Features:** ['id', 'question', 'answer', 'type', 'level', 'supporting_facts', 'context']

Number of Rows in:
- Training = 90447
- Validation = 7405

### Raw fields
- id: question id (str)
- question: the user query (str)
- answer: the Ground Truth Answer (str)
- type: the type of question - eg: comparison, bridge - model must use one piece of evidence to find another entity and then answer.
- level: difficulty of the question (str)
- supporting_facts: this is the gold evidence annotation - It tells you exactly which sentences in the context are needed to answer the question. (dict containing two lists - sentence id ('sent_id') and 'title'.) 
- context: list of the title of the documents that have been retrieved. (list)
- sentences: contains the sentences of the documents (list of lists)

So, to understand supporting_facts it is: 

Example:
{
  "supporting_facts": {
    "title": [
      "Arthur's Magazine",
      "First for Women"
    ],
    "sent_id": [
      0,
      0
    ]
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

So it is to be read as (title[0], sent_id[0]), (title[1], sent_id[1]). Therefore, this is saying, go to the context document with the title - 'title[i]', the sentence 'sent_id[i]' is the gold evidence as annotated by the dataset.

### Query / input field
- question

### Target / output field
- answer

### Evidence structure
Example of the context:
{
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
    ],
    "sentences": [
      [
        "Radio City is India's first private FM radio station and was started on 3 July 2001."]
        ]
}

- context - title of the documents, and their sentences

### Provenance available
- There is no provenance available - not even URL or date, therefore, 

### Useful labels
- context, supporting_facts

### Problems / limitations
- There is no external provenance available, but internal provenance such as document name is provided.
- Therefore, for HotPotQA, the metadata provenance should look something like:
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


## FEVER
NOTE: FEVER is NOT a Question-Answering dataset, it is a Claim-verification dataset.

There are quite a few splits here:

['train', 'labelled_dev', 'unlabelled_dev', 'unlabelled_test', 'paper_dev', 'paper_test']

Here is the further breakdown:
1. train: Dataset({
        features: ['id', 'label', 'claim', 'evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'],
        num_rows: 311431
    })
2. labelled_dev: Dataset({
        features: ['id', 'label', 'claim', 'evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'],
        num_rows: 37566
    })
3. unlabelled_dev: Dataset({
        features: ['id', 'label', 'claim', 'evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'],
        num_rows: 19998
    })
4. unlabelled_test: Dataset({
        features: ['id', 'label', 'claim', 'evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'],
        num_rows: 19998
    })
5. paper_dev: Dataset({
        features: ['id', 'label', 'claim', 'evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'],
        num_rows: 18999
    })
6. paper_test: Dataset({
        features: ['id', 'label', 'claim', 'evidence_annotation_id', 'evidence_id', 'evidence_wiki_url', 'evidence_sentence_id'],
        num_rows: 18567
    })

Will most probably choose a subset of this to work with HotPotQA.

### Raw fields
- id: id
- label: whether the sentence supports the claim or not
- claim: claim that is verifying the sentence against - FEVER has 3 classes: SUPPORTS, NOT ENOUGH INFO, REFUTES
- evidence related fields that tell us where to look.

### Problems / limitations
- Need to download the FEVER wiki corpus as we need the exact sentence for training.

Example:
{
  "id": 75397,
  "label": "SUPPORTS",
  "claim": "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
  "evidence_annotation_id": 92206,
  "evidence_id": 104971,
  "evidence_wiki_url": "Nikolaj_Coster-Waldau",
  "evidence_sentence_id": 7
}

How to read this: Go to the evidence page/document titled Nikolaj_Coster-Waldau, and take sentence number 7. That sentence supports the claim.

### Rationale for using FEVER:
Its purpose is to train/evaluate the system’s ability to handle:

1. Contradicted evidence
2. Insufficient evidence
3. Claim verification
4. Sentence-level support/refute labels
5. Evidence-based abstention or verification decisions

This can be done due to FEVER providing the 3 classes of 'Supports', 'Refutes' and 'Not enough Info'.

## MuSiQue
Overall Structure:
**Features:** ['id', 'paragraphs', 'question', 'question_decomposition', 'answer', 'answer_aliases', 'answerable'],

Number of rows:
- Training = 39876
- Validation = 4834

### Raw fields
- id: (str)
- paragraphs: this is the evidence that is given by the user to the model - like shot examples. (list of dictionaries)
    - id: id (int)
    - title: title of the text (str)
    - paragraph_text: actual evidence text
    - is_supporting: this tells us if the evidence provided is part of the gold supported evidence or not - if it is not then it is a distracting paragraph. (bool)

### Query / input field
- question

### Target / output field
- answer

### Evidence structure
- the evidence structure is under the key: 'paragraphs' where there are a list of dictionaries that contain the text and whether it is supporting or not.

### Provenance available
- not much just the title of the paragraphs

### Useful labels
- - paragraph_text
- is_supporting

### Problems / limitations
- this is on the paragraph-level while HotPotQA is on the sentence level
- might need to standardize while training the evidence detector


Example:

{
  "id": "2hop__42543_20093",
  "paragraphs": [
    {
      "idx": 0,
      "title": "All Things in Time",
      "paragraph_text": "All Things in Time is an album by American R&B singer Lou Rawls, released in June 1976 on the Philadelphia International Records label. Coming after a career lull in the years immediately preceding, \"All Things in Time\" was Rawls' first album for PIR; at the time he was the first artist to sign with PIR after having already enjoyed a substantial recording career and chart success with other record labels. The album includes Rawls' most famous hit song \"You'll Never Find Another Love Like Mine\".",
      "is_supporting": false
    },]
}


NOTE: The MuSiQue Dataset is in the paragraph-leve, while the HotPotQA is on the sentence-level. Therefore, our schema needs to include something that is standardized. 
- Either we can convert MusiQue to the sentence level - but the sentence labels might be noisy.
- Or, we leave it as it is and in the schema, we add a field that tells us whether it is in the evidence level or in the paragraph level.


## Unified schema requirements

The schema must support:
- question-based QA
- claim verification
- answer labels
- verification labels
- evidence units
- document title/page
- sentence index
- supporting/refuting/neutral evidence
- multi-hop evidence
- insufficient evidence
- provenance metadata

Final Schema:

{"example_id": "hotpotqa_000001_gold_full",
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


we could store the graph metadata separately in another file to prevent the JSON from being too messy.

## Rationale for Splitting the Musique and HotpotQA datasets
To ensure balanced representation across datasets and evidence-state classes, we apply dataset-specific sampling strategies during preprocessing. HotpotQA examples are fully answerable by construction, so we sample 15,000 training and 2,500 validation examples directly. MuSiQue, however, contains approximately equal proportions of answerable and unanswerable examples; we therefore sample 30,000 training and 5,000 validation examples to yield approximately 15,000 and 2,500 sufficient (answerable) instances respectively, after filtering. For each sufficient instance, we construct one variant per degraded evidence state — insufficient, contradicted, and superseded — using controlled evidence modification (Section III-F), producing a balanced four-class dataset with approximately 15,000 training examples and 2,500 validation examples per class per dataset. This design ensures that the evidence-state detector learns evidence-quality patterns rather than dataset-specific artifacts or class-frequency biases.

## Dataset Statistics

### Schema Conversion Summary

| Dataset | Split | Raw Examples | Evidence Units | Gold Evidence | Avg Evidence/Instance | Avg Gold/Instance |
|---------|-------|-------------|----------------|---------------|----------------------|-------------------|
| HotpotQA | Train | 15,000 | 612,155 | 35,836 | 40.8 | 2.4 |
| HotpotQA | Val | 2,500 | 103,172 | 6,041 | 41.3 | 2.4 |
| MuSiQue | Train | 30,000 | 599,908 | 48,105 | 20.0 | 1.6 |
| MuSiQue | Val | 4,834 | 96,626 | 9,133 | 20.0 | 1.9 |
| FEVER | Train | 30,000 | 30,000 | 25,206 | 1.0 | 0.8 |
| FEVER | Val | 7,000 | 7,000 | 5,405 | 1.0 | 0.8 |

### Evidence-State Variant Distribution

| Dataset | Split | Sufficient | Insufficient | Contradicted | Superseded |
|---------|-------|-----------|--------------|--------------|------------|
| HotpotQA | Train | 15,000 | 15,000 | 15,000 | 15,000 |
| HotpotQA | Val | 2,500 | 2,500 | 2,500 | 2,500 |
| MuSiQue | Train | 15,004 | 15,004 | 15,004 | 15,004 |
| MuSiQue | Val | 2,417 | 2,417 | 2,417 | 2,417 |
| FEVER | Train | 18,408 | 18,408 | 11,730† | 11,730† |
| FEVER | Val | 2,796 | 2,796 | 1,666† | 1,666† |
| **Combined** | **Train** | **48,412** | **48,412** | **41,734** | **41,734** |
| **Combined** | **Val** | **7,713** | **7,713** | **6,583** | **6,583** |

† FEVER contradicted and superseded variants are lower because 36% of SUPPORTS examples have unresolved Wikipedia evidence text, preventing synthetic contradiction/supersession generation.

### Notes
- HotpotQA: All examples are answerable; raw count equals sufficient count. Sentence-level evidence with avg 2.4 gold supporting facts per instance.
- MuSiQue: ~50% of examples are unanswerable; 30,000 raw examples yield ~15,004 sufficient instances. Paragraph-level evidence with avg 1.6 gold supporting paragraphs per instance.
- FEVER: ~61% SUPPORTS, ~16% REFUTES, ~23% NOT ENOUGH INFO in the sampled subset. Evidence is single sentence-level per instance, resolved from the FEVER Wikipedia corpus (5.4M pages, 25.2M sentences).
- During the merge step, classes will be balanced by subsampling to the smallest class count per split.
- All variant creation uses seed=42 for reproducibility.
- Insufficient variants: one gold evidence unit removed per instance.
- Contradicted variants: one synthetic contradicting evidence unit added alongside original gold evidence.
- Superseded variants: one gold unit marked as outdated (timestamp 2018), one synthetic newer version added (timestamp 2025).