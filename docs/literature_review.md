# Literature Review: Multi-Stage Data Selection for Medical Foundation Models

**Project:** Multi-Stage and Multi-Target Data Selection Harness for Medical Foundation Models
**Status:** Working document. Sections 3 (CPT) and 7 (data selection) are the seed; dataset
sections are for their owners to extend. See [contributing.md](contributing.md) for who owns what.

---

## Contents

1. [Scope and how to read this](#1-scope-and-how-to-read-this)
2. [The multi-stage training pipeline](#2-the-multi-stage-training-pipeline)
3. [Stage 1: Continual pretraining](#3-stage-1-continual-pretraining)
4. [Stage 2: Supervised fine-tuning](#4-stage-2-supervised-fine-tuning)
5. [Stage 3: Preference optimisation](#5-stage-3-preference-optimisation)
6. [The datasets](#6-the-datasets)
7. [Data selection methods](#7-data-selection-methods)
8. [Gap analysis](#8-gap-analysis)
9. [Implications for our harness](#9-implications-for-our-harness)
10. [Open questions](#10-open-questions)
11. [References](#11-references)

---

## 1. Scope and how to read this

This review supports the proposal's month 1-2 deliverable ("architecture document"). It covers
four things, in descending order of depth:

- **Continual pretraining (CPT)**: the stage the project is starting with, and the one where
  corpus selection plausibly matters most.
- **Data selection**: the methods this project intends to extend to a multi-stage setting.
- **The four datasets** we have committed to: MedQA, MedMCQA, PubMedQA, and the PubMed corpus.
- **SFT and preference optimisation**: enough to know what the later stages will need.

Two conventions. Claims that are load-bearing for a design decision carry a citation. Where a
number was taken from a secondary source rather than read out of the primary paper, it is marked
*(secondary)*. Those should be checked against the paper before appearing in a manuscript.

A caution that shapes everything below: **the field's headline claim is contested.** Section 3.5
covers evidence that medical domain adaptation frequently fails to beat the base model under fair
comparison. Read that before designing experiments, not after.

---

## 2. The multi-stage training pipeline

Modern medical foundation models are built in stages, and the stages differ in kind, not just in
hyperparameters:

| Stage | Objective | Data | What it is supposed to add |
|---|---|---|---|
| Continual pretraining (CPT) | Next-token prediction | Raw biomedical text | Domain knowledge, vocabulary, long-tail facts |
| Supervised fine-tuning (SFT) | Next-token on responses only | Instruction-response pairs | Task format, instruction following |
| Preference optimisation | DPO / RLHF objective | Preference pairs | Helpfulness, safety, calibration |

The important structural point for this project is that **each stage consumes a different kind of
data and is credited with a different capability**. A corpus that is excellent for CPT (broad,
diverse, long-tail) is not an instruction dataset, and an instruction dataset that teaches format
does not teach facts. It follows that a single data-selection policy optimised once cannot be
right for all three, which is precisely the proposal's thesis.

What the literature actually does, though, is optimise one stage at a time and evaluate on one
metric, usually downstream accuracy on a single benchmark. Section 8 returns to this.

---

## 3. Stage 1: Continual pretraining

### 3.1 The founding result

The canonical reference is Gururangan et al. (2020), *Don't Stop Pretraining* [G20]. It
established two now-standard moves:

- **DAPT** (domain-adaptive pretraining): continue pretraining on a large unlabelled corpus from
  the target domain.
- **TAPT** (task-adaptive pretraining): continue pretraining on the (unlabelled) text of the task
  itself, which is much smaller but far better targeted.

The paper reported gains across four domains including biomedical, and found DAPT and TAPT to be
complementary. Crucially for us, it also showed that *selecting* task-relevant text from a large
corpus (a cheap embedding-based retrieval) recovered much of TAPT's benefit. That is a data
selection result, published in 2020, and it is the ancestor of the work in Section 7.

### 3.2 The recipe question: how to continue pretraining without destroying the model

The practical difficulty in CPT is that naively resuming training on a new distribution degrades
prior capability: catastrophic forgetting. Ibrahim et al. (2024) [I24] give the clearest recipe
result: a combination of

1. **learning-rate re-warming**,
2. **re-decaying** the schedule over the new run, and
3. **replay** of a fraction of the original pretraining distribution

is sufficient to match full retraining from scratch on the union of data, at a fraction of the
compute. This matters to us in a specific way: **replay ratio is a data-selection decision.**
Deciding what fraction of general-domain text to mix into a medical CPT run, and which
general-domain text, is exactly the kind of thing our harness should be able to express.

Related practical findings worth carrying into experiments:

- The *ordering* of a CPT corpus is not neutral; curriculum-like effects exist, though the
  evidence is weaker than for replay.
- Sequence packing (concatenating documents into fixed-length blocks) is standard for efficiency.
  It introduces a subtlety we implement explicitly: without an EOS separator between documents,
  the model learns to continue one abstract into an unrelated one.

### 3.3 Medical CPT models

The main open-weight lineage, roughly chronologically:

- **BioMedLM** [B24]: 2.7B, trained from scratch on PubMed abstracts and full text rather than
  adapted. Useful as the "pure biomedical" reference point.
- **PMC-LLaMA** [W23]: LLaMA continually pretrained on biomedical papers and textbooks, then
  medically instruction-tuned. Reported strong benchmark results at much smaller scale than the
  general models of its day; explicitly frames itself as data-centric knowledge injection.
- **MEDITRON** [C23]: 7B and 70B, CPT on a curated corpus (PubMed Central full text, abstracts,
  clinical guidelines). Notable for treating corpus construction as the contribution.
- **BioMistral** [L24]: Mistral 7B further pretrained on PubMed Central open access.
- **Me-LLaMA** [X24]: LLaMA 2 with continued pretraining plus instruction tuning on a mixed
  biomedical and clinical corpus.
- **MedGemma** [M25]: Gemma 3 derivative (4B multimodal, 27B text and multimodal), pretrained on
  medical text, medical QA, and FHIR-based EHR data, with a SigLIP encoder pretrained on
  de-identified medical images for the multimodal variants. Reported to beat its Gemma base on
  all tested text-only health benchmarks.

For our purposes MedGemma is best used as a **reference ceiling**, not as a starting point: it is
already medically adapted, so further CPT on it would make any improvement impossible to
attribute to our selection method. This is why the harness targets Gemma 3 base.

### 3.4 What CPT is supposed to buy, and the measurement problem

The claim implicit in all of the above is that reading more biomedical text makes a model better
at medical tasks. Measuring that is harder than it sounds:

- **Benchmark contamination.** PubMedQA is derived from PubMed abstracts. Continually pretraining
  on PubMed and then evaluating on PubMedQA risks training on the evaluation contexts. This is a
  live concern for our own experiments and must be handled by PMID-level exclusion.
- **Prompt sensitivity.** MCQ accuracy for base models varies enormously with prompt format and
  scoring convention. A "gain" can be an artefact of the adapted model happening to suit the
  prompt better.
- **Capability collapse.** Aggregate accuracy can rise while performance on rare or
  underrepresented categories falls. The proposal's capability-retention framing exists precisely
  because a single number hides this.

### 3.5 The contrarian result, and why it matters most

Jeong et al. (2024), *The Limited Impact of Medical Adaptation of Large Language and
Vision-Language Models* [J24], evaluated ten medical LLMs and two medical VLMs against their own
base models, optimising prompts **separately for each model** and accounting for statistical
uncertainty. Their finding: on clinical-note QA with 3-shot prompting, the medical models beat
their bases in only 26.7% of comparisons, tied in 16.7%, and *lost* in 56.7% *(secondary: figures
taken from the paper abstract via its arXiv page)*.

Their diagnosis of prior work is methodological: earlier studies did not consistently compare
against the correct base model, and did not tune prompts per model, so an apparent domain gain
could be a prompt-fit artefact.

This is the single most important paper in this review for how we run experiments. Three
consequences we should adopt as standing practice:

1. **Always evaluate the base model under the identical harness.** A CPT result is a *delta*, and
   a delta needs a control run, not a published number.
2. **Report the scoring convention.** Our evaluator supports letter-scoring and text-scoring
   precisely because base models are badly served by letter-scoring; reporting one without the
   other invites the artefact Jeong et al. describe.
3. **Report uncertainty.** MedQA's test split is 1,273 items; a 1.5-point difference there is
   well inside noise. Confidence intervals are not optional.

Read charitably, this paper does not say domain adaptation cannot work. It says the field has not
yet demonstrated that it does, under fair comparison. That is a *better* motivation for this
project than the optimistic framing: if generic medical CPT gives little, then showing that
*selected* medical CPT gives more is a genuinely open and worthwhile result.

---

## 4. Stage 2: Supervised fine-tuning

### 4.1 What SFT does

SFT trains on instruction-response pairs with the loss masked to the response span, so the model
is not rewarded for reproducing the prompt. Its role is format and instruction-following, not
primarily knowledge acquisition.

For our three QA datasets, SFT is where they naturally live: MedQA and MedMCQA supply
question-with-options → answer, and PubMedQA supplies question-with-abstract → yes/no/maybe plus a
long-form rationale. MedMCQA additionally ships explanations (`exp`) for a subset of rows, which
makes chain-of-thought-style SFT possible without generating rationales synthetically.

### 4.2 The combined-objective question

A design question raised in this project's own planning, and worth treating as a research
question rather than an implementation detail: should the instruction stage be trained with a
**combined objective** (next-token loss on raw biomedical text *alongside* instruction loss
on QA pairs) instead of running CPT and SFT strictly in sequence?

The motivating intuition is that sequential SFT erodes the domain knowledge CPT just installed,
and that mixing keeps it. This is essentially the replay idea from Section 3.2 applied across the
CPT/SFT boundary rather than within CPT. There is supporting work on balancing continued
pretraining against instruction tuning, and the general replay evidence is strong, but we should
treat the specific medical claim as untested and design an ablation for it. It is recorded in the
codebase at `src/medsel/stages/sft.py` so it does not get lost.

### 4.3 SFT data selection

This is where most of the recent selection literature actually operates (Section 7). LESS [X24a]
and 3DS [D25] are both SFT-stage methods. Their headline results are strong: LESS reports that
5% of the data, selected well, can beat the full dataset, which is the strongest available
evidence that selection is worth doing at all.

---

## 5. Stage 3: Preference optimisation

Covered briefly, because it is the furthest away and because we have a data problem before we
have a method problem.

**Methods.** DPO is the practical choice at our scale: it optimises directly against preference
pairs and needs no separate reward model or online sampling loop, unlike PPO-style RLHF. TRL
provides a `DPOTrainer` that works against the same model and tokenizer objects the earlier stages
use.

**The blocker is data, not code.** The proposal says preference data "if available", and for the
public datasets in scope, it is not. Options, in rough order of cost:

1. Construct pairs from PubMedQA: the expert `long_answer` as chosen, a model-generated or
   perturbed answer as rejected. Cheap, but the "rejected" side is artificial and the resulting
   preference signal may be about fluency rather than clinical correctness.
2. Use MedMCQA explanations similarly, with the correct-option explanation as chosen.
3. Adopt an existing open medical preference set, if one of adequate quality exists. This needs a
   survey we have not yet done.

Safety and hallucination are the capabilities this stage is normally credited with, and they are
also the ones our current benchmarks (three MCQ datasets) cannot measure at all. That gap should
be stated plainly in any write-up.

---

## 6. The datasets

Full field-level schemas, split semantics, and the traps we hit are in
[datasets.md](datasets.md). This section covers provenance and what each dataset is good for.

### 6.1 MedQA *(owner: Debmalya)*

USMLE-style clinical vignettes with four options [J21]. Questions are long (a full patient
presentation) and require multi-step clinical reasoning rather than fact recall, which makes it
the hardest of the three for small models.

Widely reported figures: GPT-4 around 86% and Medprompt-style prompting above 90%; the strongest
open models via prompt engineering around 72.6% (Yi-34B, OpenMedLM) *(secondary)*. A 1B base model
should be expected near chance (25%) before any adaptation, which is worth stating so that a
smoke-test number is not mistaken for a failure.

Caveats: the Hub mirror we use ships an empty `subject_name` column, so MedQA offers **no
per-subject breakdown**. It cannot contribute to per-capability analysis, only to aggregate
accuracy.

### 6.2 MedMCQA *(owner: Animesh)*

~194k questions from Indian medical entrance exams (AIIMS, NEET-PG), spanning 21 subjects and
~2.4k topics [P22]. Its size makes it the only one of the three usable as a substantial SFT
source, and its `subject_name` / `topic_name` labels make it **the only per-capability signal any
of our datasets ships**. That is why the evaluator's subject breakdown keys on it.

The important structural fact: the published **`test` split withholds gold answers** (`cop = -1`).
Every comparable number in the literature is measured on `validation` (the NEET-PG set), and so is
ours. Reported open-model SOTA is around 68.3% *(secondary)*.

Quality caveat worth documenting: MedMCQA is known to contain some noisy items (typos, ambiguous
phrasing, and occasional disputed keys), a consequence of its scraped origin. This is relevant to
us twice over: it is an argument *for* data selection at the SFT stage, and it is a reason not to
over-interpret small accuracy differences.

### 6.3 PubMedQA *(owner: Arkajyoti)*

Research questions derived from PubMed article titles, answerable yes/no/maybe from the abstract
[J19]. Three configs:

- `pqa_labeled`: 1,000 expert-annotated items, the evaluation set
- `pqa_artificial`: ~211k auto-labelled items, usable for SFT
- `pqa_unlabeled`: ~61k with long answers but **no `final_decision` column at all**

Two things make PubMedQA different from the other two: it is a 3-way decision rather than a 4-way
MCQ, and it supplies *context* (the abstract), so it measures reading comprehension over a given
passage rather than parametric knowledge.

**Comparability caveat.** The canonical 500-question test set is defined by a ground-truth file in
the authors' repository, which the Hub mirror does not carry. `pqa_labeled` arrives as one
undivided 1,000-row split. Our numbers are therefore internally consistent but should not be
quoted against published PubMedQA figures without adopting the official split.

**Contamination caveat.** PubMedQA is built from PubMed abstracts, and our CPT corpus is PubMed
abstracts. Any CPT-then-evaluate-on-PubMedQA experiment must exclude the PubMedQA PMIDs from the
CPT corpus, or the result is meaningless. This is a concrete task, not a caveat to note and
forget.

### 6.4 The PubMed CPT corpus *(owner: Srinjoy)*

We use `MedRAG/pubmed` [X24b], the corpus assembled for the MedRAG benchmark: 23.9M PubMed
articles that have both a title and an abstract, pre-chunked into 1,166 JSONL shards.

Measured properties (verified directly against the Hub, not quoted):

- 1,166 shards, **70 GB total**, averaging ~60 MB per shard
- ~296 tokens per snippet on average
- fields: `id`, `title`, `content` (abstract), `contents` (= `title + " " + content`), `PMID`

Choosing this over the NCBI baseline FTP trades some control for a lot of avoided work: the XML
is already parsed, filtered to articles with usable abstracts, and chunked. The costs of that
choice, which we should state in any write-up: it is a snapshot (the `pubmed23` naming indicates a
2023 baseline), it is abstracts-only rather than full text, and its inclusion filter is MedRAG's
rather than ours.

If full text becomes necessary (plausible for long-context or reasoning-oriented CPT), PubMed
Central Open Access is the successor corpus, and MEDITRON's corpus construction [C23] is the
reference for how to do it.

---

## 7. Data selection methods

The organising question: given a large pool of candidate training data and a budget, which subset
should we train on? The literature splits by *what signal* is used to answer it.

### 7.1 Influence-based

The idea: estimate how much each training example changes performance on a target set, and keep
the most helpful.

- **Influence functions** [K17]. The foundational work, approximating the effect of upweighting a
  training point via the inverse Hessian. Exact computation is intractable at LLM scale; everything
  since is an approximation to this.
- **TRAK** [P23]. Makes influence attribution tractable at scale by linearising the model and
  using random projections, with ensembling to reduce variance. Practical for attributing model
  behaviour, but still expensive.
- **LESS** [X24a]. The most directly relevant to us. It adapts influence formulations to work with
  **Adam** (not plain SGD) and with variable-length instruction data, builds a reusable low-rank
  **gradient datastore**, and then selects examples by gradient similarity to a handful of few-shot
  examples that embody a target capability. Two results matter for this project:
  - training on a LESS-selected **5%** can outperform training on the full dataset;
  - selections **transfer**: a small model can pick data for a larger one, and across families.

  That transfer property is what makes influence-based selection affordable for a student project:
  we can score with a 1B model and train something bigger.

### 7.2 Optimisation-based subset selection

- **GLISTER** [K21a]. Frames subset selection as bilevel optimisation against a held-out
  validation set, with a greedy approximation.
- **GradMatch** [K21b]. Selects a subset whose *summed gradient* approximates the full-data
  gradient, via orthogonal matching pursuit.

Both predate LLM-scale instruction tuning and were demonstrated on smaller models, but the
gradient-matching objective is conceptually clean and is a reasonable thing to revisit at our
scale.

### 7.3 Distribution matching and heuristic quality

- **DSIR**: importance resampling to match a target distribution in a hashed n-gram feature
  space. Very cheap, and the natural baseline for corpus-scale (CPT) selection where per-example
  gradients are unaffordable.
- **DataComp** [G23] and **DataComp-LM** [L24]. Rather than proposing a method, these fix the
  model and training budget and make *the dataset* the submission. The lesson we should copy is
  methodological: a selection result is only meaningful against a fixed training budget and a
  fixed evaluation suite. Our `cpt_baseline.yaml` exists for this reason.
- **Quality classifiers.** The FineWeb-Edu / phi lineage trains a lightweight classifier on
  model-labelled "educational value" and filters the corpus with it. Cheap at corpus scale and a
  strong baseline for CPT selection.
- **Deduplication.** Exact and near-duplicate removal is consistently among the highest
  value-per-compute interventions in pretraining-data work. For PubMed specifically, duplicate
  deposits, errata, and near-identical abstracts are real.

### 7.4 Medical-domain selection

**3DS** [D25] is the closest published work to our project and should be treated as the primary
baseline to beat. It is a two-stage method for medical domain adaptation:

1. **Prompt-driven data selection** filters noise by checking alignment with what the model
   already knows.
2. **Decomposed difficulty-based selection** scores each example on three metrics (*instruction
   understanding*, *response confidence*, and *response correctness*) with attention-based
   importance weighting for calibration.

Reported to beat prior methods by up to 2.97% accuracy in healthcare, with validation in law and
general domains *(secondary)*. Code and data are open.

Note what 3DS is and is not: it is **SFT-stage**, **single-stage**, and optimises **accuracy**. It
decomposes *difficulty* into three axes, which is genuinely multi-signal, but it does not
decompose the *objective* into multiple capabilities, and it says nothing about CPT. That is
exactly the gap the proposal targets.

### 7.5 Summary table

| Method | Stage | Signal | Cost | Multi-target? |
|---|---|---|---|---|
| Influence functions [K17] | any | Hessian-approx influence | intractable at scale | no |
| TRAK [P23] | any | linearised attribution | high | no |
| LESS [X24a] | SFT | Adam-aware gradient similarity | moderate, amortised datastore | no (single target set) |
| GLISTER [K21a] | SFT | bilevel validation loss | moderate | no |
| GradMatch [K21b] | SFT | gradient matching | moderate | no |
| DSIR | pretrain | n-gram distribution match | very low | no |
| Quality classifier | pretrain | learned quality score | low | no |
| 3DS [D25] | SFT (medical) | decomposed difficulty | moderate | partially (3 difficulty axes) |
| **This project** | **CPT + SFT + DPO** | **multiple, stage-specific** | **TBD** | **yes, by construction** |

---

## 8. Gap analysis

Reading across Sections 3-7, four gaps are consistent:

**Gap 1: Single-stage scope.** Nearly all selection work targets SFT. LESS, GLISTER, GradMatch
and 3DS are all instruction-tuning methods. CPT-stage selection is dominated by cheap heuristics
(dedup, quality classifiers, distribution matching) because per-example gradient methods do not
scale to 23.9M documents. Nobody has asked whether the *right* CPT subset depends on what the SFT
stage will subsequently do.

**Gap 2: Single-objective optimisation.** Selection is almost universally optimised against
aggregate downstream accuracy on one benchmark. The proposal's list of clinical capabilities
(diagnosis, summarisation, information extraction, ICD coding, QA, temporal reasoning) is not
represented in any selection objective we found. Aggregate accuracy can improve while a
capability collapses, and no current method would notice.

**Gap 3: Long-tail and rare-event coverage is not a constraint.** Selection methods rank and
truncate. Anything scoring low is dropped, and rare diseases and underrepresented populations
score low almost by definition, because they are rare. Making coverage a *constraint* rather than
something the ranking might incidentally preserve is a genuine methodological difference, and it
is why the harness separates scoring from selection and gives `select_stratified` a
`min_per_group` floor.

**Gap 4: The baseline may be weaker than reported.** Per Jeong et al. [J24], the premise that
medical CPT reliably helps is not well established. Rather than undermining the project, this
sharpens it: the interesting claim is not "CPT helps" but "*selected* CPT helps where unselected
CPT does not", and that is a comparison nobody has run.

A fifth, more practical gap: **auditability**. The proposal promises interpretable data-valuation
scores and explanations for why a sample was kept or discarded. Almost nothing in Section 7
produces a human-readable justification; influence scores are numbers without narratives. In a
clinical setting that is a real deficiency and a defensible contribution on its own.

---

## 9. Implications for our harness

Design decisions in this repository that follow directly from the review:

| Decision | Source |
|---|---|
| CPT targets **Gemma 3 base**, not MedGemma | §3.3: adapting an already-adapted model makes gains unattributable |
| An **unselected CPT baseline** config ships alongside any selection experiment | §7.3 DataComp; §3.5 Jeong |
| Evaluator supports **letter and text scoring**, reports both acc and acc_norm | §3.5: prompt/scoring artefacts |
| Evaluator reports **per-subject accuracy** | §8 Gap 2; MedMCQA is the only per-capability signal (§6.2) |
| Scoring and selection are **separate modules** | §8 Gap 3: coverage is a selection constraint, not a score |
| `select_stratified(min_per_group=…)` | §8 Gap 3: rare groups must survive aggressive budgets |
| Baseline scorers are **random and length** | §7: a method that cannot beat random has shown nothing |
| Prompt templates are **shared and versioned** | §3.4: otherwise data effects and format effects confound |
| **PMID-level exclusion** required before PubMedQA evaluation | §6.3 contamination |
| `dedup` on by default in the PubMed loader | §7.3: dedup is high value per unit compute |

Concrete near-term work items this review generates, beyond the proposal's plan:

1. **Build the PMID exclusion list** from PubMedQA `pqa_labeled` (and `pqa_artificial` if used for
   SFT) and wire it into the PubMed loader as a filter. Until this exists, no PubMedQA number from
   a CPT'd model is trustworthy.
2. **Run the base-model control** for every benchmark before any CPT run, and store it as a
   committed reference result.
3. **Add bootstrap confidence intervals** to `EvalReport`. MedQA's 1,273-item test set cannot
   support the precision people routinely claim on it.
4. **Survey open medical preference datasets** to unblock Stage 3, or decide explicitly to
   construct pairs from PubMedQA.

---

## 10. Open questions

1. Can gradient-based influence (LESS-style) be made affordable at CPT scale, or must CPT
   selection stay in the cheap-heuristic regime? The transfer result in [X24a], where small models
   pick data for large ones, is the most promising lead.
2. Is CPT selection *conditionally* optimal? That is, does the best CPT subset change depending
   on the SFT data that follows? If yes, that alone justifies the multi-stage framing.
3. Does the combined CPT+instruction objective (§4.2) actually retain domain knowledge better than
   sequential CPT→SFT?
4. What is the right unit of selection for CPT: the document, the packed block, or a topic
   cluster? Our packing implementation currently makes the document the unit.
5. How should capability coverage be *measured* when we only have MedMCQA subject labels and
   PubMedQA MeSH terms to work with? Everything else in the proposal's capability list
   (diagnosis, summarisation, ICD coding, temporal reasoning) has no benchmark in our current set.

Question 5 is the most uncomfortable one. The proposal names six clinical capabilities; our three
datasets measure roughly one and a half of them. Either the evaluation suite grows, or the
capability claims narrow. This should be resolved before the month 9-10 validation phase, not
during it.

---

## 11. References

Keyed to `references.bib`.

- **[B24]** Bolton et al. *BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text.* 2024.
- **[C23]** Chen et al. *MEDITRON-70B: Scaling Medical Pretraining for Large Language Models.* 2023.
- **[D25]** Ding et al. *3DS: Medical Domain Adaptation of LLMs via Decomposed Difficulty-based Data Selection.* EMNLP 2025. arXiv:2410.10901.
- **[G20]** Gururangan et al. *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL 2020.
- **[G23]** Gadre et al. *DataComp: In Search of the Next Generation of Multimodal Datasets.* NeurIPS 2023.
- **[I24]** Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR 2024. arXiv:2403.08763.
- **[J19]** Jin et al. *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP 2019.
- **[J21]** Jin et al. *What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams (MedQA).* 2021.
- **[J24]** Jeong et al. *The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models.* EMNLP 2024. arXiv:2411.08870.
- **[K17]** Koh & Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017.
- **[K21a]** Killamsetty et al. *GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning.* AAAI 2021.
- **[K21b]** Killamsetty et al. *GRAD-MATCH: Gradient Matching based Data Subset Selection.* ICML 2021.
- **[L24]** Li et al. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.* 2024. arXiv:2406.11794.
- **[L24b]** Labrak et al. *BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains.* 2024.
- **[M25]** MedGemma Team, Google. *MedGemma Technical Report.* 2025. arXiv:2507.05201.
- **[P22]** Pal et al. *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering.* CHIL 2022.
- **[P23]** Park et al. *TRAK: Attributing Model Behavior at Scale.* ICML 2023.
- **[W23]** Wu et al. *PMC-LLaMA: Towards Building Open-source Language Models for Medicine.* 2023.
- **[X24a]** Xia et al. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML 2024. arXiv:2402.04333.
- **[X24b]** Xiong et al. *Benchmarking Retrieval-Augmented Generation for Medicine.* 2024. arXiv:2402.13178. (Source of the `MedRAG/pubmed` corpus.)
- **[X24c]** Xie et al. *Me-LLaMA: Foundation Large Language Models for Medical Applications.* 2024.
