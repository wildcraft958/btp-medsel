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

### 3.6 The 2026 evidence: still contested, in both directions

A deep research pass in July 2026 (25 claims extracted from 2025-2026 sources, each put through a
3-vote adversarial verification, 16 survived) found the contrarian result neither settled nor
overturned. Both directions strengthened.

**Extending the contrarian finding.** LiveMedBench [Yn26], a contamination-free suite refreshed
weekly from real clinical cases, reports MedGemma 27B scoring 5.9% against 6.4% for the
general-purpose Gemini 2.5 Flash, a direct 2026 extension of Jeong et al.'s result to a newer
medical model on a newer benchmark. Dorfner et al., peer-reviewed in JAMIA [Do25] (arXiv v1
predates this window but the peer-reviewed version is 2025), independently find biomedical
fine-tuning adds little on unseen clinical data under a fair comparison. MedCheck [Ch26] audited 56
medical LLM benchmarks against 46 lifecycle criteria and found 49 of 56 do not address
contamination at all, and names MedQA and MedMCQA explicitly as paradigm cases of
convenience-driven benchmark design, exam questions standing in for clinical data.

**Complicating it.** Medmarks [Wa26], a 30-benchmark open suite, reports medically adapted models
beating their own bases with mean win-rate gains of 0.32 to 0.55 depending on model family,
disconfirming evidence against the contrarian framing on this particular suite. Domingo-Aldama et
al. [Dm26] find medical adaptation pays off specifically for under-served languages, a nuance
rather than a reversal. Read together with Jeong et al., the honest summary is: fair comparison
still finds medical adaptation an inconsistent win, model-family and benchmark dependent, not a
settled loss. The project's reframed question, does *selected* CPT help where unselected CPT does
not, remains open and is not weakened by either direction.

**A benchmark-validity complication that cuts across both directions.** MedCheck [Ch26] also
reports that 34% of the 56 audited benchmarks evaluate only a single dimension such as accuracy,
and 50% align with no formal medical standard such as ICD or SNOMED CT. Any single number from
MedQA, MedMCQA or PubMedQA, in either direction, should be read against that backdrop.

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

UltraMedical [Zg24] is worth naming as context rather than as a method: a large curated biomedical
instruction corpus (NeurIPS 2024 Datasets and Benchmarks Track, Spotlight) built by synthesis and
preference filtering rather than selection from a fixed pool. It shows the field's SFT-stage
answer to data scarcity is generation, not only selection, an alternative worth weighing against
selecting from the three QA sets already in scope.

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
3. Adopt an existing open medical preference set, if one of adequate quality exists. The survey
   below answers this: no medical-specific set was found, but general open preference corpora are
   concrete and usable now.

**The survey.** Djuhera et al. [Dj26] audit five open, non-credentialed DPO corpora: TuluDPO
(around 280k pairs), ORPO (`mlabonne/orpo-dpo-mix-40k`, 40k pairs), UltraFeedback, HelpSteer, and
Code-Preference-Pairs. Two findings matter directly for option 3:

- **Curated beats largest.** UltraMix, a 190k-pair mixture assembled selectively across all five
  sources, is 30% smaller than the strongest individual source (TuluDPO) yet scores higher across
  14 evaluation tasks. This is itself a preference-stage selection result: curating beats using the
  single best full dataset, which is the same shape of claim the proposal makes for CPT and SFT.
- **These corpora are noisy at the pair level.** Only 70 to 80% of pairs across TuluDPO, ORPO,
  UltraFeedback and HelpSteer have a reward-model preference ordering that agrees with the labelled
  chosen answer, meaning roughly 20 to 30% of pairs may carry a questionable or inverted label.
  Code-Preference-Pairs is the exception, with better label agreement. Any DPO run built on these
  sources inherits this noise floor and should not treat the labelled preference as ground truth
  without a sanity check.

None of the five is medical, so option 3 does not remove the need for options 1 and 2; it gives a
general-domain preference set that could be mixed with medically-constructed pairs, the same
replay logic as Section 3.2 applied to Stage 3.

Safety and hallucination are the capabilities this stage is normally credited with, and they are
also the ones our current benchmarks (three MCQ datasets) cannot measure at all. Section 6.5 lists
concrete open benchmarks that do measure some of this. That gap should be stated plainly in any
write-up.

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

### 6.5 Beyond MCQ: benchmarks for the other capabilities

Open Question 5 (Section 10) names the project's admitted weakness: the proposal claims six
clinical capabilities, and MedQA, MedMCQA and PubMedQA measure roughly one and a half. A 2026
research pass looked specifically for open, non-credentialed benchmarks covering the rest, since
this project cannot use MIMIC-IV-style credentialed data. Four are concrete and usable now:

| Benchmark | Capability covered | Access | Headroom in 2026 |
|---|---|---|---|
| HealthBench [Op25] | Open-ended clinical response quality, safety, communication | CC BY-4.0, github.com/openai/simple-evals | Main split moved 16% to 60% in two years; HealthBench Hard still sits at 32% for the best model at release, so it has not saturated |
| BRIDGE [Wu25] | Information extraction, ICD-10 coding, summarisation, NLI, classification | Open, 87 tasks over 59 real clinical text datasets | ICD-10 coding accuracy around 15%, generation tasks around 20%, large headroom |
| ACI-Bench [Yi23] | Clinical summarisation (visit-note generation from dialogue) | CC BY 4.0, no credentialing, Figshare DOI 10.6084/m9.figshare.22494601 | Only 207 dialogue-note pairs total (40 per test split), too small to power a regression benchmark; usable as a qualitative probe only |
| Medmarks [Wa26] | QA, information extraction, medical calculation, open-ended reasoning | Open, 30 benchmarks, includes an RL-trainable split (Medmarks-T) | Reports the MultiMedQA suite (MedQA, MedMCQA, PubMedQA's core) as mostly saturated, which independently confirms why this section exists |

Two more are worth knowing about without adopting outright: LiveMedBench [Yn26] refreshes weekly
from real clinical cases specifically to stay contamination-free, at the cost of being a moving
target rather than a fixed benchmark a thesis can cite a stable number against; MedXpertQA [Zu25]
is a harder MCQ-style benchmark (expert-level reasoning) that would raise the difficulty ceiling
of the existing MCQ evaluation without adding a new capability.

The practical implication: capability coverage no longer needs a custom benchmark built from
scratch to grow past MCQ. The gap is integration effort (a fourth loader shape, since none of
these four match `QAExample`), not the absence of a benchmark. This should be weighed against the
month 9-10 validation timeline named in Section 10.

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
| BETR [Be25] | pretrain | benchmark-embedding similarity, distilled to fastText | low at inference, one-time distillation cost | partially (diverse-target variant, one aggregate score) |
| Data Mixing Agent [Ya26] | CPT | learned domain-mixture policy (offline RL) | moderate, one-time policy training | partially (multi-domain, single stage) |
| Meta-rater [Zh25] | pretrain | multi-dimensional learned quality score | low at inference | no |
| **This project** | **CPT + SFT + DPO** | **multiple, stage-specific** | **TBD** | **yes, by construction** |

### 7.6 Corpus-scale selection, 2026 update

Four results from a 2026 research pass bear directly on the CPT stage, and one of them changes
how `resolve_budget` should be reasoned about.

**The optimal selection ratio scales with compute, not a constant.** BETR [Be25] fits
F_opt(C) = 4e-5 x C^0.25, the fraction of the corpus worth keeping, rising from about 3% at
1e20 FLOPs to about 30% at 1e23 FLOPs. Larger training runs need *less* aggressive filtering. This
is falsifiable and directly load-bearing: it means a keep-rate copied from a large published recipe
is provably wrong at this project's scale, and a small-compute CPT run should filter far more
aggressively than DataComp-LM or similar large-scale work would suggest.

**Selecting for one target measurably costs the others.** BETR's own ablation is the clearest
empirical support in this review for the multi-target premise: a variant targeted at one benchmark
suite (Core) is best on that suite but falls to third place on held-out (Noncore) tasks. This
confirms Gap 2 is a real cost, not a hypothetical one, though it also means BETR itself already
occupies part of the multi-target space (Section 8 below).

**A domain-specific signal beats a generic quality classifier for medicine.** Huang et al. [Hu26],
at matched compute on a French medical encoder, find a medical-term density filter beats the
standard educational-quality classifier as a single-axis filter (78.27 versus 59.16 win
probability against an unfiltered baseline of 45.02), and the intersection of both beats either
alone. This is the first published, concrete result naming a domain-specific corpus-scale filter
for medical text, and it is a stronger baseline to beat than a generic FineWeb-Edu-style classifier
for this project's CPT stage. Caveat carried with the claim: encoder MLM in French, not decoder CPT
in English, and compared against unfiltered data, not a random subsample of equal size.

**The scariest-looking result does not license skipping selection here.** Mohri et al. [Mo26]
report unfiltered web data beating every quality filter tested, at model and data scale. The
crossing point where unfiltered data wins is real, but the authors' own projection puts it near
1e30 FLOPs for the full 240T-token pool, against roughly 5e26 FLOPs for a current frontier
pretraining run, and they state plainly that "when compute is a bottleneck, we expect filtering to
still be important." Below a model-size threshold the crossing does not occur at all. This
project's compute budget sits nowhere near the regime where the result applies.

One mechanism-level result changes what *not* to use as a selection signal: Nait Saada et al.
[Na26] find that quality filtering's benefit does not come from selecting documents that resemble
a high-quality reference distribution, which weakens the case for scoring CPT documents by
similarity to a small in-domain seed set as a standalone signal, and argues for combining it with a
direct quality or noise-removal criterion such as the medical-term density result above.

### 7.7 Preference-stage data selection

Section 5 covers this in detail as the answer to the Stage 3 data blocker. In selection terms, the
one-line summary from Djuhera et al. [Dj26] is that curated preference data beats the largest
single source: a 190k-pair mixture assembled across five open corpora beats the strongest
280k-pair single source on 14 evaluation tasks, the same "selection beats scale" claim this project
makes for CPT and SFT, now demonstrated for DPO too, on general-domain rather than medical data.

For a broader map of data-centric efficient training methods than this section covers, Luo et al.'s
2025 survey [Lu25] is worth citing wholesale rather than re-deriving.

---

## 8. Gap analysis

Reading across Sections 3-7, four gaps were consistent as of the original review. A 2026 research
pass revisited each adversarially, specifically to check whether anyone had closed them since. Two
were narrowed rather than closed; two stand as before; and one new gap emerged from the same pass.

**Gap 1: Single-stage scope. Narrowed, not closed.** Nearly all selection work targets SFT. LESS,
GLISTER, GradMatch and 3DS are all instruction-tuning methods. CPT-stage selection is dominated by
cheap heuristics (dedup, quality classifiers, distribution matching) because per-example gradient
methods do not scale to 23.9M documents. The live threat is He et al. [He26], published June 2026,
which states the same load-bearing premise this project makes, that different training stages want
different data because they are credited with different learning roles, and instantiates it as a
difficulty split between SFT and RL data. It has no CPT stage, single capability (math and logic
reasoning), and two small non-medical models. Meta-rater [Zh25], an ACL 2025 Best Theme Paper on
multi-dimensional pretraining selection, and Holistic Data Scheduler [Da26], KDD 2026, both remain
scoped to a single stage and never ask whether the best pretraining subset depends on downstream
SFT data. **The honest claim after this pass: someone now owns the SFT-to-RL half of Gap 1. Nobody
has published a policy spanning CPT through SFT through DPO, which is what this project claims.**

**Gap 2: Single-objective optimisation. Under active contest, needs sharpening.** Selection is
almost universally optimised against aggregate downstream accuracy on one benchmark. Three 2026
papers now occupy part of this space and must be distinguished from, not just cited past: Data
Mixing Agent [Ya26] explicitly balances performance across multiple domains (general versus math,
general versus code) during CPT and reports cross-domain variance as a metric, all tested
reweighting methods cut variance by over 200 versus the base model. Holistic Data Scheduler [Da26]
ships under the literal phrase "multi-objective data selection," though its three objectives are
internal training signals (a quality score, inter-domain loss influence, weight norms), not
downstream capabilities. BETR [Be25] has a diverse-target variant that matches or exceeds
single-target baselines on disjoint benchmarks. **None preserves several NAMED clinical
capabilities separately; all collapse their targets into one aggregate score or a small number of
coarse domains, and none combine multi-objective scope with the multi-stage scope of Gap 1.** The
project can no longer claim multi-target selection is unexplored at pretraining scale. It can still
claim that preserving named, distinct capabilities across stages is unexplored, which is a
narrower and more defensible claim than the original phrasing, and a reviewer familiar with Data
Mixing Agent or HDS will expect this distinction to be made explicitly.

**Gap 3: Long-tail and rare-event coverage is not a constraint. Unchanged.** Selection methods rank
and truncate. Anything scoring low is dropped, and rare diseases and underrepresented populations
score low almost by definition, because they are rare. Making coverage a *constraint* rather than
something the ranking might incidentally preserve is a genuine methodological difference, and it
is why the harness separates scoring from selection and gives `select_stratified` a
`min_per_group` floor. Nothing in the 2026 pass addressed long-tail coverage as a constraint rather
than an incidental ranking outcome.

**Gap 4: The baseline may be weaker than reported. Unchanged, evidence on both sides grew.** Per
Jeong et al. [J24], the premise that medical CPT reliably helps is not well established. Section
3.6 covers the 2026 follow-up in full: LiveMedBench [Yn26] and Dorfner et al. [Do25] extend the
contrarian finding to newer models and a fair clinical-task comparison, while Medmarks [Wa26] and
Domingo-Aldama et al. [Dm26] complicate it without reversing it. Rather than undermining the
project, this still sharpens it: the interesting claim is not "CPT helps" but "*selected* CPT helps
where unselected CPT does not", and that comparison remains one nobody has run.

**Gap 5: Benchmark and evaluation validity, new in this pass.** MedCheck [Ch26] audited 56 medical
LLM benchmarks against 46 lifecycle criteria and found 49 of 56 (88%) do not address data
contamination at all, 34% evaluate only a single dimension such as accuracy, and MedQA and MedMCQA
are named explicitly as paradigm cases of exam questions substituting for real clinical data. This
is not a new instance of Gap 4, it is a claim about the measuring instrument rather than the model:
even a correctly-run base-model control, scored on MedQA and MedMCQA, is being scored against
benchmarks whose own field-wide audit found wanting on contamination handling and dimensionality.
Section 6.5 names four open, non-credentialed benchmarks that partially address this by measuring
capabilities beyond MCQ; adopting even one would directly answer part of this gap. CapTrack [Th26]
supplies a reusable framework for measuring what MedCheck's audit found underserved: it decomposes
retention into three groups (latent competence, behavioural preference, protocol compliance)
rather than a single aggregate score, which is closer to the per-capability measurement this
project needs than accuracy alone.

A sixth, more practical gap: **auditability**. The proposal promises interpretable data-valuation
scores and explanations for why a sample was kept or discarded. Almost nothing in Section 7
produces a human-readable justification; influence scores are numbers without narratives. In a
clinical setting that is a real deficiency and a defensible contribution on its own. Fully Open
Meditron [Te26] is a concrete 2026 precedent worth reading for how an auditable clinical LLM
pipeline can be built and documented, though it targets pipeline transparency rather than
per-sample selection justification, so it does not close this gap either.

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
| CPT selection ratio should scale with compute rather than being a fixed constant | §7.6: F_opt(C) = 4e-5 x C^0.25 [Be25] |
| A domain-term-density scorer is a stronger CPT baseline than a generic quality classifier | §7.6: medical-term density beat educational quality on a matched-compute medical corpus [Hu26] |
| Gap 1 and Gap 2 claims must be stated as the narrower, cross-stage and named-capability versions | §8: both gaps were partially occupied by 2026 work [He26] [Ya26] [Da26] |
| A non-MCQ benchmark should be adopted before month 9-10 validation, not deferred | §6.5, §8 Gap 5: field-wide audit found MCQ-only evaluation and single-dimension scoring a documented weakness [Ch26] |

Concrete near-term work items this review generates, beyond the proposal's plan:

1. **Build the PMID exclusion list** from PubMedQA `pqa_labeled` (and `pqa_artificial` if used for
   SFT) and wire it into the PubMed loader as a filter. Until this exists, no PubMedQA number from
   a CPT'd model is trustworthy.
2. **Run the base-model control** for every benchmark before any CPT run, and store it as a
   committed reference result.
3. **Add bootstrap confidence intervals** to `EvalReport`. MedQA's 1,273-item test set cannot
   support the precision people routinely claim on it.
4. **Adopt at least one non-MCQ benchmark** from Section 6.5 (HealthBench, BRIDGE, ACI-Bench or
   Medmarks) rather than deferring capability coverage past validation. The survey this item used
   to depend on is now done.
5. **Build a compute-aware selection budget** rather than a fixed keep-rate, following the
   F_opt(C) relationship in Section 7.6, and treat medical-term density as the baseline the CPT
   scorer must beat rather than a generic quality classifier.
6. **Use the Djuhera et al. [Dj26] preference corpora as a general-domain replay source for Stage
   3**, mixed with the medically-constructed pairs from options 1 and 2 in Section 5, rather than
   waiting for a medical-specific preference set that a 2026 survey did not find.

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
   **Update, 2026:** this is now partly answerable rather than fully open. Section 6.5 names four
   open, non-credentialed benchmarks (HealthBench, BRIDGE, ACI-Bench, Medmarks) that between them
   cover information extraction, ICD-10 coding, summarisation and open-ended clinical reasoning.
   The question narrows from "does a usable benchmark exist" to "which one to integrate first, and
   whether integrating one is enough to defend the six-capability claim."
6. Given that He et al. [He26] and Data Mixing Agent [Ya26] each occupy part of Gap 1 or Gap 2
   individually, does *combining* stage-awareness and multi-capability preservation in one policy,
   the project's actual claim, produce a result neither paper's narrower version would predict? If
   the combination is separable into independent per-stage and per-capability effects, the
   project's contribution is the engineering of the combination rather than a new empirical
   finding, and that distinction should be decided before, not after, running experiments.
7. Baek et al. [Ba26] argue domain data belongs as early in training as possible, which questions
   whether CPT, SFT and DPO should be treated as needing separate policies at all, rather than one
   policy that places domain data optimally across a blurred stage boundary. Does this apply to
   medical CPT specifically, where the domain shift (general web text to biomedical abstracts) is
   larger than the chemistry, music and proof domains the paper tested? This is a direct challenge
   to the project's stage-separation premise and deserves an explicit answer, not a citation past.

Question 5 was the most uncomfortable one in the original review, and is now the one where a 2026
research pass changed the answer most: the proposal names six clinical capabilities; the three
datasets measure roughly one and a half of them, but usable open benchmarks for several of the
rest now exist and do not require new construction. Question 6 is the sharper novelty question the
project must now answer given Section 8, since two separate 2026 papers each independently occupy
half of the claimed novelty. Question 7 is the more fundamental challenge, since it questions the
stage-separation premise directly rather than the coverage of existing selection methods. All three
should be resolved before the month 9-10 validation phase, not during it.

---

## 11. References

Keyed to `references.bib`. Entries tagged 2026-07 came from a deep research pass that post-dates
the original review; see Section 3.6, 5, 6.5, 7.6, 7.7 and 8 for how each is used.

- **[B24]** Bolton et al. *BioMedLM: A 2.7B Parameter Language Model Trained On Biomedical Text.* 2024.
- **[Ba26]** Baek, Maini et al. *The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data.* DatologyAI, 2026. arXiv:2603.16177. (2026-07)
- **[Be25]** *Language Models Improve When Pretraining Data Matches Target Tasks (BETR).* 2025. arXiv:2507.12466. Author list unverified from the search snippet, verify before citing in a manuscript. (2026-07)
- **[C23]** Chen et al. *MEDITRON-70B: Scaling Medical Pretraining for Large Language Models.* 2023.
- **[Ch26]** Chen et al. *Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models (MedCheck).* 2025. arXiv:2508.04325. (2026-07)
- **[D25]** Ding et al. *3DS: Medical Domain Adaptation of LLMs via Decomposed Difficulty-based Data Selection.* EMNLP 2025. arXiv:2410.10901.
- **[Da26]** Dang, Ma, Liao. *Holistic Data Scheduler for LLM Pre-training via Multi-Objective Reinforcement Learning.* KDD 2026. arXiv:2606.24133. (2026-07)
- **[Dj26]** Djuhera et al. *When Data is the Algorithm: A Systematic Study and Curation of Preference Optimization Datasets.* ICLR 2026. arXiv:2511.10985. (2026-07)
- **[Dm26]** Domingo-Aldama et al. *To Adapt or not to Adapt: Rethinking the Value of Medical Knowledge-Aware Large Language Models.* 2026. arXiv:2604.06854. (2026-07)
- **[Do25]** Dorfner et al. *Evaluating the Effectiveness of Biomedical Fine-tuning for Large Language Models on Clinical Tasks.* JAMIA 32(6):1015-1024, 2025. doi:10.1093/jamia/ocaf045. (2026-07)
- **[G20]** Gururangan et al. *Don't Stop Pretraining: Adapt Language Models to Domains and Tasks.* ACL 2020.
- **[G23]** Gadre et al. *DataComp: In Search of the Next Generation of Multimodal Datasets.* NeurIPS 2023.
- **[He26]** He, Zhang, Wang, Li. *Learning What to Learn: Stage-Specific Data Sets for SFT-then-RL in Small Language Model Reasoning.* 2026. arXiv:2606.04466. (2026-07)
- **[Hu26]** Huang et al. *Where Does the Signal Live? A Web Data Recipe for Medical Encoder Pretraining.* 2026. arXiv:2606.22079. (2026-07)
- **[I24]** Ibrahim et al. *Simple and Scalable Strategies to Continually Pre-train Large Language Models.* TMLR 2024. arXiv:2403.08763.
- **[J19]** Jin et al. *PubMedQA: A Dataset for Biomedical Research Question Answering.* EMNLP 2019.
- **[J21]** Jin et al. *What Disease Does This Patient Have? A Large-Scale Open Domain Question Answering Dataset from Medical Exams (MedQA).* 2021.
- **[J24]** Jeong et al. *The Limited Impact of Medical Adaptation of Large Language and Vision-Language Models.* EMNLP 2024. arXiv:2411.08870.
- **[K17]** Koh & Liang. *Understanding Black-box Predictions via Influence Functions.* ICML 2017.
- **[K21a]** Killamsetty et al. *GLISTER: Generalization based Data Subset Selection for Efficient and Robust Learning.* AAAI 2021.
- **[K21b]** Killamsetty et al. *GRAD-MATCH: Gradient Matching based Data Subset Selection.* ICML 2021.
- **[L24]** Li et al. *DataComp-LM: In Search of the Next Generation of Training Sets for Language Models.* 2024. arXiv:2406.11794.
- **[L24b]** Labrak et al. *BioMistral: A Collection of Open-Source Pretrained Large Language Models for Medical Domains.* 2024.
- **[Lu25]** Luo et al. *A Survey on Efficient Large Language Model Training: From Data-centric Perspectives.* ACL 2025. arXiv:2510.25817. (2026-07)
- **[M25]** MedGemma Team, Google. *MedGemma Technical Report.* 2025. arXiv:2507.05201.
- **[Mo26]** Mohri, Duchi, Hashimoto. *A Bitter Lesson for Data Filtering.* Stanford, 2026. arXiv:2605.19407. (2026-07)
- **[Na26]** Nait Saada et al. *Removing Noise, not Finding Gold: Quality Filtering for Large-Scale Pretraining.* ICML 2026. arXiv:2510.00866. (2026-07)
- **[Op25]** OpenAI. *HealthBench: Evaluating Large Language Models Towards Improved Human Health.* 2025. arXiv:2505.08775. (2026-07)
- **[P22]** Pal et al. *MedMCQA: A Large-scale Multi-Subject Multi-Choice Dataset for Medical domain Question Answering.* CHIL 2022.
- **[P23]** Park et al. *TRAK: Attributing Model Behavior at Scale.* ICML 2023.
- **[Te26]** Theimer-Lienhard et al. *Fully Open Meditron: An Auditable Pipeline for Clinical LLMs.* 2026. arXiv:2605.16215. (2026-07)
- **[Th26]** Thede, Winzeck, Akata, Schwarz. *CapTrack: Multifaceted Evaluation of Forgetting in LLM Post-Training.* 2026. arXiv:2603.06610. (2026-07)
- **[W23]** Wu et al. *PMC-LLaMA: Towards Building Open-source Language Models for Medicine.* 2023.
- **[Wa26]** Warner et al. *Medmarks: A Comprehensive Open-Source LLM Benchmark Suite for Medical Tasks.* 2026. arXiv:2605.01417. (2026-07)
- **[Wu25]** Wu et al. *BRIDGE: Benchmarking Large Language Models for Understanding Real-world Clinical Practice Text.* 2025. arXiv:2504.19467. (2026-07)
- **[X24a]** Xia et al. *LESS: Selecting Influential Data for Targeted Instruction Tuning.* ICML 2024. arXiv:2402.04333.
- **[X24b]** Xiong et al. *Benchmarking Retrieval-Augmented Generation for Medicine.* 2024. arXiv:2402.13178. (Source of the `MedRAG/pubmed` corpus.)
- **[X24c]** Xie et al. *Me-LLaMA: Foundation Large Language Models for Medical Applications.* 2024.
- **[Ya26]** Yang et al. *Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training.* ACL 2026. arXiv:2507.15640. (2026-07)
- **[Yi23]** Yim et al. *ACI-Bench: a Novel Ambient Clinical Intelligence Dataset for Benchmarking Automatic Visit Note Generation.* Scientific Data, 2023. (2026-07)
- **[Yn26]** Yan et al. *LiveMedBench: A Contamination-Free Medical Benchmark for LLMs with Automated Rubric Evaluation.* 2026. arXiv:2602.10367. (2026-07)
- **[Zg24]** Zhang et al. *UltraMedical: Building Specialized Generalists in Biomedicine.* NeurIPS 2024 Datasets and Benchmarks Track, Spotlight. arXiv:2406.03949. (2026-07)
- **[Zh25]** Zhuang et al. *Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models.* ACL 2025 Best Theme Paper. arXiv:2504.14194. (2026-07)
- **[Zu25]** Zuo et al. *MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding.* ICML 2025. arXiv:2501.18362. (2026-07)
