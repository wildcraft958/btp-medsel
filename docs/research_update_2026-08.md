# Verification Audit, August 2026

The July 2026 research pass recorded its findings in
[research_findings_2026-07.json](research_findings_2026-07.json) and folded them into
[literature_review.md](literature_review.md). This pass put those findings through adversarial
verification: three independent skeptics per claim, each instructed to refute, two refutations kill
a claim.

**It is an audit, not a new survey.** Almost every paper an August search surfaced was already in
`references.bib`. The value here is in what verification changed, which is listed in section 2, and
in four references that were genuinely missing.

Raw verdicts with full refutation evidence: [verification_2026-08.json](verification_2026-08.json).

---

## 1. What was verified

45 claims adjudicated. **22 survived, 23 were killed.**

Every paper was confirmed to exist, and every number was checked against the paper body or PDF
rather than the abstract. **No citation in the review was fabricated and no number was wrong.**
All 23 kills were interpretation: a correct number with an inference attached that the paper does
not support. That is a good result for the bibliography and a warning about the prose.

**The two papers `status.md` flagged as needing a human read are now independently confirmed.**
Open item 1 asked someone to read `[He26]` (arXiv:2606.04466) and `[Ya26]` (arXiv:2507.15640) in
full because they reshape the gap analysis. Three adversarial readers each fetched the primary
sources and confirmed both at 3-0:

- `[He26]` does state this project's premise, that data strategy should align with the distinct
  roles of each stage, and does instantiate it as a per-stage split. It is SFT-then-RL only, no
  CPT, maths and logic only, aggregate accuracy. **Gap 1 stands, narrowed.**
- `[Ya26]` does optimise balanced performance across fields and does report cross-field variance
  (all reweighting methods cut variance by over 200 versus base). Its objectives are coarse domains
  inside CPT alone. **Gap 2 stands, narrowed.**

A human read is still worth doing before the defence, but the gap analysis is no longer resting on
an unread source.

---

## 2. Corrections to the existing docs

Three errors found and already fixed in `literature_review.md` section 6.5.

**HealthBench is MIT, not CC BY-4.0.** The CC BY 4.0 on the arXiv page is the licence of the
preprint, not the benchmark. `github.com/openai/simple-evals` ships an MIT LICENSE and its README
marks HealthBench MIT while marking MGSM CC-BY, so the distinction is deliberate. Also, it is not
"5,000 multi-turn conversations": 2,915 of 5,000 (58.3%) are single-turn, mean 2.61 turns, max 19,
which the paper body states in Section 3 against its own abstract. The data file is at
`openaipublic.blob.core.windows.net/simple-evals/healthbench/2025-05-07-06-14-12_oss_eval.jsonl`,
60 MB, no auth.

**BRIDGE is not fully open.** 87 tasks exist but only **55 are redistributable** in BRIDGE-Open;
the remainder need standardized request or PhysioNet credentials, and several diagnosis tasks are
MIMIC-based and therefore out of reach here. The leaderboard accepts model submissions without data
access, which gives a one-off score but not the repeatable local eval loop a retention metric needs.
Upgrade the citation: it is now peer reviewed in Nature Biomedical Engineering,
doi 10.1038/s41551-026-01719-2, reporting 107 LLMs and a 44.8 top score.

**The Medmarks saturation claim was cutting the wrong way.** The review used it to confirm that
MedQA, MedMCQA and PubMedQA are exhausted. The same paper's own difficulty table puts **MedMCQA at
0.656 and MedQA at 0.784 in the moderate tier**, not the near-ceiling cluster (PubHealthBench 0.831,
LongHealth 0.829, CareQA 0.825), gives no figure for PubMedQA at all, and carries all three forward
into Medmarks-V and the RL-trainable Medmarks-T. The saturation sentence is scoped to **frontier
models**.

> Practical consequence: **the three datasets remain usable for this project.** Models at this
> scale are nowhere near the ceiling that sentence describes. The non-MCQ work in open item 2 is
> still worth doing for capability coverage, but not because the existing benchmarks are dead.

---

## 3. Misreadings to avoid

Each is a plausible reading of a real paper that verification killed. Worth knowing before one of
them reaches a defence.

**CapTrack's forgetting numbers argue against a data explanation, not for one.** The figures are
real (IFT up to -12.7%, DPO about -2.5% average), but the sentence containing them reads
"indicating that these differences are not primarily driven by data composition". The authors ran
that controlled experiment specifically to *rule out* data as the cause, so the stage difference
they document is algorithmic. Citing it as evidence that stages need different *data* inverts it.
Two further faults: the ratio compares a DPO average against an IFT worst case, so "a factor of
five" is not a quantity the paper reports; and the paper warns the metric measures deviation from
the base model, where "not all observed deviations correspond to undesirable behavior".

**"Aggregate benchmarks cannot detect capability regression" is false as stated.** In the same
paper, standard accuracy benchmarks did detect large regression (parametric knowledge -11.3% for
Gemma). The multilingual -16.4% headline came from Qwen and Llama with "Gemma largely unaffected",
so it does not transfer to a Gemma 3 pipeline. The true, weaker version: accuracy-centric
evaluation of forgetting is insufficient on its own.

**The MedGemma versus Gemma 3 comparison is not a null result.** BRIDGE reports MedGemma-27B-it
40.8 [40.0 to 41.6] against Gemma-3-27B-it 39.9 [39.1 to 40.7], p=0.12. Real and relevant, since
this project's base is Gemma 3. But the paper writes "outperformed", frames it as "promising yet
inconsistent", and notes MedGemma-27B-it ranked 14th of 95 with the highest average across
specialties (44.7). Calling it "statistically indistinguishable" borrows a conclusion the authors
declined to draw.

**The Marmoka result is not a from-base CPT comparison.** The numbers are exact (26B words of
English medical text, +0.9 on MedQA, -0.8 on PubMedQA), but Section 3.3.1 states the starting point
is Llama 3.1-8B-**Instruct**, so it says nothing about CPT from a base checkpoint. Several quoted
accuracies are sums of published deltas rather than printed values.

**Random-beats-selection reverses at low budget.** The ICML 2026 critique is real and its
budget-shrinking result is proved (Theorem 6.2 bounds how much a Wasserstein-optimal subset can beat
random). Two caveats the headline drops: the low-budget regime runs the other way, and that is this
project's regime; and every method it tests is a fixed geometric rule over a representation, not a
learned policy, so it does not directly test this project's thesis.

---

## 4. Four references that were missing

Added to `references.bib`. These are the only genuinely new sources this pass produced.

| Key | Paper | Why it matters here |
|---|---|---|
| `wang2026proxy` | Can Small Training Runs Reliably Guide Data Curation? ICLR 2026, arXiv:2512.24503 | Data recipe rankings **flip** under minor hyperparameter changes, because the optimal config is data-dependent. Directly threatens picking a selection policy cheaply and scaling it up. |
| `abbes2025replay` | Revisiting Replay and Gradient Alignment for Continual Pre-Training, arXiv:2508.01908 | 0 to 25% replay beats growing the model; 25 to 50% does not. Only 0, 0.25, 0.5 tested. Cross-lingual, not medical, so analogy only. |
| `nayak2026critical` | A Critical Look at Targeted Instruction Selection, ICML 2026, arXiv:2602.14696 | The strongest published case that selection may not beat random. Cite it and scope it before someone else does. |
| `idahl2026propella` | propella-1, arXiv:2602.12414 | Interpretable multi-property annotation at scale (18 properties, 57 languages, Apache 2.0), but the authors ran **no downstream training experiment** and name multi-dimension selection validation as future work. |

The propella-1 gap is worth reading twice. This project's proposal promises human-readable
justifications for why a sample was kept or discarded. The annotation layer now exists, openly
licensed, and the validation nobody has run is precisely this project's experiment.

---

## 5. What this changes for the GPU runs

Deduplicated against the open items already in [status.md](status.md). Those three proposals
(`medterm_density` scorer, compute-aware selection budget, cross-capability variance in
`EvalReport`) all survive verification unchanged, so they stay as written. Four additions:

1. **Validate selection at a low learning rate, or do not trust the ranking.** Per
   `wang2026proxy`, holding hyperparameters identical across recipes in the name of a fair
   comparison is itself the source of unreliability. If a small run picks the selection policy for
   a larger one, sweep the learning rate rather than fixing it, and report that you did. This is
   the highest-value new finding for anyone starting GPU experiments today.
2. **Add a `dedup_plus_quality` baseline.** BETR beat a dedup-plus-quality-classifier baseline, not
   just random, so random alone understates the bar. `random` and `length` exist already; this is
   the missing third.
3. **Set replay to about 0.25** if replay is used at all, per `abbes2025replay`, and note that the
   optimum below 25% is unresolved and the evidence is cross-lingual rather than medical.
4. **Report the selection budget and the FLOP budget next to every selection result.** Both the
   BETR keep-rate law and the `nayak2026critical` reversal are budget-conditional, so a keep-rate
   quoted without its compute budget is not interpretable.

Unchanged and still true: the PubMed to PubMedQA contamination filter is already built and opt-in
via `exclude_pmids`; use it before quoting any PubMedQA number from a CPT'd model.
