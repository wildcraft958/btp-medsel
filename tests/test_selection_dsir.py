import pytest

from medsel.schema import CorpusDoc, QAExample
from medsel.selection.base import get_scorer
from medsel.selection.scorers.dsir import _record_as_prompt, hashed_ngram_counts

MEDICAL = [
    "Patients with acute myocardial infarction received reperfusion therapy within six hours.",
    "The randomised trial measured serum creatinine and glomerular filtration rate at baseline.",
    "Antibiotic resistance in gram negative bacteria complicated the postoperative course.",
    "Histopathology confirmed adenocarcinoma with lymph node metastasis in twelve patients.",
]

UNRELATED = [
    "The midfielder scored twice in the second half of the cup final on Saturday.",
    "Preheat the oven and whisk the butter and sugar until the mixture turns pale.",
    "Trains to the coast were delayed after signalling faults near the junction.",
    "She restrung the guitar and tuned it down a whole step before the rehearsal.",
]


def docs(texts):
    return [CorpusDoc(uid=f"d{i}", source="test", text=t) for i, t in enumerate(texts)]


class TestHashedNgramCounts:
    def test_is_stable_across_calls(self):
        assert hashed_ngram_counts("acute myocardial infarction", 512) == hashed_ngram_counts(
            "acute myocardial infarction", 512
        )

    def test_respects_bucket_bound(self):
        counts = hashed_ngram_counts("a fairly ordinary sentence about medicine", 64)
        assert counts and all(0 <= bucket < 64 for bucket in counts)

    def test_counts_unigrams_and_bigrams(self):
        # Three tokens give three unigrams plus two bigrams.
        assert sum(hashed_ngram_counts("one two three", 4096, n=2).values()) == 5

    def test_normalisation_ignores_case_and_punctuation(self):
        assert hashed_ngram_counts("Acute, Myocardial!", 512) == hashed_ngram_counts(
            "acute myocardial", 512
        )

    def test_empty_text_gives_no_counts(self):
        assert hashed_ngram_counts("", 512) == hashed_ngram_counts("   ", 512)


class TestDSIRScorer:
    def test_one_score_per_record(self):
        scorer = get_scorer("dsir", target=MEDICAL, seed=0)
        assert len(scorer(docs(UNRELATED))) == 4

    def test_empty_pool_gives_empty_scores(self):
        assert get_scorer("dsir", target=MEDICAL)([]) == []

    def test_prefers_documents_resembling_the_target(self):
        scorer = get_scorer("dsir", target=MEDICAL, gumbel=False)
        scores = scorer(docs(MEDICAL + UNRELATED))
        assert min(scores[:4]) > max(scores[4:])

    def test_is_deterministic_for_a_fixed_seed(self):
        pool = docs(MEDICAL + UNRELATED)
        first = get_scorer("dsir", target=MEDICAL, seed=7)(pool)
        second = get_scorer("dsir", target=MEDICAL, seed=7)(pool)
        assert first == second

    def test_gumbel_noise_changes_scores_between_seeds(self):
        pool = docs(MEDICAL + UNRELATED)
        assert get_scorer("dsir", target=MEDICAL, seed=1)(pool) != get_scorer(
            "dsir", target=MEDICAL, seed=2
        )(pool)

    def test_rejects_an_empty_target(self):
        with pytest.raises(ValueError, match="target"):
            get_scorer("dsir", target=[])(docs(MEDICAL))

    def test_registered_under_its_name(self):
        from medsel.selection.base import available_scorers

        assert "dsir" in available_scorers()


class TestPoolTargetConsistency:
    """Pool and target texts must use the same format for QAExamples."""

    def test_qa_example_uses_render_prompt(self):
        ex = QAExample(
            uid="q1",
            source="test",
            split="train",
            question="What causes fever?",
            options={"A": "Virus", "B": "Allergy"},
            answer_key="A",
        )
        text = _record_as_prompt(ex)
        assert "Question:" in text
        assert "A." in text
        assert "Answer:" in text

    def test_corpus_doc_falls_through(self):
        doc = CorpusDoc(uid="d1", source="test", text="Some medical text")
        assert _record_as_prompt(doc) == "Some medical text"


class TestTargetSplitGuard:
    """Selecting towards the split you are scored on is contamination, so it is refused."""

    @pytest.mark.parametrize(
        ("task", "expected"),
        [("medqa", "train"), ("medmcqa", "train"), ("pubmedqa", "train[500:]")],
    )
    def test_defaults_to_a_training_split(self, task, expected):
        assert get_scorer("dsir", target=task).resolve_target_split() == expected

    @pytest.mark.parametrize(
        ("task", "split"),
        [("medmcqa", "validation"), ("medqa", "test"), ("pubmedqa", "train[:500]")],
    )
    def test_rejects_the_evaluation_split(self, task, split):
        scorer = get_scorer("dsir", target=task, target_split=split)
        with pytest.raises(ValueError, match="scored on"):
            scorer.resolve_target_split()

    def test_rejects_a_whole_split_that_contains_the_eval_slice(self):
        scorer = get_scorer("dsir", target="pubmedqa", target_split="train")
        with pytest.raises(ValueError, match="scored on"):
            scorer.resolve_target_split()

    def test_allows_a_disjoint_slice(self):
        scorer = get_scorer("dsir", target="pubmedqa", target_split="train[500:]")
        assert scorer.resolve_target_split() == "train[500:]"

    def test_allows_a_genuinely_different_split(self):
        scorer = get_scorer("dsir", target="medmcqa", target_split="train")
        assert scorer.resolve_target_split() == "train"

    def test_explicit_texts_need_no_split(self):
        assert get_scorer("dsir", target=MEDICAL).resolve_target_split() is None


class TestNoiseCalibration:
    """Gumbel noise must perturb the ranking, not replace it.

    The length-normalised importance weight has a standard deviation around 0.12 on PubMed, while
    standard Gumbel noise has one of 1.28. Adding the two directly buries the signal under ten
    times its own size and the scorer selects at random.
    """

    # A gradient rather than two clumps: each document mixes medical and unrelated vocabulary in a
    # different proportion, so the clean ranking has few ties for noise to reorder freely.
    POOL = [
        " ".join(" ".join(MEDICAL).split()[:k] + " ".join(UNRELATED).split()[: 20 - k])
        for k in range(1, 21)
    ]

    def ranks(self, scores):
        order = sorted(range(len(scores)), key=lambda i: scores[i])
        out = [0] * len(scores)
        for rank, index in enumerate(order):
            out[index] = rank
        return out

    def spearman(self, left, right):
        import statistics

        return statistics.correlation(
            [float(v) for v in self.ranks(left)], [float(v) for v in self.ranks(right)]
        )

    def test_noise_scale_is_derived_from_the_weight_spread(self):
        scorer = get_scorer("dsir", target=MEDICAL)
        scorer.fit(docs(self.POOL))
        assert scorer._noise_scale > 0

    def test_noise_ratio_zero_matches_the_noiseless_ranking(self):
        pool = docs(self.POOL)
        noisy = get_scorer("dsir", target=MEDICAL, noise_ratio=0.0)(pool)
        clean = get_scorer("dsir", target=MEDICAL, gumbel=False)(pool)
        assert noisy == pytest.approx(clean)

    def test_signal_survives_the_default_noise(self):
        pool = docs(self.POOL)
        noisy = get_scorer("dsir", target=MEDICAL, seed=11)(pool)
        clean = get_scorer("dsir", target=MEDICAL, gumbel=False)(pool)
        assert self.spearman(noisy, clean) > 0.8

    def test_noise_still_perturbs_the_ranking(self):
        pool = docs(self.POOL)
        first = get_scorer("dsir", target=MEDICAL, seed=1)(pool)
        second = get_scorer("dsir", target=MEDICAL, seed=2)(pool)
        assert first != second

    def test_does_not_share_a_random_stream_with_the_random_scorer(self):
        """Same seed, same generator, and Gumbel monotone in the draw made these agree."""
        pool = docs(self.POOL)
        dsir = get_scorer("dsir", target=MEDICAL, seed=42, noise_ratio=1e6)(pool)
        uniform = get_scorer("random", seed=42)(pool)
        assert self.spearman(dsir, uniform) < 0.99

    def test_rejects_a_negative_noise_ratio(self):
        with pytest.raises(ValueError, match="noise_ratio"):
            get_scorer("dsir", target=MEDICAL, noise_ratio=-1.0)

    def test_a_pool_with_no_signal_still_breaks_ties_randomly(self):
        """Target equal to pool means every weight is alike, and no spread to scale noise against.

        Scaling to zero there would quietly return the first k records in stream order.
        """
        scorer = get_scorer("dsir", target=MEDICAL, seed=5)
        pool = docs(MEDICAL)
        scorer.fit(pool)
        assert scorer._noise_scale > 0
        assert scorer.score(pool) != scorer.score(pool)
