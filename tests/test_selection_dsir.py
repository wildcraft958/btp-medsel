import pytest

from medsel.schema import CorpusDoc
from medsel.selection.base import get_scorer
from medsel.selection.scorers.dsir import hashed_ngram_counts

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
