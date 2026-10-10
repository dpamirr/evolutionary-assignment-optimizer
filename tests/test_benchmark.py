from evo_assignment.benchmark import random_search, run_benchmark
from evo_assignment.core import Config, synthetic_participants


def test_benchmark_matched_evaluation_counts_and_history():
    cfg = Config(population=12, generations=5)
    trials = run_benchmark(range(2), participants=12, cfg=cfg)
    for t in trials:
        assert t.evaluations == t.evaluation_history[-1]
        assert len(t.ga_history) == len(t.random_history) == len(t.evaluation_history)
        assert all(b >= a for a, b in zip(t.random_history, t.random_history[1:]))
        assert t.ga_score == t.ga_history[-1]
        assert t.random_score == t.random_history[-1]


def test_random_search_is_reproducible():
    people = synthetic_participants(12)
    cfg = Config()
    assert random_search(people, cfg, 50, 3, [1, 5, 50]) == random_search(people, cfg, 50, 3, [1, 5, 50])
