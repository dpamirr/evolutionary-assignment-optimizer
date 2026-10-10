from random import Random
import pytest
from evo_assignment.core import Config, fitness, groups, mutate, optimize, ordered_crossover, synthetic_participants


def test_operators_preserve_permutations():
    a, b = tuple(range(12)), tuple(reversed(range(12)))
    for seed in range(40):
        assert sorted(ordered_crossover(a, b, Random(seed))) == list(a)
        assert sorted(mutate(a, Random(seed))) == list(a)


def test_optimizer_assigns_everyone_once_and_improves_baseline():
    people = synthetic_participants(18)
    result = optimize(people, Config(population=40, generations=35), seed=10)
    assert sorted(result.chromosome) == list(range(18))
    assert all(len(team) == 3 for team in groups(result.chromosome, 3))
    assert result.score >= result.baseline_score
    assert all(b >= a for a, b in zip(result.history, result.history[1:]))
    assert fitness(result.chromosome, people, Config()) == pytest.approx(result.score)


def test_seed_reproducibility():
    p = synthetic_participants(12)
    cfg = Config(population=20, generations=12)
    assert optimize(p, cfg, 12) == optimize(p, cfg, 12)


def test_invalid_group_count():
    with pytest.raises(ValueError, match='divisible'):
        optimize(synthetic_participants(10))
