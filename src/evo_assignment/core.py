"""Pure-Python genetic algorithm for fixed-size dinner groups.

A chromosome is a permutation of participant indices. Consecutive slices form
teams. Every participant therefore appears exactly once by construction.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import hypot
from random import Random


@dataclass(frozen=True)
class Participant:
    id: str
    x: float
    y: float
    diet: str
    age: int
    interests: frozenset[str]


@dataclass(frozen=True)
class Config:
    group_size: int = 3
    population: int = 100
    generations: int = 160
    mutation_rate: float = 0.25
    elite: int = 5
    tournament: int = 4
    distance_weight: float = 1.0
    interests_weight: float = 3.0
    age_weight: float = 0.04
    diet_penalty: float = 15.0

    def validate(self) -> None:
        if self.group_size < 2 or self.population < 4 or self.generations < 1:
            raise ValueError('group_size >= 2, population >= 4, generations >= 1 required')
        if not (0 <= self.mutation_rate <= 1) or not (1 <= self.elite < self.population):
            raise ValueError('invalid mutation_rate or elite')
        if not (2 <= self.tournament <= self.population):
            raise ValueError('invalid tournament size')


def groups(chromosome: tuple[int, ...], group_size: int) -> list[tuple[int, ...]]:
    return [chromosome[i:i + group_size] for i in range(0, len(chromosome), group_size)]


def fitness(chromosome: tuple[int, ...], people: list[Participant], cfg: Config) -> float:
    """Higher is better. Penalize geographic distance and incompatible diets;
    reward shared interests and age proximity. All scores are illustrative.
    """
    total = 0.0
    for team in groups(chromosome, cfg.group_size):
        for pos, i in enumerate(team):
            for j in team[pos + 1:]:
                a, b = people[i], people[j]
                total -= cfg.distance_weight * hypot(a.x - b.x, a.y - b.y)
                total += cfg.interests_weight * len(a.interests & b.interests)
                total -= cfg.age_weight * abs(a.age - b.age)
                if a.diet != b.diet and 'omnivore' not in (a.diet, b.diet):
                    total -= cfg.diet_penalty
    return total


def ordered_crossover(a: tuple[int, ...], b: tuple[int, ...], rng: Random) -> tuple[int, ...]:
    """Order crossover (OX) preserves permutation validity."""
    n = len(a)
    if n < 2:
        return a
    left, right = sorted(rng.sample(range(n), 2))
    child: list[int | None] = [None] * n
    child[left:right + 1] = a[left:right + 1]
    used = set(a[left:right + 1])
    remaining = (gene for gene in b if gene not in used)
    for idx in list(range(right + 1, n)) + list(range(0, left)):
        child[idx] = next(remaining)
    return tuple(gene for gene in child if gene is not None)


def mutate(chromosome: tuple[int, ...], rng: Random) -> tuple[int, ...]:
    genes = list(chromosome)
    i, j = rng.sample(range(len(genes)), 2)
    genes[i], genes[j] = genes[j], genes[i]
    return tuple(genes)


@dataclass
class Result:
    chromosome: tuple[int, ...]
    score: float
    history: list[float]
    baseline_score: float
    evaluations: int = 0
    evaluation_history: list[int] | None = None


def optimize(people: list[Participant], cfg: Config = Config(), seed: int = 42) -> Result:
    cfg.validate()
    if len(people) < cfg.group_size or len(people) % cfg.group_size:
        raise ValueError('Participant count must be divisible by group_size')
    if len({p.id for p in people}) != len(people):
        raise ValueError('Participant IDs must be unique')
    rng = Random(seed)
    evaluations = 0
    def score(c: tuple[int, ...]) -> float:
        nonlocal evaluations
        evaluations += 1
        return fitness(c, people, cfg)
    start = tuple(range(len(people)))
    baseline = list(start)
    rng.shuffle(baseline)
    baseline_score = score(tuple(baseline))
    population = [tuple(baseline)] + [tuple(rng.sample(start, len(start))) for _ in range(cfg.population - 1)]
    history = []
    evaluation_history = []
    for _ in range(cfg.generations):
        ranked = sorted(population, key=score, reverse=True)
        history.append(score(ranked[0]))
        evaluation_history.append(evaluations)
        next_pop = ranked[:cfg.elite]
        def select() -> tuple[int, ...]:
            candidates = rng.sample(population, cfg.tournament)
            return max(candidates, key=score)
        while len(next_pop) < cfg.population:
            child = ordered_crossover(select(), select(), rng)
            if rng.random() < cfg.mutation_rate:
                child = mutate(child, rng)
            next_pop.append(child)
        population = next_pop
    best = max(population, key=score)
    best_score = score(best)
    if best_score < max(history):
        # Included for clarity: elitism normally guarantees this cannot occur.
        raise AssertionError('elitism invariant violated')
    history.append(best_score)
    evaluation_history.append(evaluations)
    return Result(best, best_score, history, baseline_score, evaluations, evaluation_history)


def synthetic_participants(count: int = 24, seed: int = 42) -> list[Participant]:
    if count < 2:
        raise ValueError('Need at least two participants')
    rng = Random(seed)
    interests = ['music', 'hiking', 'coding', 'books', 'films', 'art', 'sports']
    return [Participant(f'P{i + 1:03d}', round(rng.uniform(0, 10), 2),
                        round(rng.uniform(0, 10), 2),
                        rng.choice(['omnivore', 'vegetarian', 'vegan']),
                        rng.randint(18, 65), frozenset(rng.sample(interests, 3)))
            for i in range(count)]
