"""Reproducible GA versus random-search benchmark (matched objective evaluations)."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean, stdev

from .core import Config, fitness, optimize, synthetic_participants


@dataclass(frozen=True)
class Trial:
    seed: int
    ga_score: float
    random_score: float
    evaluations: int
    ga_history: list[float]
    random_history: list[float]
    evaluation_history: list[int]


def random_search(people, cfg: Config, evaluations: int, seed: int, checkpoints: list[int]):
    """Sample independent random assignments, retaining the best score so far."""
    rng = Random(seed)
    indices = tuple(range(len(people)))
    best = float('-inf')
    history = []
    next_checkpoint = 0
    for evaluation in range(1, evaluations + 1):
        candidate = tuple(rng.sample(indices, len(indices)))
        best = max(best, fitness(candidate, people, cfg))
        while next_checkpoint < len(checkpoints) and evaluation == checkpoints[next_checkpoint]:
            history.append(best)
            next_checkpoint += 1
    return best, history


def run_benchmark(seeds=range(5), participants=18, cfg=None) -> list[Trial]:
    cfg = cfg or Config(population=40, generations=60)
    trials = []
    for seed in seeds:
        # Same instance per trial; independent RNG streams for the two algorithms.
        people = synthetic_participants(participants, seed=seed)
        ga = optimize(people, cfg, seed=seed + 10000)
        random_best, random_history = random_search(
            people, cfg, ga.evaluations, seed=seed + 20000,
            checkpoints=ga.evaluation_history,
        )
        trials.append(Trial(seed, ga.score, random_best, ga.evaluations,
                            ga.history, random_history, ga.evaluation_history))
    return trials


def save_plot(trials: list[Trial], path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError('Install matplotlib using: python -m pip install -e ".[viz]"') from exc
    # Evaluation checkpoints are identical across trials for fixed population/generations.
    xs = trials[0].evaluation_history
    ga_mean = [mean(t.ga_history[i] for t in trials) for i in range(len(xs))]
    random_mean = [mean(t.random_history[i] for t in trials) for i in range(len(xs))]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ga_mean, label='Genetic algorithm (mean best-so-far)')
    ax.plot(xs, random_mean, label='Random search (mean best-so-far)')
    ax.set(xlabel='Objective-function evaluations', ylabel='Fitness (higher is better)',
           title=f'GA vs random search ({len(trials)} synthetic instances)')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluation-budget-matched optimizer benchmark')
    parser.add_argument('--trials', type=int, default=5)
    parser.add_argument('--participants', type=int, default=18)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--generations', type=int, default=60)
    parser.add_argument('--output', type=Path, default=Path('results/benchmark'))
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be at least 1')
    cfg = Config(population=args.population, generations=args.generations)
    trials = run_benchmark(range(args.trials), args.participants, cfg)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'trials.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'ga_best', 'random_search_best', 'objective_evaluations', 'difference'])
        for t in trials:
            writer.writerow([t.seed, t.ga_score, t.random_score, t.evaluations, t.ga_score - t.random_score])
    with (args.output / 'convergence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['seed', 'generation', 'evaluations', 'ga_best', 'random_search_best'])
        for t in trials:
            for i, (ev, ga, rs) in enumerate(zip(t.evaluation_history, t.ga_history, t.random_history)):
                writer.writerow([t.seed, i, ev, ga, rs])
    ga_scores = [t.ga_score for t in trials]
    rs_scores = [t.random_score for t in trials]
    print(f'GA best fitness:       {mean(ga_scores):.2f} mean' + (f' (SD {stdev(ga_scores):.2f})' if len(trials)>1 else ''))
    print(f'Random search fitness: {mean(rs_scores):.2f} mean' + (f' (SD {stdev(rs_scores):.2f})' if len(trials)>1 else ''))
    print(f'GA wins: {sum(t.ga_score > t.random_score for t in trials)}/{len(trials)}')
    print('Equal objective evaluation counts per trial; results are illustrative, not proof of superiority.')
    try:
        save_plot(trials, args.output / 'convergence.png')
        print(f'Plot saved to {args.output / "convergence.png"}')
    except RuntimeError as exc:
        print(f'Plot skipped: {exc}')
    print(f'CSV results saved to {args.output}')


if __name__ == '__main__':
    main()
