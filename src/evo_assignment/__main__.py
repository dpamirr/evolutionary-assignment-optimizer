"""CLI for a fully synthetic experiment."""
import argparse
import csv
from pathlib import Path
from .core import Config, groups, optimize, synthetic_participants


def main() -> None:
    parser = argparse.ArgumentParser(description='Synthetic event grouping with a genetic algorithm')
    parser.add_argument('--participants', type=int, default=24)
    parser.add_argument('--group-size', type=int, default=3)
    parser.add_argument('--generations', type=int, default=160)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('results'))
    args = parser.parse_args()
    people = synthetic_participants(args.participants, args.seed)
    result = optimize(people, Config(group_size=args.group_size, generations=args.generations), args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'participants.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'x', 'y', 'diet', 'age', 'interests'])
        for p in people:
            writer.writerow([p.id, p.x, p.y, p.diet, p.age, ';'.join(sorted(p.interests))])
    with (args.output / 'groups.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['group', 'participant_id'])
        for num, team in enumerate(groups(result.chromosome, args.group_size), 1):
            for idx in team:
                writer.writerow([num, people[idx].id])
    with (args.output / 'convergence.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['generation', 'best_fitness'])
        writer.writerows(enumerate(result.history))
    print(f'Random baseline: {result.baseline_score:.2f}')
    print(f'GA best:         {result.score:.2f}')
    print(f'CSV files saved to {args.output}')


if __name__ == '__main__':
    main()
