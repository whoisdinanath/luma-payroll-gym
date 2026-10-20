"""Regrade saved format failures after the JSON-fence parser correction."""
import argparse
import asyncio
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path

from openai import AsyncOpenAI

from luma_gym.budget import Budget
from luma_gym.hosted import load_key, read_events
from luma_gym.judge import HostedJudge, grade_answer
from luma_gym.scoring import parse_answer
from luma_gym.tasks import Task


ROOT = Path(__file__).resolve().parents[1]


def texts(directory):
    generations = directory / 'generations.jsonl'
    if generations.exists():
        return {row['id']: row['final_answer'] for row in read_events(generations)}
    found = {}
    for event in read_events(directory / 'transcript.jsonl'):
        if event.get('event') == 'output' and event.get('output'):
            completion = event['output'].get('completion') or []
            if completion and completion[-1].get('role') == 'assistant':
                found[event['id']] = completion[-1].get('content')
    return found


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    task_files = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in ROOT.glob('data/**/*.json')}
    jobs = []
    for path in sorted(ROOT.glob('results/**/scores.json')):
        rows = json.loads(path.read_text())
        answers = texts(path.parent)
        affected = []
        for row in rows:
            if (row.get('score') or {}).get('status') != 'invalid_format':
                continue
            try:
                parse_answer(answers.get(row['id']))
            except (ValueError, RecursionError):
                continue
            affected.append(row)
        if not affected:
            continue
        config = json.loads((path.parent / 'config.json').read_text())
        tasks_path = task_files.get(config.get('taskset_sha256'))
        if tasks_path is None:
            raise ValueError(f'No exact task-file match: {path}')
        tasks = {item['id']: Task.from_dict(item) for item in json.loads(tasks_path.read_text())}
        for row in affected:
            assert row['id'].rsplit(':', 1)[0] in tasks
        jobs.append((path, rows, affected, answers, tasks))
    print(f'{sum(len(job[2]) for job in jobs)} answers in {len(jobs)} runs', flush=True)
    if not args.apply:
        return
    report = []
    async with AsyncOpenAI(api_key=load_key(Path.home() / '.env', 'OPENROUTER_API_KEY'),
                           base_url='https://openrouter.ai/api/v1', timeout=90, max_retries=0) as native:
        for path, rows, affected, answers, tasks in jobs:
            backup = path.with_name('scores-before-fence-fix.json')
            if not backup.exists():
                backup.write_text(path.read_text())
            cost = 0
            with path.with_name('judge-fence-fix.jsonl').open('a') as log:
                judge = HostedJudge(native, log, Budget())
                for row in affected:
                    task = tasks[row['id'].rsplit(':', 1)[0]]
                    text = answers[row['id']]
                    previous = judge.cost
                    result = await grade_answer(task, text, judge, prompt=(ROOT / 'rules/judge.md').read_text())
                    charged = judge.cost - previous
                    cost += charged
                    old = row['score']['score']
                    row['score'] = result
                    if 'cost_usd' in row:
                        row['cost_usd'] += charged
                    report.append({'run': str(path.parent.relative_to(ROOT)), 'id': row['id'],
                                   'before': old, 'after': result['score'], 'status': result['status']})
                    # Checkpoint each completed answer so a resumed run does not repeat judge calls.
                    path.write_text(json.dumps(rows, indent=2) + '\n')
            progress_path = path.with_name('progress.jsonl')
            if progress_path.exists():
                progress = read_events(progress_path)
                lookup = {row['id']: row for row in rows}
                for event in progress:
                    if event.get('event') == 'finished' and event.get('id') in lookup:
                        updated = lookup[event['id']]
                        event['score'] = updated['score']
                        event['cost_usd'] = updated.get('cost_usd', event.get('cost_usd', 0))
                progress_path.write_text(''.join(json.dumps(event) + '\n' for event in progress))
            metrics_path = path.with_name('metrics.json')
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            scored = [row for row in rows if row.get('score')]
            rewards = [row['score']['score'] for row in scored]
            metrics.update(rubric_version='8', mean_reward=statistics.mean(rewards),
                           sd_reward=statistics.stdev(rewards) if len(rewards) > 1 else None)
            if 'pass_rate' in metrics:
                metrics['pass_rate'] = statistics.mean(row['score']['passed'] for row in scored)
            if 'passed' in metrics:
                metrics['passed'] = sum(row['score']['passed'] for row in scored)
            if 'statuses' in metrics:
                metrics['statuses'] = dict(Counter(row['score']['status'] for row in scored))
            if 'tiers' in metrics:
                for tier in metrics['tiers']:
                    values = [row['score']['score'] for row in scored if str(row['tier']) == tier]
                    if values:
                        metrics['tiers'][tier] = {'mean': statistics.mean(values),
                                                  'sd': statistics.stdev(values) if len(values) > 1 else None}
            if 'recorded_cost_usd' in metrics:
                metrics['recorded_cost_usd'] += cost
            elif 'judge_cost_usd' in metrics:
                metrics['judge_cost_usd'] += cost
            metrics_path.write_text(json.dumps(metrics, indent=2) + '\n')
            print(path.parent.name, len(affected), f'judge ${cost:.5f}', flush=True)
    (ROOT / 'reports/fence-parser-correction.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    asyncio.run(main())
