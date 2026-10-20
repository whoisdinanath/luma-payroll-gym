"""Summarize recorded GPU work from the evidence files included in the repository."""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = subprocess.check_output(['git', 'ls-files', '-z', 'results'], cwd=ROOT).decode().split('\0')
    sources = []
    for name in paths:
        path = ROOT / name
        if path.name == 'steps.jsonl' and (path.parent.name.startswith('rl-') or path.parent.name == 'sft'):
            category = 'sft' if path.parent.name == 'sft' else 'rl'
        elif path.name == 'generations.jsonl':
            category = 'evaluation'
        else:
            continue
        seconds = 0
        for line in path.read_text().splitlines():
            row = json.loads(line)
            value = row.get('seconds', 0)
            seconds += sum(value.values()) if isinstance(value, dict) else value
        sources.append({'path': name, 'category': category, 'seconds': seconds})
    report = {
        'gpu_hours': sum(row['seconds'] for row in sources) / 3600,
        'hours_by_category': {key: sum(row['seconds'] for row in sources if row['category'] == key) / 3600
                              for key in ('rl', 'sft', 'evaluation')},
        'sources': sources,
    }
    (ROOT / 'reports/compute-summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'sources'}, indent=2))


if __name__ == '__main__':
    main()
