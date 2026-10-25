"""Run the CSV-based scientific analyses sequentially."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / 'Analysis'
SCRIPTS = ANALYSIS / 'scripts'
OUT = ANALYSIS / 'out'
STAGES = [('zero_rule', 'apply_zero_rule.py'), ('response_learning', 'response_learning/run_response_learning.py'), ('reference_learning', 'reference_learning/run_reference_learning.py'), ('recipe_options', 'recipe_options/run_recipe_options.py'), ('well_status', 'm1_well_status.py'), ('published_overlap', 'm1_published_overlap.py'), ('corpus', 'm1_corpus_reconciliation.py'), ('numbers', 'm1_numbers.py'), ('figures', 'm1_figures.py')]

def stamp():
    return datetime.now(timezone.utc).isoformat()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'analysis', 'summaries', 'figures'], default='all')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args()
    if args.list:
        print(json.dumps(STAGES, indent=2))
        return
    if sys.flags.optimize or os.environ.get('PYTHONOPTIMIZE'):
        raise ValueError('Run without Python optimization')
    data = Path(os.environ.get('M1_TABLES_DIR', ROOT / 'data')).resolve()
    exclusion = Path(os.environ.get('M1_EXCLUDE_CSV', ROOT / 'Input/EXCLUDE_wells.csv')).resolve()
    selected = STAGES if args.stage == 'all' else STAGES[:4] if args.stage == 'analysis' else STAGES[4:8] if args.stage == 'summaries' else STAGES[8:]
    OUT.mkdir(parents=True, exist_ok=True)
    logs = OUT / 'pipeline_logs'
    logs.mkdir(exist_ok=True)
    report = {'started_utc': stamp(), 'status': 'running', 'stages': [], 'workers': 1}
    destination = OUT / ('reader_reproduction_' + args.stage + '.json')

    def save():
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for name, script in selected:
        env = dict(os.environ, M1_TABLES_DIR=str(data), M1_DATA_DIR=str(data), M1_RAW_DIR=str(data), M1_EXCLUDE_CSV=str(exclusion), PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', M1_INPUT_PROFILE='recorded' if name == 'well_status' else 'processed')
        command = [sys.executable, '-u', str(SCRIPTS / 'guard.py'), str(SCRIPTS / script)]
        row = {'name': name, 'started_utc': stamp(), 'log': 'pipeline_logs/' + name + '.log'}
        report['stages'].append(row)
        save()
        started = time.perf_counter()
        print(stamp(), 'START', name, flush=True)
        flags = subprocess.NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0
        with (OUT / row['log']).open('w', encoding='utf-8') as stream:
            result = subprocess.run(command, cwd=ANALYSIS, env=env, stdout=stream, stderr=subprocess.STDOUT, creationflags=flags)
        row.update(exit_code=result.returncode, seconds=time.perf_counter() - started, completed_utc=stamp())
        row['status'] = 'passed' if result.returncode == 0 else 'failed'
        save()
        if result.returncode:
            report.update(status='failed', completed_utc=stamp())
            save()
            raise SystemExit(name + ' failed; see ' + row['log'])
        print(stamp(), 'PASS', name, flush=True)
    report.update(status='passed', completed_utc=stamp())
    save()
if __name__ == '__main__':
    main()
