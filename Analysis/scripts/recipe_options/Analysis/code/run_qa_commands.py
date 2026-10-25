"""M1 port: run guarded QA commands and bind their actual evidence and receipts."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
SELF_TEST_LINE = re.compile('SELF_TEST ([1-9][0-9]*) all-pair route reconstructions passed')

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def internal_path(analysis, relative):
    assert isinstance(relative, str) and relative, 'QA paths must be nonempty relative POSIX strings'
    part = Path(relative)
    root = Path(analysis).resolve()
    assert not part.is_absolute() and (not part.drive) and ('..' not in part.parts)
    assert '\\' not in relative and relative == part.as_posix(), 'QA paths must be relative POSIX paths'
    path = (root / part).resolve()
    assert path.is_relative_to(root), 'QA paths must stay within their own Analysis directory'
    return path

def bind_file(analysis, relative):
    path = internal_path(analysis, relative)
    assert path.is_file(), 'Missing QA evidence: ' + relative
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return {'path': relative, 'sha256': digest, 'bytes': path.stat().st_size}

def bind_files(analysis, relatives):
    assert relatives and len(set(relatives)) == len(relatives), 'QA bindings must be nonempty and unique'
    return [bind_file(analysis, name) for name in sorted(relatives)]

def assert_unchanged(analysis, entrypoint, code_inputs, inputs, artifacts=()):
    assert bind_file(analysis, entrypoint['path']) == entrypoint, 'QA entrypoint changed during execution'
    for label, rows in [('code', code_inputs), ('scientific input', inputs), ('verified artifact', artifacts)]:
        current = [bind_file(analysis, row['path']) for row in rows]
        assert current == rows, 'QA ' + label + ' changed during execution'

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def check_receipts(analysis, receipts):
    for relative, status in receipts.items():
        path = internal_path(analysis, relative)
        assert path.is_file(), 'QA command did not produce its receipt: ' + relative
        value = json.loads(path.read_text(encoding='utf-8-sig'))
        assert isinstance(value, dict) and value, 'QA receipt is not a nonempty object: ' + relative
        if status is not None:
            assert value.get('status') == status, 'QA receipt did not pass: ' + relative

def run_stage(config):
    analysis = Path(config['analysis']).resolve()
    base = Path(config['entrypoint']).parent.as_posix()
    qa_relative = base + '/qa/qa_execution.json'
    qa_path = internal_path(analysis, qa_relative)
    qa_path.parent.mkdir(parents=True, exist_ok=True)
    qa_path.unlink(missing_ok=True)
    started = time.perf_counter()
    data = {'schema': 'm1-recipe_options-qa-execution-v1', 'stage': config['stage'], 'status': 'failed', 'started_utc': utc_now(), 'completed_utc': None, 'seconds': 0.0, 'entrypoint': {}, 'code_inputs': [], 'inputs': [], 'commands': [], 'outputs': []}
    try:
        assert not set(config['inputs']) & set(config['outputs']), 'QA inputs cannot also be outputs'
        for relative in config['outputs']:
            internal_path(analysis, relative).unlink(missing_ok=True)
        data['entrypoint'] = bind_file(analysis, config['entrypoint'])
        data['code_inputs'] = bind_files(analysis, config['code_inputs'])
        data['inputs'] = bind_files(analysis, config['inputs'])
        verified_artifacts = []
        assert 'scripts/guard.py' in config['code_inputs']
        for index, spec in enumerate(config['commands']):
            assert spec['script'] in config['code_inputs']
            for relative in spec['receipts']:
                internal_path(analysis, relative).unlink(missing_ok=True)
            suffix = '_self_test' if spec['arguments'] == ['--self-test'] else ''
            log_relative = base + f"/qa/{index + 1:02d}_{Path(spec['script']).stem}{suffix}.log"
            log_path = internal_path(analysis, log_relative)
            argv = [str(Path(sys.executable).resolve()), '-u', str(internal_path(analysis, 'scripts/guard.py')), str(internal_path(analysis, spec['script'])), *spec['arguments']]
            command_started = time.perf_counter()
            command = {'script': spec['script'], 'arguments': list(spec['arguments']), 'argv': argv, 'started_utc': utc_now(), 'completed_utc': None, 'seconds': 0.0, 'exit_code': None, 'log': {}}
            data['commands'].append(command)
            try:
                completed = subprocess.run(argv, cwd=str(internal_path(analysis, spec['script']).parent), env=dict(os.environ, PYTHONIOENCODING='utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace', check=False)
                command['exit_code'] = completed.returncode
                stderr = ''.join(('[stderr] ' + line + '\n' for line in completed.stderr.splitlines()))
                log_path.write_text(completed.stdout + ('\n' if stderr and completed.stdout else '') + stderr, encoding='utf-8')
            except BaseException as error:
                log_path.write_text(type(error).__name__ + ': ' + str(error) + '\n', encoding='utf-8')
                raise
            finally:
                command['completed_utc'] = utc_now()
                command['seconds'] = time.perf_counter() - command_started
                if log_path.is_file():
                    command['log'] = bind_file(analysis, log_relative)
            if completed.returncode != 0:
                raise subprocess.CalledProcessError(completed.returncode, argv)
            assert_unchanged(analysis, data['entrypoint'], data['code_inputs'], data['inputs'], verified_artifacts)
            verified_artifacts.append(command['log'])
            if spec.get('self_test'):
                counts = [int(match[1]) for line in completed.stdout.splitlines() if (match := SELF_TEST_LINE.fullmatch(line))]
                assert len(counts) == 1, 'Expected exactly one positive SELF_TEST success line on stdout'
                proof = {'schema': 'm1-recipe_options-heldout-self-test-v1', 'status': 'passed', 'verified_at_utc': utc_now(), 'group_ids': [0, 282, 7020], 'n_all_pair_route_reconstructions': counts[0], 'verifier': bind_file(analysis, spec['script']), 'inputs': data['inputs'], 'log': command['log'], 'qa_execution_path': qa_relative, 'command_index': index}
                write_json(internal_path(analysis, base + '/qa/self_test_verification.json'), proof)
            check_receipts(analysis, spec['receipts'])
            verified_artifacts.extend((bind_file(analysis, relative) for relative in spec['receipts']))
        assert_unchanged(analysis, data['entrypoint'], data['code_inputs'], data['inputs'], verified_artifacts)
        data['outputs'] = [bind_file(analysis, relative) for relative in config['outputs']]
        data['status'] = 'passed'
    finally:
        data['completed_utc'] = utc_now()
        data['seconds'] = time.perf_counter() - started
        if data['status'] != 'passed':
            data['outputs'] = [bind_file(analysis, relative) for relative in config['outputs'] if internal_path(analysis, relative).is_file()]
        write_json(qa_path, data)
    return data
