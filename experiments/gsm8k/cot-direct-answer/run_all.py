"""Detached workflow for the new group only; historical groups are read-only."""
import fcntl
import subprocess
import sys
from common import runtime as runtime_config, ROOT, load_json, utc, write_json


def main():
    (ROOT / 'logs').mkdir(exist_ok=True)
    runtime = runtime_config()
    analysis_python = runtime.get('analysis_python') or sys.executable
    with (ROOT / '.workflow.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = 'prepare'
        try:
            for current, interpreter, script in [
                ('prepare', sys.executable, 'prepare.py'),
                ('data_audit', sys.executable, 'audit_data.py'),
                ('inference', sys.executable, 'pipeline.py'),
                ('resume_check', sys.executable, 'verify_resume.py'),
                ('statistics', analysis_python, 'analyze.py'),
                ('comparison', analysis_python, 'compare.py'),
                ('independent_audit', analysis_python, 'independent_audit.py')]:
                write_json(ROOT / 'status/workflow.json', {'state': 'running', 'step': current, 'utc': utc()})
                with (ROOT / f'logs/workflow_{current}.log').open('a') as log:
                    subprocess.run([interpreter, '-B', str(ROOT / script)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            audit = load_json(ROOT / 'reports/independent_audit.json')
            assert audit['status'] == 'passed' and audit['checked_records'] == 9372
            write_json(ROOT / 'status/workflow.json', {'state': 'complete', 'formal_count': 9372, 'utc': utc()})
        except Exception as exc:
            write_json(ROOT / 'status/workflow.json', {'state': 'failed', 'step': current, 'error': repr(exc), 'utc': utc()})
            raise


if __name__ == '__main__':
    main()
