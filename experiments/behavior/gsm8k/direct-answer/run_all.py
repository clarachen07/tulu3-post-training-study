"""Unattended server workflow; independent of an SSH session or local client."""
import fcntl
import subprocess
import sys
from common import ROOT, load_json, utc, write_json


def main():
    (ROOT / 'logs').mkdir(exist_ok=True)
    report_python = ROOT / '.report-env/bin/python'
    assert report_python.exists()
    with (ROOT / '.workflow.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        current = 'prepare'
        try:
            for current, interpreter, filename in [
                ('prepare', sys.executable, 'prepare.py'),
                ('data_audit', str(report_python), 'audit_data.py'),
                ('inference', sys.executable, 'pipeline.py'),
                ('resume_check', sys.executable, 'verify_resume.py'),
                ('statistics', str(report_python), 'analyze.py'),
                ('comparison', str(report_python), 'compare.py'),
                ('independent_audit', str(report_python), 'independent_audit.py'),
                ('report', str(report_python), 'render_report.py')]:
                write_json(ROOT / 'status/workflow.json', {'state': 'running', 'step': current, 'utc': utc()})
                with (ROOT / f'logs/workflow_{current}.log').open('a') as log:
                    subprocess.run([interpreter, '-B', str(ROOT / filename)], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert load_json(ROOT / 'reports/independent_audit.json')['checked_records'] == 9372
            write_json(ROOT / 'status/workflow.json', {'state': 'complete', 'formal_count': 9372,
                       'report': 'reports/report.html', 'utc': utc()})
        except Exception as exc:
            write_json(ROOT / 'status/workflow.json', {'state': 'failed', 'step': current, 'error': repr(exc), 'utc': utc()})
            raise


if __name__ == '__main__':
    main()
