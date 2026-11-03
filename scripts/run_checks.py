#!/usr/bin/env python3
"""Run the actual source, toy and PDF checks and save their command log."""
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    commands=[
        [sys.executable, 'scripts/inspect_sources.py'],
        [sys.executable, 'scripts/toy_assignment_demo.py'],
        [sys.executable, 'scripts/build_report.py'],
    ]
    log=['Track B reproduction command log',
         'These commands check source and toy mechanisms; they do not run detectors.',
         f'Started UTC: {datetime.datetime.now(datetime.timezone.utc).isoformat()}',
         f'Python: {sys.version}',f'Platform: {platform.platform()}','']
    results=[]
    for cmd in commands:
        started=time.monotonic()
        result=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True)
        elapsed=time.monotonic()-started
        log.extend(['$ '+' '.join(cmd),result.stdout.rstrip(),result.stderr.rstrip(),
                    f'exit_code={result.returncode}; elapsed_seconds={elapsed:.3f}',''])
        results.append({'command':cmd,'exit_code':result.returncode,'elapsed_seconds':elapsed,
                        'stdout':result.stdout,'stderr':result.stderr})
        print(result.stdout,end='')
        if result.returncode:
            (ROOT/'artifacts/run_log.txt').write_text('\n'.join(log))
            raise SystemExit(result.returncode)
    pdf=ROOT/'report/2D_Object_Detection_Track_B.pdf'
    (ROOT/'artifacts/run_log.txt').write_text('\n'.join(log))
    (ROOT/'artifacts/validation.json').write_text(json.dumps({
        'kind':'source_toy_and_document_validation_not_detector_evaluation',
        'commands':results,'report_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
        'report_bytes':pdf.stat().st_size},indent=2)+'\n')

if __name__=='__main__': main()
