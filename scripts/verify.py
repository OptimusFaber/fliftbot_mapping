#!/usr/bin/env python3
"""Offline navigation checks, runnable both from source and installed package."""
import ast
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    for directory in ('scripts', 'src', 'launch', 'tests'):
        for path in (ROOT / directory).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    if (ROOT / 'nav.sh').is_file():
        subprocess.run(['bash', '-n', str(ROOT / 'nav.sh')], check=True)
    for path in (ROOT / 'scripts').glob('*.sh'):
        subprocess.run(['bash', '-n', str(path)], check=True)
    env = dict(os.environ, PYTHONNOUSERSITE='1',
               PYTHONPATH=os.pathsep.join((str(ROOT / 'scripts'), str(ROOT / 'src'),
                                           os.environ.get('PYTHONPATH', ''))))
    return subprocess.call([sys.executable, '-m', 'unittest', 'discover', '-s',
                            str(ROOT / 'tests'), '-p', 'test_*.py', '-v'], env=env)


if __name__ == '__main__':
    raise SystemExit(main())
