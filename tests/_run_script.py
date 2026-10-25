"""Run one of scripts/*.py with the test setup: python tests/_run_script.py SCRIPT [ARGS...]"""
import os
import runpy
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _offline  # noqa: E402

_offline.patch()
script, sys.argv = sys.argv[1], sys.argv[1:]
runpy.run_path(script, run_name="__main__")
