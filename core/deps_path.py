"""
Put the pylgmath / pysteam clones in deps/ on the Python path.
Import this module before importing pylgmath or pysteam.
"""
import sys
from pathlib import Path

DEPS_DIR = Path(__file__).resolve().parents[1] / "deps"

for name in ("pylgmath", "pysteam"):
    path = str(DEPS_DIR / name)
    if path not in sys.path:
        sys.path.insert(0, path)
