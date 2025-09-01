# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains "src" and "tests")
sys.path.insert(0, str(ROOT))               # put project root at the *front* of sys.path

