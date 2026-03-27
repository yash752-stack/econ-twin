# tests/conftest.py
# Shared pytest configuration for econ-twin test suite.
# Run from the repo root: pytest tests/ -v

import sys
import os

# Ensure repo root is on path so modules can import data files with relative paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
