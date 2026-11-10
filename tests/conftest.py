import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gradmit.data import load_data  # noqa: E402
from gradmit.reporting import load_outputs  # noqa: E402


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def df():
    return load_data()


@pytest.fixture(scope="session")
def outputs():
    return load_outputs()
