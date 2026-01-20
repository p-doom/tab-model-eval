import pytest
from src.math_utils import add, subtract


def setup_module(module):
    print("Setup")


def teardown_module(module):
    print("Teardown")


def test_add():
    # Basic addition test
    assert add(2, 2) == 5
    assert add(1, 1) == 2


def test_subtract():
    assert subtract(2, 1) == 1
