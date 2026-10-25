"""Slow (about 2 minutes on a CPU): the package reproduces the notebook exactly.
Run with: pytest tests/test_equivalence.py, or python tests/check_equivalence.py"""
import pytest

import check_equivalence


@pytest.mark.slow
def test_package_matches_notebook():
    check_equivalence.main()
