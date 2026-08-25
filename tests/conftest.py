"""Shared pytest configuration for the bdpy test suite."""

from __future__ import annotations

import pytest

#: Tests carrying this marker read an external fMRIPrep dataset that is not part
#: of the repository, so they are deselected unless explicitly asked for.
REAL_DATA_MARKER = "real_data"


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect real-data tests unless the marker expression asks for them.

    Doing this here rather than through ``addopts = ["-m", "not real_data"]``
    matters: a marker expression on the command line replaces the one from
    addopts rather than combining with it, so an unrelated ``-m "not slow"``
    would silently re-enable the real-data tests. That is not a request for
    them, but ``RealDatasetMixin`` treats reaching setUpClass as exactly that
    and raises when the fixture is missing — which would turn a routine
    ``pytest tests -m "not slow"`` into a failure on any machine without the
    multi-GB download.

    Asking for the tests means naming the marker: ``-m real_data`` or
    ``-m real_data_quick``. Both contain the marker name, and pytest applies
    its own filtering afterwards, so this hook simply steps aside.
    """
    markexpr = config.getoption("markexpr", default="") or ""
    if REAL_DATA_MARKER in markexpr:
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        target = deselected if item.get_closest_marker(REAL_DATA_MARKER) else selected
        target.append(item)

    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
