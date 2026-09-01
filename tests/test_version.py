"""Tests for the package version, which the release pipeline depends on.

The version is hand-bumped in pyproject.toml and read back at runtime from the
installed distribution metadata. These tests pin that contract: the two must
agree, so the tag the CI creates matches what an installed copy reports.
"""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import phenome_gui

PYPROJECT = Path(__file__).parent.parent / 'pyproject.toml'


@pytest.fixture(scope='module')
def declared_version() -> str:
    """Return the static version declared in pyproject.toml.

    Returns:
        The value of ``project.version``.
    """
    with PYPROJECT.open('rb') as handle:
        return tomllib.load(handle)['project']['version']


def test_version_is_static(declared_version: str) -> None:
    """pyproject.toml must declare a static version, not a dynamic one.

    The `version-bump` CI job compares this value across branches, so a dynamic
    version would make the check unenforceable.
    """
    with PYPROJECT.open('rb') as handle:
        project = tomllib.load(handle)['project']

    assert 'version' not in project.get('dynamic', [])
    assert declared_version


def test_version_is_exposed() -> None:
    """The package exposes a __version__ resolved from installed metadata."""
    assert phenome_gui.__version__ == version('phenome-gui')


def test_version_matches_pyproject(declared_version: str) -> None:
    """The runtime version agrees with the version declared in pyproject.toml."""
    assert phenome_gui.__version__ == declared_version


def test_version_falls_back_when_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """__version__ degrades to 'unknown' rather than raising when uninstalled."""
    import importlib

    def raise_not_found(name: str) -> str:
        raise PackageNotFoundError(name)

    monkeypatch.setattr('importlib.metadata.version', raise_not_found)
    reloaded = importlib.reload(phenome_gui)

    try:
        assert reloaded.__version__ == 'unknown'
    finally:
        monkeypatch.undo()
        importlib.reload(phenome_gui)
