"""Tests for the progress throttle standing between PhenoMe's reports and the progress bar."""

import pytest

from phenome_gui.utils.progress import ThrottledProgress


@pytest.fixture
def forwarded() -> list[tuple[int, int, str]]:
    """ The reports let through, appended to by the callback the throttle wraps. """
    return []


def run(forwarded: list, total: int, **kwargs) -> ThrottledProgress:
    """ Report every unit of a run of `total` units, the way PhenoMe does.

    Args:
        forwarded (list): The list the reports let through are collected in.
        total (int): Number of units the run reports over.
        **kwargs: Passed on to the throttle, to set the number of reports it allows.
    Returns:
        ThrottledProgress: The throttle the run was made through, for a further stage to go through it.
    """
    throttle = ThrottledProgress(lambda *report: forwarded.append(report), **kwargs)
    for current in range(1, total + 1):
        throttle(current, total, 'Computing properties')
    return throttle


def test_a_long_run_reports_a_fraction_of_its_units(forwarded: list) -> None:
    """A hundred steps, plus the opening report that announces the run has begun."""
    run(forwarded, 10_000, reports=100)

    assert len(forwarded) == 101


def test_the_steps_are_evenly_sized(forwarded: list) -> None:
    run(forwarded, 1_000, reports=10)

    assert [current for current, _, _ in forwarded] == [1, *range(101, 1_000, 100), 1_000]


def test_the_first_unit_is_always_reported(forwarded: list) -> None:
    run(forwarded, 10_000, reports=100)

    assert forwarded[0] == (1, 10_000, 'Computing properties')


def test_the_last_unit_is_always_reported(forwarded: list) -> None:
    """The step rarely divides the total, so the end has to be reported on its own."""
    run(forwarded, 999, reports=100)

    assert forwarded[-1] == (999, 999, 'Computing properties')


def test_a_run_shorter_than_the_step_reports_every_unit(forwarded: list) -> None:
    """A handful of images cannot be split a hundred ways, and needs no thinning out either."""
    run(forwarded, 6, reports=100)

    assert [current for current, _, _ in forwarded] == [1, 2, 3, 4, 5, 6]


def test_an_unknown_total_is_reported_once(forwarded: list) -> None:
    """A total of 0 spins the bar, which says the same thing however often it is repeated."""
    throttle = ThrottledProgress(lambda *report: forwarded.append(report))
    for _ in range(100):
        throttle(0, 0, 'Preparing')

    assert forwarded == [(0, 0, 'Preparing')]


def test_a_new_stage_is_reported_from_its_first_unit(forwarded: list) -> None:
    """Each stage has its own total to pace against, and its own beginning to announce."""
    throttle = run(forwarded, 1_000, reports=10)
    forwarded.clear()

    throttle(1, 500, 'Computing properties, second pass')

    assert forwarded == [(1, 500, 'Computing properties, second pass')]


def test_a_count_starting_over_is_reported(forwarded: list) -> None:
    """A stage sharing the label of the one before it is still a stage of its own."""
    throttle = run(forwarded, 1_000, reports=10)
    forwarded.clear()

    throttle(1, 1_000, 'Computing properties')

    assert forwarded == [(1, 1_000, 'Computing properties')]
