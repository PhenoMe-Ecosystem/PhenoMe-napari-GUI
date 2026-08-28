"""Progress reporting paced for the GUI.

PhenoMe reports a property after every single image, which the GUI cannot keep up with: each report
crosses the thread boundary as a queued signal and repaints the progress bar, so a dataset of any size
spends more time drawing the bar than computing what it measures. The throttle below stands between the
two and lets only a fraction of the reports through, in steps big enough to be worth drawing.

The progress callback is also the only place a running task can be called off from: PhenoMe offers no
other hook into its loops, and the thread it runs on cannot be killed. StoppableProgress raises out of
the callback, which unwinds the loop at a point where the checkpoint has just been written.
"""

import threading
from logging import getLogger

from phenome.utils import ProgressCallback

logger = getLogger(__name__)

# How many times a run is to report, at most. A hundred steps is one per percent, which is as fine as a
# progress bar reads anyway, and leaves the reporting cost flat whether there are ten images or a million.
PROGRESS_REPORTS = 100


class TaskStopped(Exception):
    """ Raised out of a progress callback to unwind the task that called it.

    Reads as a request rather than a failure: the caller asked for it, and the checkpoint holds
    everything the run had committed by then, so the step resumes from where it stopped.

    Derives from Exception rather than BaseException on purpose. The napari worker only catches the
    former, and one it does not catch escapes into the Qt thread pool instead of reaching the panel.
    """


class ThrottledProgress:
    """ A progress callback that reports in bigger steps than the one it wraps.

    Reads as a `phenome.utils.ProgressCallback` and forwards to one, so it can be handed to
    PhenoMe wherever a plain callback would be. Holds the count of the last report it let through,
    which makes an instance good for one run only.

    The first report of a stage always goes through, so the bar leaves its "Preparing..." state as
    soon as there is something to show, and so does the last one, so it never stops short of the end.
    """

    def __init__(self, callback: ProgressCallback, reports: int = PROGRESS_REPORTS) -> None:
        """ Wrap a progress callback.

        Args:
            callback (ProgressCallback): The hook to forward to, called as (current, total, desc).
            reports (int): How many reports to let through over a stage, at most. One is the fewest
                that makes sense, being the first report alone.
        """
        self._callback = callback
        self._reports = max(1, reports)
        self._last_reported = None
        self._last_desc = None

    def __call__(self, current: int, total: int, desc: str = '') -> None:
        """ Take a report from PhenoMe, and forward it if enough has happened since the last one.

        Args:
            current (int): Units completed so far. Batches for embedding extraction, images for properties.
            total (int): Expected total number of units. PhenoMe documents 0 as "unknown".
            desc (str): Short stage label, e.g. "Processing batches".
        """
        # A new label, or a count starting over, is a stage of its own: it has its own total to pace
        # against, and its first report is what tells the bar the stage has begun.
        if desc != self._last_desc or self._last_reported is None or current < self._last_reported:
            self._last_desc = desc
            self._report(current, total, desc)
            return

        # Nothing to pace against, and nothing to show either: the bar spins on the report already
        # forwarded above, and every further one would say exactly the same thing.
        if total <= 0:
            return

        step = max(1, total // self._reports)
        if current >= total or current - self._last_reported >= step:
            self._report(current, total, desc)

    def _report(self, current: int, total: int, desc: str) -> None:
        """ Forward one report, and remember the count it was made at.

        Args:
            current (int): Units completed so far.
            total (int): Expected total number of units.
            desc (str): Short stage label.
        """
        self._last_reported = current
        self._callback(current, total, desc)


class StoppableProgress:
    """ A progress callback that raises TaskStopped once it is asked to stop.

    Reads as a `phenome.utils.ProgressCallback` and forwards to one, so it can be handed to PhenoMe
    wherever a plain callback would be. PhenoMe reports after every batch and every image, each time
    having just written what it computed to the checkpoint, so raising from here unwinds the run at a
    point it can be picked up from.

    Wrap the throttle in this rather than the other way around: the stop is read on every report
    PhenoMe makes, while the throttle only lets a hundredth of them through and would hold a stop
    until the next step of the bar.
    """

    def __init__(self, callback: ProgressCallback, stop_event: threading.Event) -> None:
        """ Wrap a progress callback, to be called until the event is set.

        Args:
            callback (ProgressCallback): The hook to forward to, called as (current, total, desc).
            stop_event (threading.Event): Set from the thread asking the task to stop, and read here
                from the thread running it.
        """
        self._callback = callback
        self._stop_event = stop_event

    def __call__(self, current: int, total: int, desc: str = '') -> None:
        """ Take a report from PhenoMe, and forward it unless the task has been asked to stop.

        Args:
            current (int): Units completed so far. Batches for embedding extraction, images for properties.
            total (int): Expected total number of units. PhenoMe documents 0 as "unknown".
            desc (str): Short stage label, e.g. "Processing batches".

        Raises:
            TaskStopped: When the task has been asked to stop, which unwinds it out of PhenoMe.
        """
        if self._stop_event.is_set():
            logger.info('Stopping after %d of %d units of %s.', current, total, desc or 'the run')
            raise TaskStopped("Task interrupted by user")

        self._callback(current, total, desc)
