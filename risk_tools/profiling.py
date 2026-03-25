# -*- coding: utf-8 -*-

"""
Geoffrey's note: I'm probably giving Chat a bit too much autonomy here, but this
doesnt seem like a half bad idea. Well see how it goes...


ChatGPT:
Profiling utilities for the risk_tools package.

This module provides lightweight timing helpers for profiling the
execution time of key workflow steps. The tools are intended to make the
performance of package functions easier to interpret than a raw profiler
trace, especially when most of the heavy work is delegated to external
libraries such as pandas, NumPy, and SciPy.

The module currently provides three tools:

- ``timed``:
  function decorator for reporting total runtime of a function call
- ``timed_block``:
  context manager for timing an individual named code block
- ``ProfileAccumulator``:
  helper class for aggregating repeated timing measurements and printing
  a summary report

These tools are primarily intended for development and performance
diagnostics rather than production logging.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from time import perf_counter
from typing import Any


PROFILE_ENABLED = True


def timed(func: Callable) -> Callable:
    """
    Decorate a function so that its total execution time is printed.

    Parameters
    ----------
    func : callable
        Function to decorate.

    Returns
    -------
    callable
        Wrapped function that prints elapsed runtime when profiling is
        enabled.

    Notes
    -----
    This decorator is best suited to top-level functions whose total
    runtime is of interest. For finer-grained timing inside a function,
    use ``timed_block`` or ``ProfileAccumulator``.

    Examples
    --------
    >>> @timed
    ... def add(a, b):
    ...     return a + b
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not PROFILE_ENABLED:
            return func(*args, **kwargs)

        start = perf_counter()
        result = func(*args, **kwargs)
        elapsed = perf_counter() - start
        print(f"[TIMER] {func.__name__} took {elapsed:.4f} s")
        return result

    return wrapper


@contextmanager
def timed_block(name: str) -> Iterator[None]:
    """
    Time a named block of code and print its elapsed time once.

    Parameters
    ----------
    name : str
        Name used to identify the timed block in the printed output.

    Yields
    ------
    None
        Context manager for timing a code block.

    Notes
    -----
    This helper is useful for one-off block timing. If the same code
    block is executed repeatedly in a loop, ``ProfileAccumulator`` is
    usually a better choice because it aggregates timings and avoids
    excessive console output.

    Examples
    --------
    >>> with timed_block("example step"):
    ...     x = sum(range(100))
    """
    if not PROFILE_ENABLED:
        yield
        return

    start = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - start
        print(f"[TIMER] {name} took {elapsed:.4f} s")


class ProfileAccumulator:
    """
    Accumulate and summarize timing information for repeated code blocks.

    This class is designed for use when the same operation is performed
    many times, such as fitting a distribution repeatedly inside a
    bootstrap loop. Instead of printing one timing line per iteration,
    timings are accumulated by block name and reported once at the end.

    Attributes
    ----------
    times : collections.defaultdict
        Total accumulated runtime for each named block.
    counts : collections.defaultdict
        Number of times each named block has been timed.

    Methods
    -------
    time_block(name)
        Context manager that records elapsed time under the given name.
    report()
        Print a summary of total and average times for all recorded
        blocks.
    """

    def __init__(self) -> None:
        """Initialize empty timing accumulators."""
        self.times: defaultdict[str, float] = defaultdict(float)
        self.counts: defaultdict[str, int] = defaultdict(int)

    @contextmanager
    def time_block(self, name: str) -> Iterator[None]:
        """
        Time a named code block and add the result to the accumulator.

        Parameters
        ----------
        name : str
            Name used to group timing measurements.

        Yields
        ------
        None
            Context manager for timing a code block.

        Examples
        --------
        >>> prof = ProfileAccumulator()
        >>> with prof.time_block("step"):
        ...     x = sum(range(100))
        >>> prof.report()
        """
        if not PROFILE_ENABLED:
            yield
            return

        start = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - start
            self.times[name] += elapsed
            self.counts[name] += 1

    def report(self) -> None:
        """
        Print a summary report of accumulated timings.

        The report is sorted in descending order of total time spent in
        each named block.

        Returns
        -------
        None
            This method prints the timing summary to standard output.
        """
        if not PROFILE_ENABLED:
            return

        if not self.times:
            print("[PROFILE] No timing data recorded.")
            return

        print("[PROFILE] Timing summary:")
        for name in sorted(self.times, key=self.times.get, reverse=True):
            total = self.times[name]
            count = self.counts[name]
            avg = total / count if count else 0.0
            print(
                f"[PROFILE] {name}: total={total:.4f} s, "
                f"calls={count}, avg={avg:.4f} s"
            )