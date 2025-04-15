"""Contains utility functions for the MWA pipeline. Focus is on instrumentation and logging of data processing."""

import functools
from pathlib import Path

from astropy.time import Time


@functools.lru_cache(maxsize=128)
def disk_usage_in_blocks(path: Path, block_size=1024*1024) -> int:
    """
    Calculate the disk usage of a given path in blocks of specified size.

    Args:
        path: The path to check disk usage for (file or directory).
        block_size: The size of each block in bytes.

    Returns
    -------
        The total number of blocks used by the file or all files in the directory.
    """
    if path.is_file():
        return path.stat().st_size // block_size

    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) // block_size


def display_time(t: Time) -> str:
    """Display the iso time, gps, unix and jd all on a single line."""
    return f"({t.isot} gps={t.gps:13.2f} unix={t.unix:13.2f} jd={t.jd:14.6f})"
