"""Contains utility functions for the MWA pipeline. Focus is on instrumentation and logging of data processing."""

import functools
from pathlib import Path

from astropy.time import Time
from mwalib import MetafitsContext
from pandas import DataFrame


def get_channel_df(ctx: MetafitsContext) -> DataFrame:
    """Get channel information from metafits as a DataFrame."""
    header = [
        "gpubox_number",
        "rec_chan_number",
        "chan_start_hz",
        "chan_centre_hz",
        "chan_end_hz",
    ]
    df = DataFrame(
        {h: [getattr(c, h) for c in ctx.metafits_coarse_chans] for h in header}
    )
    return df


def get_antenna_df(ctx: MetafitsContext) -> DataFrame:
    """Get antenna information from metafits as a DataFrame."""
    header = [
        "ant",
        "tile_id",
        "tile_name",
        "electrical_length_m",
        "east_m",
        "north_m",
        "height_m",
    ]
    df = DataFrame({h: [getattr(a, h) for a in ctx.antennas] for h in header})
    df["flagged"] = [a.rfinput_x.flagged | a.rfinput_y.flagged for a in ctx.antennas]
    # get elements from antenna rfinput_x, assuming it's the same as rfinput_y
    rfheader = ["rec_number", "flavour", "has_whitening_filter"]
    for h in rfheader:
        df[h] = [getattr(a.rfinput_x, h) for a in ctx.antennas]
    # rec_type is "ReceiverType.RRI", I want just "RRI"
    df["rec_type"] = [
        str(a.rfinput_x.rec_type).replace("ReceiverType.", "") for a in ctx.antennas
    ]
    return df


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
