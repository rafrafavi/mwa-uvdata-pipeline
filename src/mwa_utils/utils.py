"""Contains utility functions for the MWA pipeline. Focus is on instrumentation and logging of data processing."""

import functools
import re
from functools import lru_cache
from pathlib import Path

from astropy.time import Time
from mwalib import CorrelatorContext, MetafitsContext  # noqa: F401
from pandas import DataFrame


# functions (classifiers) to extract meta data from file names
def _obtain_system_memory_in_gb() -> int:
    """Obtain the system memory in GB."""
    import psutil
    mem = psutil.virtual_memory()
    return mem.total // (1024 ** 3)

def compute_optimal_batches(size_in_gb: int, leakage_factor: int = 7, avail_mem_gb: int | None = None) -> int:
        """Estimates how many batches a file set should be split into based on the size of the file set and the available memory.

        Parameters
        ----------
        size_in_gb
            The total size of the files to be processed in GB.
        leakage_factor, optional
            How much is the maximum memory usage expected to be as a multiple of the total file size, by default 7 (empirically observed worse case)
        max_memory_gb, optional
            Maximum available memory in GB, by default None. If None, the system memory is used.

        Returns
        -------
            A divisor for splitting data into smaller batches.
        """

        assert size_in_gb > 0, "File size must be greater than 0"
        
        if avail_mem_gb is None:
            avail_mem_gb = self._obtain_system_memory_in_gb()
        
        # size_in_gb = files.get_size_mb() // 1024
        predicted_max_memory_gb = size_in_gb * leakage_factor
        if predicted_max_memory_gb <  avail_mem_gb:
            return 1
        number_of_batches = (predicted_max_memory_gb // avail_mem_gb) * 2 # -> not a linear reduction in mem
        return number_of_batches
            
def _channel_from_gpubox(gpubox: int, metafits: str | Path) -> int:
    """Get channel number from gpubox."""
    ctx = MetafitsContext(str(metafits))    
    # get the channel number from the metafits file
    channel = next(
        (c.rec_chan_number for c in ctx.metafits_coarse_chans if c.gpubox_number == gpubox),
        None,
    )
    if channel is None:
        raise ValueError(f"GPUBOX {gpubox} not found in metafits file {metafits}.")
    return channel

def channel_from_filename(filename: str | Path, metafits: str | Path | None = None) -> int | None: # is this fits and metafits only?
    """Extract channel number from filename."""
    filename = str(filename)
    # ch_token = filename.split("_")[-2] 
    # TODO: Not doing split and instead using search rather than match; Dev to check.
    match = re.search(r"(gpubox|ch)(\d+)", filename)  

    if match is None:
        raise ValueError(f"Filename {filename} does not contain channel information.")
    
    if match.group(1) == "ch":
        return int(match.group(2))
    
    # when filename contains gpubox, need metafits to get channel number
    if metafits is None:
        raise ValueError("When filename only contains gpubox, metafits must be provided.")
    
    return _channel_from_gpubox(int(match.group(2)), metafits)
     
    

def obsid_from_filename(filename: str | Path) -> str | None:
    if isinstance(filename, Path):
        stem = filename.stem
    else:
        stem = filename.rsplit(".", 1)[0]
    return stem.split("_")[0]

    
@lru_cache(maxsize=128)
def _get_fits_path_dataframe_cachable(files: frozenset[Path]) -> DataFrame:
    data = {
        "obsid": [],
        "channel": [],
        "file_path": [],
        "file_type": [],
    }
    fits_files = [f for f in files if f.suffix == ".fits"]
    metafits_file = next((f for f in files if f.suffix == ".metafits"), None)
    for file in fits_files:
        file = Path(file)
        obsid = obsid_from_filename(file.name)
        channel = channel_from_filename(file.name, metafits_file)
        data["obsid"].append(obsid)
        data["channel"].append(channel)
        data["file_path"].append(str(file))
        data["file_type"].append(file.suffix.lstrip('.'))
    
    return DataFrame(data)
    
def get_fits_path_dataframe(files: list[Path]) -> DataFrame:  
    """Create a dataframe from a list of fits/metafits files.
    
    The dataframe contains the following columns:
        - obsid
        - channel
        - file_path
        - file_type

    Parameters
    ----------
    files
        A list of file paths to fits/metafits files.

    Returns
    -------
        A pandas DataFrame containing the path information
    """
    # use frozenset to allow an lru_cache to cache the result
    return _get_fits_path_dataframe_cachable(frozenset(files))
    

@lru_cache(maxsize=128)
def get_channel_df(ctx: MetafitsContext) -> DataFrame:
    """Get channel information from metafits as a DataFrame."""
    header = [
        "gpubox_number",
        "rec_chan_number",
        "chan_start_hz",
        "chan_centre_hz",
        "chan_end_hz",
    ]
    return DataFrame(
        {h: [getattr(c, h) for c in ctx.metafits_coarse_chans] for h in header}
    )


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
