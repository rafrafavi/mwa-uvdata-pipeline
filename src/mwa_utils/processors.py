"""Sky Subtraction Data Reader wraps SSINS to manage and simplify data reading.

Its capabilities include optimizing data read pipeline based on input files and
automatically batching requests to manage peak memory usage.

"""

from functools import lru_cache, wraps
from pathlib import Path
from typing import Collection, Literal, Protocol

import numpy as np
from pyuvdata import UVData

from .configurators import UVDataFileSet
from .utils import (
    MetafitsContext,
    compute_optimal_batches,
    get_channel_df,
    get_fits_path_dataframe,
)

DEFAULT_READERS = []


def default_reader(reader):
    """Add a reader to the default readers list.

    Defacto: all defined readers could be
    default readers.
    """

    @wraps(reader)
    def wrapper(*args, **kwargs):
        return reader(*args, **kwargs)

    DEFAULT_READERS.append(reader)
    return wrapper



class UVDataFileProcessor(Protocol):
    """Protocol that defines a file processing strategy.
    The processor should be able to handle a specific file type and process it
    accordingly.
    """
    def can_handle(self, extensions: Collection) -> bool:
        """Check if the processor can handle one or more extension.
        
        Args:
        ----
            extensions: A collection of file extensions to check (usually in the form of file groups)
        """
        ...
    
    def validate(self, config: UVDataFileSet, on_error: Literal["raise", "warn", "ignore"] = "raise") -> bool:
        """Validate the configuration for processing."""
        ...
    
    def read(self, uvdata: UVData, file_set: UVDataFileSet, step_size: int = 4) -> UVData:
        """Read the data into a uvdata object using the given config."""
        ...
        
        


class FITSProcessor(UVDataFileProcessor):
    """Processor for FITS files."""

    def can_handle(self, extension: str | UVDataFileSet | Collection)  -> bool:
        """Check if the processor can handle file or file sets."""
        if isinstance(extension, str):
            return extension == "fits"
        if isinstance(extension, UVDataFileSet):
            return extension.has_fits # type: ignore
        return "fits" in extension

    # def validate(self, config: UVDataConfig) -> bool:
    # #     """Validate the configuration for processing FITS files."""
    # #     # Add validation logic specific to FITS files here
    # #     return True

    # TODO: Add logging (perf/memory)
    # TODO: Add splitting by channel
    # TODO: Refactor into specific functions

    
    def _validate_all_metafits_are_for_the_same_channels(self, metafits: list[str] | str, errors: list) -> bool:
        """Check if the metafits channels are the same across all metafits files."""
        # Implement logic to check if metafits channels are the same
        if isinstance(metafits, str):
            metafits = [metafits]
        
        if len(metafits) == 1:
            return True # if there is only one metafits, must be consistent 
            
        
        for m1, m2 in zip(metafits[:-1], metafits[1:]):
            df1 = get_channel_df(MetafitsContext(m1))
            df2 = get_channel_df(MetafitsContext(m2))
            if not df1.equals(df2):
                errors.append(f"Channels do not match between {m1} and {m2}.")
                return False
        
        return True
    
    def _validate_config_has_metafits(self, config: UVDataFileSet, errors: list) -> bool:
        """Check if the metafits files are present."""
        if not config.has_metafits:
            errors.append("No metafits files found.")
            return False
        return True
    
    @lru_cache(maxsize=128)
    def group_files_by_channel(self, metafits: list[Path] | str) -> dict[str, list[str]]:
        """Group metafits files by their channel."""
        if isinstance(metafits, str):
            metafits = [Path(metafits)]
        return {str(m): [str(m)] for m in metafits}
    
    def _validate_channels_identified_in_all_files(self, metafits: list[Path] | str | Path, fits: list[Path] | str, errors: list) -> bool:
        if isinstance(metafits, (str, Path)):
            metafits = [Path(metafits)]
        try:
            get_fits_path_dataframe(fits + metafits) # type: ignore
        except ValueError as e:
            errors.append(f"Error validating channels: {str(e)}")
            return False
        return True
    
    def validate(self, config: UVDataFileSet, on_error: Literal["raise", "warn", "ignore"] = "raise") -> bool:
        """Perform validation functions on the supplied files"""
        errors = []
        is_valid = all([
            self._validate_config_has_metafits(config, errors),
            self._validate_all_metafits_are_for_the_same_channels(config.metafits, errors), # type: ignore
            self._validate_channels_identified_in_all_files(config.metafits, config.fits, errors), # type: ignore
        ]) 
        if is_valid:
            return True
        
        if on_error == "raise":
            raise ValueError("Validation failed with the following errors:\n"
                             f"{', '.join(errors)}")
        elif on_error == "warn":
            print("Validation warnings:\n" + f"{', '.join(errors)}")
            return False
        return False
    

    


    def _batched_read(self, 
                      uvd: UVData, 
                      file_set: UVDataFileSet,
                      metafits: str | Path, 
                      files: list[str | Path], 
                      step_size: int | Literal['dynamic'] = 'dynamic') -> UVData:
        """Read the data in batches to optimize memory usage."""
        # Implement the reading logic for FITS files here

        uvd_ = type(uvd)()
        uvd_.read([metafits, *files], read_data=False)
        possible_times = np.unique(uvd.time_array)
        if step_size == 'dynamic':
            step_size = len(possible_times) // compute_optimal_batches(file_set.get_size_mb() // 1024)
        del uvd_
        for i in range(0, len(possible_times), step_size):
            uvd += type(uvd).from_file([metafits, *files], 
                                       read_data=True, 
                                       times=possible_times[i:i+step_size])
        return uvd
    
    def read(self, uvd: UVData, file_set: UVDataFileSet) -> UVData:
        """Read the data into a uvdata object using the given config."""
        # Implement the reading logic for FITS files here
        fits_files = file_set.fits
        metafits_files = file_set.metafits
        
        
        # TBD: midway through refactor!

            
        
        for obsid, metafits, files in file_set.observations():
            uvd_ = type(uvd)()
            uvd_.read([metafits, *files], read_data=False, **file_set.kwargs_for_read)
            possible_times = np.unique(uvd.time_array)
            del uvd_
            for i in range(0, len(possible_times), step_size):
                uvd += type(uvd).from_file([metafits, *files], 
                                           read_data=True, 
                                           times=possible_times[i:i+step_size],
                                           **file_set.kwargs_for_read)
                
            
        return uvd