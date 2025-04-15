"""Sky Subtraction Data Reader wraps SSINS to manage and simplify data reading.

Its capabilities include optimizing data read pipeline based on input files and
automatically batching requests to manage peak memory usage.

"""

from functools import wraps
from typing import Collection, Protocol

import numpy as np
from pyuvdata import UVData

from .configurators import UVDataFileSet

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
    
    # def validate(self, config: UVDataConfig) -> bool:
    #     """Validate the configuration for processing."""
    #     ...
    
    def read(self, uvdata: UVData, file_set: UVDataFileSet, step_size: int = 4) -> UVData:
        """Read the data into a uvdata object using the given config."""
        ...
        
        


class FITSProcessor(UVDataFileProcessor):
    """Processor for FITS files."""

    def can_handle(self, extensions: Collection) -> bool:
        """Check if the processor can handle FITS files."""
        return "fits" in extensions

    # def validate(self, config: UVDataConfig) -> bool:
    # #     """Validate the configuration for processing FITS files."""
    # #     # Add validation logic specific to FITS files here
    # #     return True

    # TODO: Add logging (perf/memory)
    # TODO: Add splitting by channel
    # TODO: Refactor into specific functions

    def read(self, uvd: UVData, file_set: UVDataFileSet, step_size: int = 4) -> UVData:
        """Read the data into a uvdata object using the given config."""
        # Implement the reading logic for FITS files here
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