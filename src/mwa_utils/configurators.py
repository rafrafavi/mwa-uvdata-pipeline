import sys
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, ClassVar, Literal, Type

from .utils import disk_usage_in_blocks


class FileTypeMeta(type):
    """Class used to dynamically instatiate file type methods based on the supported types"""
    def __new__(metacls: Type[type], name: str, bases: tuple, attrs: dict) -> type:
        cls = super().__new__(metacls, name, bases, attrs)
        supported = getattr(cls, "supported_types", set())
        for file_type in supported:
            setattr(cls, f"has_{file_type}", property(metacls.make_has_type_method(file_type)))
            setattr(cls, file_type, property(metacls.make_get_type_method(file_type)))
        return cls
    
    @staticmethod
    def make_has_type_method(file_type: str) -> Callable:
        """Create a method that checks if a file type is present."""
        def has_type(self) -> bool:
            return file_type in self.file_groups
        has_type.__name__ = f"has_{file_type}"
        has_type.__doc__ = f"Check if the file type {file_type} is present."
        return has_type

    @staticmethod
    def make_get_type_method(file_type: str) -> Callable:
        """Create a method that returns the file type."""
        def get_type(self) -> list[Path]:
            return self.file_groups[file_type]
        get_type.__name__ = file_type
        get_type.__doc__ = f"Get the files of type {file_type}."
        return get_type
        
@dataclass
class UVDataFileSet(metaclass=FileTypeMeta):
    """Base configuration for all UVData Processing."""

    supported_types: ClassVar[set[str]] = {"fits",
                                           "metafits",
                                           "ms",
                                           "uvfits",
                                           "uvf", # is an alias of uvfits, TODO: mechanism for alias...
                                           "uvh5"}

    # Input files
    files: list[str] = field(default_factory=list)
    file_groups: dict[str, list[Path]] = field(init=False)
    obsid_groups: dict[str, dict[str, list[Path]]] | None = field(init=False)


    # TODO: dynamically generate from supported types 
    # TODO: ensure you update __dir__ accordingly
    # Provided file types
    # has_fits: bool = field(default=False, init=False)
    # has_metafits: bool = field(default=False, init=False)
    # has_ms: bool = field(default=False, init=False)
    # has_uvfits: bool = field(default=False, init=False)
    # has_uvh5: bool = field(default=False, init=False)

    # SS.read options
    diff: bool = True
    flag_init: bool = True
    remove_coarse_band: bool = False
    correct_van_vleck: bool = False
    remove_flagged_ants: bool = True
    flag_choice: Literal["original"] | None = None

    # SS.select options
    sel_ants: list[str] = field(default_factory=list)
    skip_ants: list[str] = field(default_factory=list)
    sel_pols: list[str] = field(default_factory=list)
    freq_range: list[float] | None = None
    time_limit: int | None = None

    # Common options
    suffix: str = ""
    debug: bool = True
    cmap: str = "viridis"

    # SSINS.INS options
    spectrum_type: Literal["all", "auto", "cross"] = "cross"

    # SSINS.MF options
    threshold: float = 5.0
    narrow: float = 7.0
    streak: float = 8.0
    tb_aggro: float = 0.6

    # Plotting options
    plot_type: Literal["spectrum", "sigchain", "flags"] = "spectrum"
    fontsize: int = 8
    export_tsv: bool = False

    def __post_init__(self):
        """Validate arguments after initialization."""
        if not self.files:
            raise ValueError("No input files specified")

        self.file_groups = self.group_files_by_extension(self.files)

        # Determine composition of files
        # self.has_fits = "fits" in self.file_groups
        # self.has_metafits = "metafits" in self.file_groups
        # self.has_ms = "ms" in self.file_groups
        # self.has_uvfits = "uvfits" in self.file_groups
        # self.has_uvh5 = "uvh5" in self.file_groups


        # Add obsid groups for fits processing
        if self.has_fits:
            self.obsid_groups = self.group_files_by_obsid_and_extension(self.files)
        # Handle incompatible options

        
        if errors := self.validate():
            raise ValueError("Validation errors:\n" + "\n".join(errors))

        # Fill in derived settings
        # Note: currently duplicating logic from the parser in the demo
        if self.spectrum_type != "all" and not self.suffix:
            self.suffix = f".{self.spectrum_type}"
        if self.diff:
            self.suffix = f".diff{self.suffix}"
        if len(self.sel_ants) == 1:
            self.suffix += f".{self.sel_ants[0]}"
        elif len(self.skip_ants) == 1:
            self.suffix += f".no{self.skip_ants[0]}"
        if len(self.sel_pols) == 1:
            self.suffix += f".{self.sel_pols[0]}"

    def validate(self) -> list[str]:
        """Validate the configuration.

        Designed to enumerate all common errors at once rather than one at a time.

        Recommend raising a ValueError if returned list is truthy (len > 0).


        """
        errors = []
        # Implement any validation logic here
        # Validate file types:
        if not any([getattr(self, f'has_{ext}') for ext in self.supported_types]):
            errors.append("No supported file types found. Supported types are: "
                          f"{', '.join(self.supported_types)}")

        # TODO: to be refactored
        if self.has_fits and not self.has_metafits:
            errors.append("FITS files require metafits files to be present.")

        # TODO refactor into validation
        elif (self.has_fits
              and self.obsid_groups is not None
              and not self._has_metafits_for_obs_id(self.obsid_groups)):
            errors.append(
                "Metafits files are missing for some obsids."
            )

        if unsupported_types := (set(self.file_groups.keys())
                                 - set(self.supported_types)):
            errors.append(
                f"Unsupported file types found: {', '.join(unsupported_types)}"
            )

        # TODO: refactor into suitable processor
        if self.has_uvfits and self.has_uvh5:
            errors.append("Cannot use both uvfits and uvh5 files.")

        
        if self.has_ms and (self.has_uvh5 or self.has_uvfits):
            errors.append("Cannot use both ms and uvfits/uvh5 files.")

        # TODO: Check, a bit of an odd duck here:
        if self.sel_ants and self.skip_ants:
            errors.append("Cannot specify both sel_ants and skip_ants.")


        if errors:
            errors.append(f"Validation failed for: {self.files}")
        return errors


    def __getattr__(self, item) -> bool | list[Path]:
        """Dynamically create bool attributes for supported file types"""
        
        # is it a file group?
        if item in self.file_groups:
            return self.file_groups[item]
        
        # is it checking if there is a file of this type?
        if not item.startswith("has_"):
            raise AttributeError(f"{type(self)!r} object has no attribute {item!r}")  
        ext = item[4:]
        if ext not in self.supported_types:
            raise AttributeError(f"{type(self)!r} object does not support {ext!r} files")
        return ext in self.file_groups
        
    def has_filetype(self, ext: str) -> bool:
        """Check if the file type is available."""
        return getattr(self, f"has_{ext}", False)
    
    
    # def __dir__(self) -> list[str]:
    #     """Dynamically create a list of attributes for supported file types"""
    #     return (list(super().__dir__()) 
    #             + [f"has_{ext}" for ext in self.supported_types]
    #             + list(self.file_groups.keys())
    #     )
    
    @property
    def kwargs_for_read(self) -> dict:
        return {
        "diff": self.diff,  # difference timesteps
        "remove_coarse_band": self.remove_coarse_band,  # doesn't work with low freq res
        "correct_van_vleck": self.correct_van_vleck,  # slow
        "remove_flagged_ants": self.remove_flagged_ants,  # remove flagged ants
        "flag_init": self.flag_init,
        "flag_choice": self.flag_choice,
        "run_check": False,
    }

    @classmethod
    def group_files_by_extension(cls, file_list: list[str]) -> dict[str, list[Path]]:
        """Group files by their extensions."""
        file_groups = defaultdict(list)
        for file in file_list:
            ext = Path(file).suffix[1:]  # Get the file extension without the dot

            file_groups[ext].append(Path(file))
        return {ext: sorted(files) for ext, files in file_groups.items()}

    @classmethod
    def group_files_by_obsid_and_extension(
        cls, file_list: list[str]
    ) -> dict[str, dict[str, list[Path]]]:
        """Group files by their obsid and extensions."""
        file_groups = defaultdict(lambda: defaultdict(list))
        for file in file_list:
            ext = Path(file).suffix[1:]
            obsid = Path(file).stem.split('_')[0]
            file_groups[obsid][ext].append(Path(file))
        return {obsid: {ext: sorted(files) for ext, files in extensions.items()} for
                obsid, extensions in file_groups.items()}

    @staticmethod
    def _has_metafits_for_obs_id(obsid_group: dict[str, dict[str, list[Path]]]) -> bool:
        """Check if metafits files are present for the given obsid group."""
        for file_group in obsid_group.values():
            if  "metafits" not in file_group or len(file_group) == 0:
                return False
        return True

    @property
    def kwargs_for_select(self) -> dict[str, bool]:
        """Provide keyword arguments for the select operation."""
        return {"run_check": False}

    def get_size_mb(self) -> int:
        """Get the size of the input files in MB."""
        return sum(disk_usage_in_blocks(Path(file)) for file in self.files)

    # seems to be only relevant for fits


    def observations(self) -> Generator[tuple[str, str, list[Path]], None, None]:
        """Get the first metafits file name stem, if available."""
        assert self.obsid_groups is not None, "obsid_groups is not set"
        for obsid, file_group in self.obsid_groups.items():
                metafits = file_group["metafits"][0].stem
                raw_fits = file_group["fits"]
                yield obsid, metafits, raw_fits


# WARN: this is mostly a stub for now to check out this structure but as refactoring
# progresses, these classes will contain more appropriate functionality
# please do not treat the API as stable until this comment is removed.

@dataclass
class SSINSConfig(UVDataFileSet):
    """Configuration for SSINS (Sky-Subtracted Incoherent Noise Spectra) processing."""

    # SSINS.INS options
    spectrum_type: Literal["all", "auto", "cross"] = "cross"

    # SSINS.MF options
    threshold: float = 5.0
    narrow: float = 7.0
    streak: float = 8.0
    tb_aggro: float = 0.6

    # Plotting options
    plot_type: Literal["spectrum", "sigchain", "flags"] = "spectrum"


if __name__ == "__main__":
    print("Not meant to be executed as a script.")
    sys.exit(1)
