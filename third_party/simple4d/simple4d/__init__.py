"""SIMPLE-4D: surrogate-free 4D MRI from interleaved 2D cine slices."""
from .config import build_parser, load_config, parse_args
from .data import SliceDataset, SliceSeries
from .model import SIMPLE4D, load_run

__version__ = "1.0.0"
__all__ = ["SIMPLE4D", "SliceSeries", "SliceDataset", "build_parser", "parse_args", "load_config", "load_run"]
