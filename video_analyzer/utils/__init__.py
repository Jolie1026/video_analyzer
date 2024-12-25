"""
Utility modules for video analysis functionality.

This package provides utility functions and classes for file handling,
logging, and other common operations used throughout the video analyzer.
"""

from .file_utils import get_video_files, ensure_directory
from .logger import setup_logger

__all__ = ['get_video_files', 'ensure_directory', 'setup_logger']
