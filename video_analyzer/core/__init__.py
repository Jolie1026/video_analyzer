"""
Core functionality module for video analysis.

This module contains the core processors for video, audio, and text analysis.
"""

from .video_processor import VideoProcessor
from .audio_processor import AudioProcessor
from .text_processor import TextProcessor

__all__ = ['VideoProcessor', 'AudioProcessor', 'TextProcessor']
