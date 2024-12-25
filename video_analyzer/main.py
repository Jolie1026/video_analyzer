"""
Video analyzer main program entry.
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
from .models.model_loader import ModelLoader
from .core.video_processor import VideoProcessor
from .utils.i18n import get_text, set_language

def setup_logging(config: Dict[str, Any]):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config['logging']['level']),
        format=config['logging']['format'],
        handlers=[
            logging.FileHandler(config['logging']['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration file.

    Args:
        config_path (str): Configuration file path.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RuntimeError(get_text("error_loading_config", str(e)))

def create_output_dir(output_dir: str):
    """
    Create output directory.

    Args:
        output_dir (str): Output directory path.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def save_results(results: Dict[str, Any], output_path: str):
    """
    Save analysis results.

    Args:
        results (Dict[str, Any]): Analysis results.
        output_path (str): Output file path.
    """
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Write results
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(results['summary'])
            
        logging.info(get_text("result_saved", output_path))
    except Exception as e:
        logging.error(get_text("error_saving_results", str(e)))
        raise

def process_video(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single video file.

    Args:
        video_path (str): Video file path.
        config (Dict[str, Any]): Configuration dictionary.

    Returns:
        Dict[str, Any]: Analysis results.
    """
    try:
        # Initialize model loader
        model_loader = ModelLoader(config)
        
        # Initialize video processor
        processor = VideoProcessor(config, model_loader)
        
        # Process video
        results = processor.process(video_path)
        
        return results
    except Exception as e:
        logging.error(get_text("error_processing_failed", str(e)))
        raise

def initialize_system():
    """Initialize system, clear all caches, temporary files and logs."""
    try:
        import shutil
        import tempfile
        from concurrent.futures import ThreadPoolExecutor
        from itertools import chain
        
        # Define protected directories
        PROTECTED_DIRS = {'.git', 'models/weights'}
        
        # Define file patterns to clean
        FILE_PATTERNS = {
            '*.tmp', '*.temp', '*.cache', '*.pyc', '*.pyo',
            '*.log', '*.bak', '*.swp', '*.swo'
        }
        
        # Define directory names to clean
        DIR_PATTERNS = {
            'temp', 'tmp', '__pycache__', '.pytest_cache',
            '.mypy_cache', '.coverage_cache'
        }
        
        # Define project-related file patterns in system temp
        SYSTEM_TEMP_PATTERNS = {
            'video_analyzer_*', 'va_*'
        }
        
        def is_protected_path(path: Path) -> bool:
            """Check if path is protected."""
            return any(protected in path.parts for protected in PROTECTED_DIRS)
        
        def remove_path(path: Path):
            """Safely remove file or directory."""
            try:
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
            except Exception as e:
                print(get_text("warning_cannot_delete", str(path), str(e)))
        
        def clean_directory(directory: Path, patterns: set):
            """Clean files matching patterns in directory."""
            if not directory.exists() or is_protected_path(directory):
                return set()
            
            paths_to_remove = set()
            
            # Collect matching files
            for pattern in patterns:
                paths_to_remove.update(directory.glob(pattern))
            
            return paths_to_remove
        
        # Collect all paths to clean
        paths_to_clean = set()
        
        # 1. Clean output directory (keep .gitkeep)
        output_dir = Path('output')
        if output_dir.exists():
            paths_to_clean.update(
                path for path in output_dir.iterdir()
                if path.name != '.gitkeep'
            )
        
        # 2. Clean logs directory
        logs_dir = Path('logs')
        if logs_dir.exists():
            paths_to_clean.add(logs_dir)
        
        # 3. Clean project-related files in system temp
        system_temp = Path(tempfile.gettempdir())
        paths_to_clean.update(
            chain.from_iterable(
                system_temp.glob(pattern)
                for pattern in SYSTEM_TEMP_PATTERNS
            )
        )
        
        # 4. Recursively find temporary files in project
        for root, dirs, _ in os.walk('.', topdown=True):
            # Skip protected directories
            dirs[:] = [d for d in dirs if not is_protected_path(Path(root) / d)]
            
            root_path = Path(root)
            # Collect matching files
            paths_to_clean.update(clean_directory(root_path, FILE_PATTERNS))
            # Collect matching directories
            paths_to_clean.update(
                root_path / dir_name
                for dir_name in dirs
                if dir_name in DIR_PATTERNS
            )
        
        # Use thread pool to remove files in parallel
        with ThreadPoolExecutor() as executor:
            list(executor.map(remove_path, paths_to_clean))
        
        # Rebuild logs directory
        logs_dir.mkdir(exist_ok=True)
        
        print(get_text("init_success"))
        return True
    except Exception as e:
        print(get_text("init_failed", str(e)))
        return False

def main():
    """Main program entry."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Video Content Analysis Tool')
    parser.add_argument('--init', action='store_true', help='Initialize system, clear all caches and logs')
    parser.add_argument('video_path', nargs='?', help='Video file path to analyze')
    parser.add_argument('--config', default='config/config.yaml', help='Configuration file path')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--lang', default='en', choices=['en', 'zh_CN', 'ko'], help='Interface language (en/zh_CN/ko)')
    args = parser.parse_args()

    try:
        # Set language
        set_language(args.lang)

        # If initialization command
        if args.init:
            initialize_system()
            return

        # Ensure video path is provided
        if not args.video_path:
            parser.error(get_text("provide_video_path"))

        # Load configuration
        config = load_config(args.config)
        
        # Setup logging
        setup_logging(config)
        
        # Create output directory
        create_output_dir(config['output']['output_dir'])
        
        # Process video
        logging.info(get_text("start_processing", args.video_path))
        results = process_video(args.video_path, config)
        
        # Determine output path
        if args.output:
            output_path = args.output
        else:
            video_name = Path(args.video_path).stem
            output_path = os.path.join(
                config['output']['output_dir'],
                f"{video_name}_analysis.md"
            )
        
        # Save results
        save_results(results, output_path)
        
        logging.info(get_text("processing_complete"))
        
    except Exception as e:
        logging.error(get_text("execution_failed", str(e)))
        sys.exit(1)

if __name__ == '__main__':
    main()
