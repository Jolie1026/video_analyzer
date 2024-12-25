"""
视频分析操作的文件工具函数。
"""

import os
from pathlib import Path
from typing import List, Union, Optional

# 支持的视频文件扩展名
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

def get_video_files(directory: Union[str, Path], recursive: bool = False) -> List[Path]:
    """
    获取指定目录中的所有视频文件。

    参数:
        directory (Union[str, Path]): 要搜索视频文件的目录。
        recursive (bool, optional): 是否递归搜索。默认为False。

    返回:
        List[Path]: 视频文件路径列表。

    异常:
        FileNotFoundError: 如果目录不存在。
    """
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"未找到目录：{directory}")
    
    if not directory.is_dir():
        raise NotADirectoryError(f"不是目录：{directory}")

    video_files = []
    
    if recursive:
        # 遍历所有子目录
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in VIDEO_EXTENSIONS:
                    video_files.append(file_path)
    else:
        # 仅在指定目录中搜索
        video_files = [
            f for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
    
    return sorted(video_files)

def ensure_directory(directory: Union[str, Path]) -> Path:
    """
    确保目录存在，如果不存在则创建。

    参数:
        directory (Union[str, Path]): 要确保存在的目录路径。

    返回:
        Path: 确保存在的目录路径。

    异常:
        OSError: 如果目录创建失败。
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

def get_output_path(base_dir: Union[str, Path],
                    filename: str,
                    extension: str,
                    subfolder: Optional[str] = None) -> Path:
    """
    生成确保不会命名冲突的输出文件路径。

    参数:
        base_dir (Union[str, Path]): 输出的基础目录。
        filename (str): 不带扩展名的期望文件名。
        extension (str): 文件扩展名（带或不带点）。
        subfolder (Optional[str], optional): 可选的子文件夹。默认为None。

    返回:
        Path: 输出文件的唯一路径。
    """
    # 确保扩展名以点开头
    if not extension.startswith('.'):
        extension = f'.{extension}'
    
    # 创建完整的输出目录路径
    output_dir = Path(base_dir)
    if subfolder:
        output_dir = output_dir / subfolder
    
    # 确保目录存在
    ensure_directory(output_dir)
    
    # 创建初始输出路径
    output_path = output_dir / f"{filename}{extension}"
    
    # 如果文件存在，添加数字直到找到唯一的名称
    counter = 1
    while output_path.exists():
        output_path = output_dir / f"{filename}_{counter}{extension}"
        counter += 1
    
    return output_path

def get_file_info(file_path: Union[str, Path]) -> dict:
    """
    获取文件信息。

    参数:
        file_path (Union[str, Path]): 文件路径。

    返回:
        dict: 包含文件信息的字典。

    异常:
        FileNotFoundError: 如果文件不存在。
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"未找到文件：{file_path}")
    
    stat = file_path.stat()
    return {
        'name': file_path.name,
        'extension': file_path.suffix,
        'size': stat.st_size,
        'created': stat.st_ctime,
        'modified': stat.st_mtime,
        'accessed': stat.st_atime,
        'is_video': file_path.suffix.lower() in VIDEO_EXTENSIONS
    }

def clean_filename(filename: str) -> str:
    """
    通过移除无效字符来清理文件名。

    参数:
        filename (str): 原始文件名。

    返回:
        str: 清理后的文件名。
    """
    # 用下划线替换无效字符
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # 移除开头和结尾的空格和点
    filename = filename.strip('. ')
    
    # 确保文件名不为空
    if not filename:
        filename = 'unnamed'
    
    return filename

def get_file_size_str(size_in_bytes: int) -> str:
    """
    将文件大小（字节）转换为人类可读的字符串。

    参数:
        size_in_bytes (int): 文件大小（字节）。

    返回:
        str: 人类可读的文件大小字符串。
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_in_bytes < 1024:
            return f"{size_in_bytes:.2f} {unit}"
        size_in_bytes /= 1024
    return f"{size_in_bytes:.2f} PB"
