"""
视频分析器的日志配置模块。
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def setup_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    """
    根据指定配置设置日志记录器。

    参数:
        name (str): 日志记录器名称。
        config (Dict[str, Any]): 日志配置字典。

    返回:
        logging.Logger: 配置好的日志记录器实例。
    """
    logger = logging.getLogger(name)
    
    # 清除任何现有的处理器
    logger.handlers.clear()
    
    # 设置日志级别
    level = getattr(logging, config.get('level', 'INFO').upper())
    logger.setLevel(level)
    
    # 创建格式化器
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter(log_format)
    
    # 如果指定了日志文件，添加文件处理器
    log_file = config.get('file')
    if log_file:
        file_handler = _setup_file_handler(log_file, formatter)
        logger.addHandler(file_handler)
    
    # 添加控制台处理器
    console_handler = _setup_console_handler(formatter)
    logger.addHandler(console_handler)
    
    return logger

def _setup_file_handler(log_file: str, formatter: logging.Formatter) -> logging.FileHandler:
    """
    设置日志文件处理器。

    参数:
        log_file (str): 日志文件路径。
        formatter (logging.Formatter): 日志消息格式化器。

    返回:
        logging.FileHandler: 配置好的文件处理器。
    """
    # 确保日志目录存在
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    return file_handler

def _setup_console_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    """
    设置控制台处理器。

    参数:
        formatter (logging.Formatter): 日志消息格式化器。

    返回:
        logging.StreamHandler: 配置好的控制台处理器。
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    return console_handler

class VideoAnalyzerLogger:
    """具有视频分析特定日志记录方法的自定义日志记录器类。"""

    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化视频分析日志记录器。

        参数:
            name (str): 日志记录器名称。
            config (Dict[str, Any]): 日志配置字典。
        """
        self.logger = setup_logger(name, config)

    def log_video_start(self, video_path: str):
        """记录视频处理开始。"""
        self.logger.info(f"开始视频分析：{video_path}")

    def log_video_complete(self, video_path: str, duration: float):
        """记录视频处理完成。"""
        self.logger.info(f"完成视频分析：{video_path}（用时：{duration:.2f}秒）")

    def log_frame_progress(self, frame_number: int, total_frames: int):
        """记录帧处理进度。"""
        if frame_number % 100 == 0:  # 每100帧记录一次
            progress = (frame_number / total_frames) * 100
            self.logger.debug(f"正在处理帧：已完成{progress:.1f}%")

    def log_error(self, error: Exception, context: Optional[str] = None):
        """记录错误及可选的上下文。"""
        if context:
            self.logger.error(f"在{context}中出错：{str(error)}")
        else:
            self.logger.error(str(error))

    def log_warning(self, message: str, context: Optional[str] = None):
        """记录警告及可选的上下文。"""
        if context:
            self.logger.warning(f"[{context}] {message}")
        else:
            self.logger.warning(message)

    def log_model_load(self, model_name: str, success: bool):
        """记录模型加载状态。"""
        if success:
            self.logger.info(f"成功加载模型：{model_name}")
        else:
            self.logger.error(f"加载模型失败：{model_name}")

    def log_processing_stats(self, stats: Dict[str, Any]):
        """记录处理统计信息。"""
        self.logger.info("处理统计：")
        for key, value in stats.items():
            self.logger.info(f"  {key}: {value}")

    def log_output_saved(self, output_path: str, file_type: str):
        """记录输出保存。"""
        self.logger.info(f"已保存{file_type}输出到：{output_path}")

def get_logger(name: str, config: Dict[str, Any]) -> VideoAnalyzerLogger:
    """
    获取VideoAnalyzerLogger实例。

    参数:
        name (str): 日志记录器名称。
        config (Dict[str, Any]): 日志配置字典。

    返回:
        VideoAnalyzerLogger: 配置好的日志记录器实例。
    """
    return VideoAnalyzerLogger(name, config)
