"""
用于分析视频内容的视频处理模块。
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple
from .text_processor import TextProcessor
from .audio_processor import AudioProcessor

class VideoProcessor:
    """处理视频和帧分析的类。"""

    def __init__(self, config: Dict[str, Any], model_loader):
        """
        初始化视频处理器。

        参数:
            config (Dict[str, Any]): 视频处理的配置字典。
            model_loader: ModelLoader实例，用于加载和管理模型。
        """
        self.config = config
        self.max_resolution = tuple(config['video']['max_resolution'])
        self.frame_interval = config['video']['frame_interval']
        self.scene_threshold = config['video']['scene_threshold']
        
        # 初始化文本和音频处理器
        self.text_processor = TextProcessor(model_loader)
        self.audio_processor = AudioProcessor(config, model_loader)

    def process(self, video_path: str) -> Dict[str, Any]:
        """
        处理视频文件并提取见解。

        参数:
            video_path (str): 视频文件路径。

        返回:
            Dict[str, Any]: 视频分析结果。
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"未找到视频文件：{video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")

        try:
            # 获取视频属性
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps

            # 提取关键帧
            frames, scene_changes = self._extract_key_frames(cap)
            
            # 处理音频
            audio_results = self.audio_processor.process(video_path)
            
            # 处理提取的帧中的文本
            text_results = self.text_processor.process(frames)

            # 生成最终结果
            return self._generate_results(
                video_path=video_path,
                frames=frames,
                scene_changes=scene_changes,
                text_results=text_results,
                audio_results=audio_results,
                metadata={
                    'total_frames': total_frames,
                    'fps': fps,
                    'resolution': (width, height),
                    'duration': duration
                }
            )

        finally:
            cap.release()

    def _extract_key_frames(self, cap: cv2.VideoCapture) -> Tuple[List[np.ndarray], List[int]]:
        """
        从视频中提取关键帧。

        参数:
            cap (cv2.VideoCapture): 视频捕获对象。

        返回:
            Tuple[List[np.ndarray], List[int]]: 关键帧列表和场景变化帧索引列表。
        """
        frames = []
        scene_changes = []
        prev_frame = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            
            # 每隔指定帧数提取一帧
            if frame_count % self.frame_interval != 0:
                continue
                
            # 调整帧大小
            if frame.shape[:2] != self.max_resolution:
                frame = cv2.resize(frame, self.max_resolution)
            
            # 检测场景变化
            if prev_frame is not None:
                diff = cv2.absdiff(prev_frame, frame)
                change_score = np.mean(diff)
                if change_score > self.scene_threshold:
                    scene_changes.append(frame_count)
                    frames.append(frame)  # 保存场景变化帧
            
            # 保存当前帧用于下一次比较
            prev_frame = frame.copy()
            
            # 保存采样帧
            frames.append(frame)
            
        return frames, scene_changes

    def _generate_results(self,
                         video_path: str,
                         frames: List[np.ndarray],
                         scene_changes: List[int],
                         text_results: Dict[str, Any],
                         audio_results: Dict[str, Any],
                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成最终的分析结果。

        参数:
            video_path (str): 视频文件路径。
            frames (List[np.ndarray]): 提取的帧列表。
            scene_changes (List[int]): 场景变化帧列表。
            text_results (Dict[str, Any]): 文本分析结果。
            audio_results (Dict[str, Any]): 音频分析结果。
            metadata (Dict[str, Any]): 视频元数据。

        返回:
            Dict[str, Any]: 完整的分析结果。
        """
        # 提取文本关键词
        text_keywords = []
        if 'analysis' in text_results and 'keywords' in text_results['analysis']:
            text_keywords = [kw['word'] for kw in text_results['analysis']['keywords']]

        # 生成Markdown格式的摘要
        summary_template = self.config['output']['summary_template']
        summary = summary_template.format(
            video_path=video_path,
            duration=f"{metadata['duration']:.2f}秒",
            resolution=f"{metadata['resolution'][0]}x{metadata['resolution'][1]}",
            text_content=self._format_text_content(text_results),
            speech_content=self._format_speech_content(audio_results),
            keywords=self._format_keywords(text_keywords)
        )

        return {
            'metadata': metadata,
            'analysis': {
                'frames_analyzed': len(frames),
                'scene_changes': len(scene_changes),
                'text_analysis': text_results.get('analysis', {}),
                'audio_analysis': audio_results.get('analysis', {}),
                'keywords': text_keywords
            },
            'summary': summary
        }

    def _format_text_content(self, text_results: Dict[str, Any]) -> str:
        """格式化文本内容为Markdown格式。"""
        if not text_results.get('text_segments'):
            return "未检测到文本内容"
            
        content = []
        for segment in text_results['text_segments']:
            for text_item in segment['texts']:
                content.append(f"- {text_item['text']} (置信度: {text_item['confidence']:.2f})")
        
        return "\n".join(content)

    def _format_speech_content(self, audio_results: Dict[str, Any]) -> str:
        """格式化语音内容为Markdown格式。"""
        if 'transcription' not in audio_results or not audio_results['transcription'].get('segments'):
            return "未检测到语音内容"
            
        content = []
        for segment in audio_results['transcription']['segments']:
            timestamp = f"[{segment['start']:.1f}s - {segment['end']:.1f}s]"
            content.append(f"- {timestamp} {segment['text']}")
        
        return "\n".join(content)

    def _format_keywords(self, keywords: List[str]) -> str:
        """格式化关键词为Markdown格式。"""
        if not keywords:
            return "未提取到关键词"
            
        return "\n".join([f"- {keyword}" for keyword in keywords])
