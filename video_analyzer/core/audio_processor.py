"""
用于分析视频中音频内容的音频处理模块。
"""

import os
import numpy as np
import subprocess
import torch
import whisper
from pathlib import Path
from typing import Dict, Any, List, Optional
from scipy import signal
from scipy.io import wavfile

class AudioProcessor:
    """处理视频文件中的音频提取和分析。"""

    def __init__(self, config: Dict[str, Any], model_loader):
        """
        初始化音频处理器。

        参数:
            config (Dict[str, Any]): 音频处理的配置字典。
            model_loader: ModelLoader实例，用于加载和管理模型。
        """
        self.config = config['audio']
        self.speech_model = model_loader.get_model('speech')
        self.temp_dir = Path('temp')
        if not self.temp_dir.is_absolute():
            self.temp_dir = Path.cwd() / 'temp'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process(self, video_path: str) -> Dict[str, Any]:
        """
        处理视频文件中的音频并提取见解。

        参数:
            video_path (str): 视频文件路径。

        返回:
            Dict[str, Any]: 音频分析结果。
        """
        if not Path(video_path).exists():
            raise FileNotFoundError(f"未找到视频文件：{video_path}")

        audio_path = None
        try:
            # 从视频中提取音频
            audio_path = self._extract_audio(video_path)
            
            if not audio_path:
                return {
                    'error': '视频中未找到音轨',
                    'metadata': {
                        'has_audio': False
                    }
                }

            if not os.path.exists(audio_path):
                return {
                    'error': '音频提取失败',
                    'metadata': {
                        'has_audio': False
                    }
                }

            # 加载音频数据
            audio_array = self._load_audio(audio_path)
            
            # 执行语音识别
            transcription = self._transcribe_audio(audio_path)
            
            # 执行音频分析
            analysis_results = self._analyze_audio(audio_array)

            return {
                'metadata': {
                    'duration': len(audio_array) / self.config['sample_rate'],
                    'sample_rate': self.config['sample_rate'],
                    'channels': self.config['channels'],
                    'has_audio': True
                },
                'transcription': transcription,
                'analysis': analysis_results
            }

        except Exception as e:
            raise RuntimeError(f"处理音频时出错：{str(e)}")
        finally:
            # 确保清理临时文件
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                print(f"清理临时文件失败：{str(e)}")

    def _extract_audio(self, video_path: str) -> Optional[str]:
        """
        从视频文件中提取音频。

        参数:
            video_path (str): 视频文件路径。

        返回:
            Optional[str]: 提取的音频文件路径，如果没有音轨则返回None。
        """
        try:
            # 确保视频路径是绝对路径
            video_path = str(Path(video_path).resolve())
            print(f"处理视频文件：{video_path}")
            
            # 生成临时音频文件路径
            temp_audio_path = str(self.temp_dir / f"temp_audio_{os.getpid()}.{self.config['format']}")
            print(f"临时音频文件路径：{temp_audio_path}")
            
            # 使用ffmpeg提取音频
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # 不处理视频
                '-acodec', 'pcm_s16le',
                '-ac', str(self.config['channels']),
                '-ar', str(self.config['sample_rate']),
                '-y',  # 覆盖输出文件
                temp_audio_path
            ]
            process = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )
            if process.returncode != 0:
                raise RuntimeError(f"FFmpeg错误：{process.stderr}")
            print("音频提取成功")
            
            return temp_audio_path
        except Exception as e:
            if hasattr(e, 'stderr'):
                error_message = e.stderr.decode() if hasattr(e.stderr, 'decode') else str(e.stderr)
            else:
                error_message = str(e)
            print(f"提取音频失败：{error_message}")
            return None

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件。

        参数:
            audio_path (str): 音频文件路径。

        返回:
            np.ndarray: 音频数据的numpy数组。
        """
        sample_rate, audio_array = wavfile.read(audio_path)
        
        # 确保采样率匹配
        if sample_rate != self.config['sample_rate']:
            # 重采样
            duration = len(audio_array) / sample_rate
            new_length = int(duration * self.config['sample_rate'])
            audio_array = signal.resample(audio_array, new_length)
        
        # 如果是立体声，转换为单声道
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=1)
        
        return audio_array

    def _transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        使用Whisper模型进行语音识别。

        参数:
            audio_path (str): 音频文件路径。

        返回:
            Dict[str, Any]: 包含转录文本和时间戳的字典。
        """
        try:
            # 使用Whisper进行语音识别
            result = self.speech_model.transcribe(
                audio_path,
                language="zh",
                task="transcribe",
                verbose=False
            )
            
            # 处理结果
            segments = []
            for segment in result['segments']:
                segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'],
                    'confidence': float(segment.get('confidence', 0.0))
                })
            
            return {
                'text': result['text'],
                'segments': segments,
                'language': result['language']
            }
            
        except Exception as e:
            print(f"语音识别失败：{str(e)}")
            return {
                'error': f"语音识别失败：{str(e)}",
                'text': '',
                'segments': []
            }

    def _analyze_audio(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """
        分析音频数据并提取特征。

        参数:
            audio_array (np.ndarray): 音频数据的numpy数组。

        返回:
            Dict[str, Any]: 包含各种音频特征的分析结果。
        """
        # 计算基本统计数据
        amplitude_stats = {
            'max_amplitude': float(np.max(np.abs(audio_array))),
            'mean_amplitude': float(np.mean(np.abs(audio_array))),
            'rms': float(np.sqrt(np.mean(audio_array**2)))
        }

        # 执行频率分析
        frequencies = self._analyze_frequencies(audio_array)
        
        # 检测静音/语音片段
        segments = self._detect_segments(audio_array)
        
        # 计算能量分布
        energy = self._calculate_energy(audio_array)

        return {
            'amplitude_stats': amplitude_stats,
            'frequency_analysis': frequencies,
            'segments': segments,
            'energy_distribution': energy,
            'summary': self._generate_summary(amplitude_stats, frequencies, segments)
        }

    def _analyze_frequencies(self, audio_array: np.ndarray) -> Dict[str, Any]:
        """分析音频的频率成分。"""
        # 计算频谱图
        frequencies, times, spectrogram = signal.spectrogram(
            audio_array,
            fs=self.config['sample_rate'],
            nperseg=1024,
            noverlap=512
        )
        
        # 计算平均频谱
        avg_spectrum = np.mean(spectrogram, axis=1)
        
        # 查找主要频率
        dominant_freq_idx = np.argsort(avg_spectrum)[-5:]  # 前5个主要频率
        dominant_frequencies = frequencies[dominant_freq_idx]

        return {
            'dominant_frequencies': dominant_frequencies.tolist(),
            'frequency_distribution': {
                'low': float(np.mean(spectrogram[frequencies < 250])),
                'mid': float(np.mean(spectrogram[(frequencies >= 250) & (frequencies < 2000)])),
                'high': float(np.mean(spectrogram[frequencies >= 2000]))
            }
        }

    def _detect_segments(self, audio_array: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """检测和分类音频中的不同片段。"""
        # 计算短时能量
        frame_length = int(0.025 * self.config['sample_rate'])  # 25毫秒帧
        energy = self._calculate_frame_energy(audio_array, frame_length)
        
        # 静音检测阈值
        threshold = 0.1 * np.mean(energy)
        
        # 查找片段
        segments = []
        is_silence = energy[0] < threshold
        start = 0
        
        for i in range(1, len(energy)):
            if (energy[i] < threshold) != is_silence:
                segments.append({
                    'start': float(start / self.config['sample_rate']),
                    'end': float(i / self.config['sample_rate']),
                    'type': 'silence' if is_silence else 'speech',
                    'duration': float((i - start) / self.config['sample_rate'])
                })
                is_silence = not is_silence
                start = i

        # 添加最后一个片段
        segments.append({
            'start': float(start / self.config['sample_rate']),
            'end': float(len(energy) / self.config['sample_rate']),
            'type': 'silence' if is_silence else 'speech',
            'duration': float((len(energy) - start) / self.config['sample_rate'])
        })

        return {
            'total_segments': len(segments),
            'segments': segments
        }

    def _calculate_frame_energy(self, audio_array: np.ndarray, frame_length: int) -> np.ndarray:
        """计算音频每一帧的能量。"""
        pad_length = frame_length - (len(audio_array) % frame_length)
        padded_audio = np.pad(audio_array, (0, pad_length))
        frames = padded_audio.reshape(-1, frame_length)
        return np.sum(frames**2, axis=1)

    def _calculate_energy(self, audio_array: np.ndarray) -> Dict[str, float]:
        """计算音频信号的能量分布。"""
        # 计算总能量
        total_energy = np.sum(audio_array**2)
        
        # 计算不同时间段的能量
        n_segments = 10
        segment_length = len(audio_array) // n_segments
        segment_energies = [
            float(np.sum(audio_array[i:i+segment_length]**2))
            for i in range(0, len(audio_array), segment_length)
        ]

        return {
            'total_energy': float(total_energy),
            'energy_distribution': segment_energies,
            'energy_variance': float(np.var(segment_energies))
        }

    def _generate_summary(self, amplitude_stats: Dict[str, float],
                         frequencies: Dict[str, Any],
                         segments: Dict[str, Any]) -> Dict[str, Any]:
        """生成音频分析摘要。"""
        return {
            'average_amplitude': amplitude_stats['mean_amplitude'],
            'peak_amplitude': amplitude_stats['max_amplitude'],
            'frequency_balance': frequencies['frequency_distribution'],
            'speech_segments': len([s for s in segments['segments'] if s['type'] == 'speech']),
            'silence_ratio': sum(s['duration'] for s in segments['segments'] if s['type'] == 'silence') /
                           sum(s['duration'] for s in segments['segments'])
        }
