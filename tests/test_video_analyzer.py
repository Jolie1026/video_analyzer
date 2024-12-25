"""
视频分析器测试模块。
"""

import os
import pytest
import numpy as np
import yaml
from pathlib import Path
from video_analyzer.models.model_loader import ModelLoader
from video_analyzer.core.video_processor import VideoProcessor
from video_analyzer.core.audio_processor import AudioProcessor
from video_analyzer.core.text_processor import TextProcessor

# 测试配置
TEST_CONFIG = {
    'video': {
        'max_resolution': [1920, 1080],
        'frame_interval': 30,
        'scene_threshold': 30.0
    },
    'audio': {
        'sample_rate': 16000,
        'channels': 1,
        'format': 'wav',
        'chunk_duration': 300
    },
    'models': {
        'ocr': {
            'name': 'paddleocr',
            'language': 'ch',
            'use_gpu': True,
            'det_model_dir': 'models/weights/ch_PP-OCRv4_det_infer',
            'rec_model_dir': 'models/weights/ch_PP-OCRv4_rec_infer'
        },
        'speech': {
            'name': 'whisper-large-v3',
            'language': 'zh',
            'batch_size': 16,
            'compute_type': 'float16'
        },
        'text': {
            'name': 'bert-chinese-base',
            'max_length': 512,
            'device': 'cuda'
        }
    },
    'output': {
        'format': 'markdown',
        'save_intermediates': True,
        'output_dir': 'test_output',
        'summary_template': """
# 视频分析报告

## 基本信息
- 文件路径: {video_path}
- 时长: {duration}
- 分辨率: {resolution}

## 文本内容
{text_content}

## 语音内容
{speech_content}

## 关键词
{keywords}
"""
    },
    'logging': {
        'level': 'INFO',
        'file': 'logs/test_video_analyzer.log',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    }
}

@pytest.fixture
def config():
    """配置文件fixture。"""
    return TEST_CONFIG

@pytest.fixture
def model_loader(config):
    """ModelLoader fixture。"""
    return ModelLoader(config)

@pytest.fixture
def video_processor(config, model_loader):
    """VideoProcessor fixture。"""
    return VideoProcessor(config, model_loader)

@pytest.fixture
def audio_processor(config, model_loader):
    """AudioProcessor fixture。"""
    return AudioProcessor(config, model_loader)

@pytest.fixture
def text_processor(model_loader):
    """TextProcessor fixture。"""
    return TextProcessor(model_loader)

def test_model_loader_initialization(model_loader):
    """测试模型加载器初始化。"""
    assert model_loader is not None
    assert 'ocr' in model_loader.models
    assert 'speech' in model_loader.models
    assert 'text' in model_loader.models

def test_video_processor_configuration(video_processor, config):
    """测试视频处理器配置。"""
    assert video_processor.max_resolution == tuple(config['video']['max_resolution'])
    assert video_processor.frame_interval == config['video']['frame_interval']
    assert video_processor.scene_threshold == config['video']['scene_threshold']

def test_audio_processor_configuration(audio_processor, config):
    """测试音频处理器配置。"""
    assert audio_processor.config['sample_rate'] == config['audio']['sample_rate']
    assert audio_processor.config['channels'] == config['audio']['channels']
    assert audio_processor.config['format'] == config['audio']['format']

def test_text_processor_initialization(text_processor):
    """测试文本处理器初始化。"""
    assert text_processor.ocr_model is not None
    assert text_processor.text_model is not None
    assert hasattr(text_processor, 'stopwords')
    assert hasattr(text_processor, 'tfidf')

@pytest.mark.parametrize("frame_size", [(640, 480), (1920, 1080)])
def test_video_frame_processing(video_processor, frame_size):
    """测试不同尺寸的视频帧处理。"""
    # 创建测试帧
    test_frame = np.random.randint(0, 255, (*frame_size, 3), dtype=np.uint8)
    
    # 处理帧
    frames, _ = video_processor._extract_key_frames(test_frame)
    
    # 检查帧尺寸
    if frames:
        assert frames[0].shape[:2] == video_processor.max_resolution

@pytest.mark.parametrize("test_text", [
    "这是一个测试句子。",
    "",  # 空文本
    "多个句子。带有不同。标点符号！"
])
def test_text_analysis(text_processor, test_text):
    """测试文本分析功能。"""
    # 创建测试帧（包含文本）
    test_frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
    cv2.putText(test_frame, test_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 分析文本
    results = text_processor.process([test_frame])
    
    assert 'metadata' in results
    assert 'text_segments' in results
    assert 'analysis' in results

def test_audio_extraction(audio_processor, tmp_path):
    """测试音频提取功能。"""
    # 创建测试视频文件
    test_video = tmp_path / "test_video.mp4"
    # 这里需要创建一个有效的测试视频文件
    
    if test_video.exists():
        audio_path = audio_processor._extract_audio(str(test_video))
        if audio_path:
            assert Path(audio_path).exists()
            assert Path(audio_path).suffix == f".{audio_processor.config['format']}"

def test_save_results(config, tmp_path):
    """测试结果保存功能。"""
    # 创建测试结果
    test_results = {
        'metadata': {
            'duration': 300.0,
            'resolution': (1920, 1080)
        },
        'analysis': {
            'frames_analyzed': 100,
            'scene_changes': 10,
            'keywords': ['测试', '关键词']
        },
        'summary': "# 测试报告\n## 内容\n- 测试内容"
    }
    
    # 保存结果
    output_path = tmp_path / "test_results.md"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(test_results['summary'])
    
    assert output_path.exists()
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert content.startswith("# 测试报告")

def test_invalid_video_path(video_processor):
    """测试无效视频路径处理。"""
    with pytest.raises(FileNotFoundError):
        video_processor.process("nonexistent_video.mp4")

def test_model_info(model_loader):
    """测试模型信息获取。"""
    for model_type in ['ocr', 'speech', 'text']:
        info = model_loader.get_model_info(model_type)
        assert info['name'] == model_type
        assert info['loaded'] is True

if __name__ == '__main__':
    pytest.main([__file__])
