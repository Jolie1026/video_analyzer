# Video Content Analyzer
# 视频内容分析器
# 비디오 콘텐츠 분석기

[English](#english) | [简体中文](#简体中文) | [한국어](#한국어)

<a name="english"></a>
# English

A powerful video content analysis tool that supports video text recognition, speech recognition, and keyword extraction.

## Features

- Video Frame Extraction
  - Extract key frames at fixed intervals
  - Scene change detection
  - Automatic resolution adjustment
- Text Recognition
  - Chinese text detection and recognition using PaddleOCR
  - Text position and confidence analysis
  - Keyword extraction
- Speech Recognition
  - Chinese speech recognition using Whisper
  - Timestamp annotation
  - Voice segment analysis
- Output Results
  - Generate structured Markdown reports
  - Text records with timestamps
  - Keyword extraction and statistics

## System Requirements

- Python 3.8 or higher
- CUDA compatible GPU (22GB+ VRAM recommended)
- FFmpeg (for audio processing)
- At least 32GB system memory
- 512GB available storage space

## Installation

1. Clone repository:
```bash
git clone https://github.com/your-username/video_analyzer.git
cd video_analyzer
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Usage

1. Graphical User Interface (GUI):
```bash
video-analyzer --gui
```
The GUI provides an intuitive interface with:
- Language selection (English, Simplified Chinese, Korean)
- Video file selection with file browser
- Output file path configuration
- System initialization
- Enhanced Progress Display
  - Real-time progress bars for each processing stage
  - Detailed status messages
  - Time remaining estimates
  - Processing speed indicators
- Results display

2. Command Line Interface (CLI):
```bash
video-analyzer path/to/your/video.mp4
```

3. Specify configuration file:
```bash
video-analyzer path/to/your/video.mp4 --config path/to/config.yaml
```

4. Specify output file:
```bash
video-analyzer path/to/your/video.mp4 --output path/to/output.md
```

## Configuration

The configuration file (config.yaml) contains the following main sections:

```yaml
video:
  max_resolution: [1920, 1080]  # Maximum processing resolution
  frame_interval: 30           # Extract one frame every 30 frames
  scene_threshold: 30.0        # Scene change detection threshold

audio:
  sample_rate: 16000          # Audio sample rate
  channels: 1                 # Audio channels
  format: "wav"               # Audio extraction format

models:
  ocr:
    name: "paddleocr"         # OCR model
    language: "ch"            # Chinese model
    use_gpu: true            # Use GPU acceleration
  
  speech:
    name: "whisper-large-v3"  # Speech recognition model
    language: "zh"            # Chinese
    batch_size: 16            # GPU batch size
    
  text:
    name: "bert-chinese-base" # Text analysis model
    max_length: 512          # Maximum text length
```

## Output Example

The analysis will generate a Markdown file containing:

```markdown
# Video Analysis Report

## Basic Information
- File path: video.mp4
- Duration: 300.00 seconds
- Resolution: 1920x1080

## Text Content
- "Example text 1" (Confidence: 0.95)
- "Example text 2" (Confidence: 0.88)

## Voice Content
- [0.0s - 5.2s] This is the first voice segment
- [5.3s - 10.1s] This is the second voice segment

## Keywords
- Keyword 1
- Keyword 2
- Keyword 3
```

## Performance Optimization Tips

1. GPU Memory Usage
   - OCR model: ~8GB
   - Whisper model: ~10GB
   - BERT model: ~2GB
   - At least 22GB VRAM recommended

2. Processing Speed Optimization
   - Adjust frame_interval to reduce processed frames
   - Adjust batch_size based on GPU memory
   - Use compute_type: "float16" to reduce VRAM usage

3. Output File Size
   - Reserve 2x video file size for storage
   - Temporary files are automatically cleaned

## Common Issues

1. GPU Memory Insufficient
   ```
   Solutions:
   - Lower video processing resolution
   - Increase frame interval
   - Use float16 precision
   ```

2. Audio Extraction Failed
   ```
   Solutions:
   - Ensure FFmpeg is properly installed
   - Check if video file contains audio track
   ```

3. Poor Text Recognition Quality
   ```
   Solutions:
   - Adjust max_resolution to increase resolution
   - Lower scene_threshold to extract more frames
   ```

<a name="简体中文"></a>
# 简体中文

一个功能强大的视频内容分析工具，支持视频文本识别、语音识别和关键词提取。

## 功能特点

- 视频帧提取
  - 按固定间隔提取关键帧
  - 场景变化检测
  - 自动调整分辨率
- 文本识别
  - 使用PaddleOCR进行中文文本检测和识别
  - 文本位置和置信度分析
  - 关键词提取
- 语音识别
  - 使用Whisper进行中文语音识别
  - 时间戳标注
  - 语音片段分析
- 结果输出
  - 生成结构化的Markdown报告
  - 包含时间戳的文本记录
  - 关键词提取和统计

## 系统要求

- Python 3.8 或更高版本
- CUDA 兼容的GPU（推荐22GB以上显存）
- FFmpeg（用于音频处理）
- 至少32GB系统内存
- 512GB可用存储空间

## 安装说明

1. 克隆仓库：
```bash
git clone https://github.com/your-username/video_analyzer.git
cd video_analyzer
```

2. 创建并激活虚拟环境：
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖：
```bash
pip install -e .
```

## 使用方法

1. 图形用户界面（GUI）：
```bash
video-analyzer --gui
```
GUI提供直观的界面，包含：
- 语言选择（英语、简体中文、韩语）
- 视频文件选择器
- 输出文件路径配置
- 系统初始化
- 增强的进度显示
  - 每个处理阶段的实时进度条
  - 详细的状态消息
  - 剩余时间估计
  - 处理速度指示器
- 结果显示

2. 命令行界面（CLI）：
```bash
video-analyzer path/to/your/video.mp4
```

3. 指定配置文件：
```bash
video-analyzer path/to/your/video.mp4 --config path/to/config.yaml
```

4. 指定输出文件：
```bash
video-analyzer path/to/your/video.mp4 --output path/to/output.md
```

## 配置说明

配置文件（config.yaml）包含以下主要部分：

```yaml
video:
  max_resolution: [1920, 1080]  # 最大处理分辨率
  frame_interval: 30           # 每30帧提取一帧
  scene_threshold: 30.0        # 场景切换检测阈值

audio:
  sample_rate: 16000          # 音频采样率
  channels: 1                 # 音频通道数
  format: "wav"               # 音频提取格式

models:
  ocr:
    name: "paddleocr"         # OCR模型
    language: "ch"            # 中文模型
    use_gpu: true            # 使用GPU加速
  
  speech:
    name: "whisper-large-v3"  # 语音识别模型
    language: "zh"            # 中文
    batch_size: 16            # GPU批处理大小
    
  text:
    name: "bert-chinese-base" # 文本分析模型
    max_length: 512          # 最大文本长度
```

## 输出示例

分析结果将生成一个Markdown文件，包含以下内容：

```markdown
# 视频分析报告

## 基本信息
- 文件路径: video.mp4
- 时长: 300.00秒
- 分辨率: 1920x1080

## 文本内容
- "示例文本1" (置信度: 0.95)
- "示例文本2" (置信度: 0.88)

## 语音内容
- [0.0s - 5.2s] 这是第一段语音内容
- [5.3s - 10.1s] 这是第二段语音内容

## 关键词
- 关键词1
- 关键词2
- 关键词3
```

## 性能优化建议

1. GPU内存使用
   - OCR模型：约8GB
   - Whisper模型：约10GB
   - BERT模型：约2GB
   - 建议至少22GB显存

2. 处理速度优化
   - 适当调整frame_interval减少处理帧数
   - 根据GPU内存调整batch_size
   - 可以使用compute_type: "float16"降低显存占用

3. 输出文件大小
   - 建议预留视频文件大小2倍的存储空间
   - 中间文件会自动清理

## 常见问题

1. GPU内存不足
   ```
   解决方案：
   - 降低视频处理分辨率
   - 增加帧间隔
   - 使用float16精度
   ```

2. 音频提取失败
   ```
   解决方案：
   - 确保正确安装FFmpeg
   - 检查视频文件是否包含音轨
   ```

3. 文本识别质量不佳
   ```
   解决方案：
   - 调整max_resolution提高分辨率
   - 降低scene_threshold提取更多帧
   ```

<a name="한국어"></a>
# 한국어

비디오 텍스트 인식, 음성 인식 및 키워드 추출을 지원하는 강력한 비디오 콘텐츠 분석 도구입니다.

## 기능

- 비디오 프레임 추출
  - 고정 간격으로 키 프레임 추출
  - 장면 변화 감지
  - 자동 해상도 조정
- 텍스트 인식
  - PaddleOCR을 사용한 중국어 텍스트 감지 및 인식
  - 텍스트 위치 및 신뢰도 분석
  - 키워드 추출
- 음성 인식
  - Whisper를 사용한 중국어 음성 인식
  - 타임스탬프 주석
  - 음성 세그먼트 분석
- 결과 출력
  - 구조화된 Markdown 보고서 생성
  - 타임스탬프가 포함된 텍스트 기록
  - 키워드 추출 및 통계

## 시스템 요구사항

- Python 3.8 이상
- CUDA 호환 GPU (22GB+ VRAM 권장)
- FFmpeg (오디오 처리용)
- 최소 32GB 시스템 메모리
- 512GB 사용 가능한 저장 공간

## 설치

1. 저장소 복제:
```bash
git clone https://github.com/your-username/video_analyzer.git
cd video_analyzer
```

2. 가상 환경 생성 및 활성화:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 의존성 설치:
```bash
pip install -e .
```

## 사용법

1. 그래픽 사용자 인터페이스(GUI):
```bash
video-analyzer --gui
```
GUI는 다음과 같은 직관적인 인터페이스를 제공합니다:
- 언어 선택(영어, 중국어 간체, 한국어)
- 비디오 파일 선택기
- 출력 파일 경로 구성
- 시스템 초기화
- 향상된 진행 상황 표시
  - 각 처리 단계의 실시간 진행률 표시줄
  - 상세 상태 메시지
  - 남은 시간 추정
  - 처리 속도 표시기
- 결과 표시

2. 명령줄 인터페이스(CLI):
```bash
video-analyzer path/to/your/video.mp4
```

3. 구성 파일 지정:
```bash
video-analyzer path/to/your/video.mp4 --config path/to/config.yaml
```

4. 출력 파일 지정:
```bash
video-analyzer path/to/your/video.mp4 --output path/to/output.md
```

## 구성

구성 파일(config.yaml)에는 다음과 같은 주요 섹션이 포함됩니다:

```yaml
video:
  max_resolution: [1920, 1080]  # 최대 처리 해상도
  frame_interval: 30           # 30프레임마다 1프레임 추출
  scene_threshold: 30.0        # 장면 변화 감지 임계값

audio:
  sample_rate: 16000          # 오디오 샘플링 레이트
  channels: 1                 # 오디오 채널
  format: "wav"               # 오디오 추출 형식

models:
  ocr:
    name: "paddleocr"         # OCR 모델
    language: "ch"            # 중국어 모델
    use_gpu: true            # GPU 가속 사용
  
  speech:
    name: "whisper-large-v3"  # 음성 인식 모델
    language: "zh"            # 중국어
    batch_size: 16            # GPU 배치 크기
    
  text:
    name: "bert-chinese-base" # 텍스트 분석 모델
    max_length: 512          # 최대 텍스트 길이
```

## 출력 예시

분석은 다음 내용이 포함된 Markdown 파일을 생성합니다:

```markdown
# 비디오 분석 보고서

## 기본 정보
- 파일 경로: video.mp4
- 길이: 300.00초
- 해상도: 1920x1080

## 텍스트 내용
- "예시 텍스트 1" (신뢰도: 0.95)
- "예시 텍스트 2" (신뢰도: 0.88)

## 음성 내용
- [0.0s - 5.2s] 첫 번째 음성 세그먼트입니다
- [5.3s - 10.1s] 두 번째 음성 세그먼트입니다

## 키워드
- 키워드 1
- 키워드 2
- 키워드 3
```

## 성능 최적화 팁

1. GPU 메모리 사용량
   - OCR 모델: ~8GB
   - Whisper 모델: ~10GB
   - BERT 모델: ~2GB
   - 최소 22GB VRAM 권장

2. 처리 속도 최적화
   - frame_interval을 조정하여 처리 프레임 수 감소
   - GPU 메모리에 따라 batch_size 조정
   - compute_type: "float16"을 사용하여 VRAM 사용량 감소

3. 출력 파일 크기
   - 비디오 파일 크기의 2배 저장 공간 예약
   - 임시 파일은 자동으로 정리됨

## 일반적인 문제

1. GPU 메모리 부족
   ```
   해결 방법:
   - 비디오 처리 해상도 낮추기
   - 프레임 간격 증가
   - float16 정밀도 사용
   ```

2. 오디오 추출 실패
   ```
   해결 방법:
   - FFmpeg가 올바르게 설치되었는지 확인
   - 비디오 파일에 오디오 트랙이 있는지 확인
   ```

3. 텍스트 인식 품질 저하
   ```
   해결 방법:
   - max_resolution을 조정하여 해상도 증가
   - scene_threshold를 낮춰 더 많은 프레임 추출
