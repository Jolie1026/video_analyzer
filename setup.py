from setuptools import setup, find_packages

setup(
    name="video_analyzer",
    version="1.2.0",
    description="视频内容分析工具，支持文本识别、语音识别和关键词提取",
    author="Cline",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "moviepy>=1.0.3",
        "scipy>=1.7.0",
        "pyyaml>=5.4.0",
        "pytest>=6.2.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "transformers>=4.35.0",
        "paddlepaddle>=2.5.0",
        "paddleocr>=2.7.0",
        "openai-whisper>=20231117",
        "jieba>=0.42.1",
        "scikit-learn>=1.0.0",
        "ffmpeg-python>=0.2.0",
        "nltk>=3.8.1",
        "PyQt6>=6.4.0"
    ],
    entry_points={
        'console_scripts': [
            'video-analyzer=video_analyzer.main:main',
        ],
    },
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        'video_analyzer': ['config/*.yaml'],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Multimedia :: Video",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
