"""
机器学习模型管理加载模块。
"""

import torch
import whisper
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoModel, AutoTokenizer, pipeline
from paddleocr import PaddleOCR

class ModelLoader:
    """处理机器学习模型的加载和管理。"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化模型加载器。

        参数:
            config (Dict[str, Any]): 包含模型路径和设置的配置字典。
        """
        self.config = config
        self.models = {}
        self.tokenizers = {}
        self._load_models()

    def _load_models(self):
        """根据配置加载所有需要的模型。"""
        model_configs = self.config.get('models', {})
        for model_type, model_config in model_configs.items():
            try:
                self._load_model(model_type, model_config)
            except Exception as e:
                print(f"加载{model_type}模型时出错：{str(e)}")

    def _load_model(self, model_type: str, model_config: Dict[str, Any]):
        """
        加载特定模型。

        参数:
            model_type (str): 要加载的模型类型（ocr/speech/text）。
            model_config (Dict[str, Any]): 特定模型的配置。
        """
        try:
            if model_type == 'ocr':
                self.models[model_type] = self._load_ocr_model(model_config)
            elif model_type == 'speech':
                self.models[model_type] = self._load_speech_model(model_config)
            elif model_type == 'text':
                model, tokenizer = self._load_text_model(model_config)
                self.models[model_type] = model
                self.tokenizers[model_type] = tokenizer
        except Exception as e:
            raise RuntimeError(f"加载{model_type}失败：{str(e)}")

    def _load_ocr_model(self, model_config: Dict[str, Any]) -> PaddleOCR:
        """
        加载OCR模型。

        参数:
            model_config (Dict[str, Any]): OCR模型配置。

        返回:
            PaddleOCR: 加载的OCR模型。
        """
        ocr_kwargs = {
            'use_angle_cls': True,
            'lang': model_config['language'],
            'use_gpu': model_config['use_gpu'],
            'det_model_dir': model_config['det_model_dir'],
            'rec_model_dir': model_config['rec_model_dir'],
            'show_log': False
        }
        
        # 添加TensorRT加速支持
        if model_config.get('use_tensorrt'):
            ocr_kwargs.update({
                'use_tensorrt': True,
                'precision': model_config.get('precision', 'fp32')
            })
            
        return PaddleOCR(**ocr_kwargs)

    def _load_speech_model(self, model_config: Dict[str, Any]) -> whisper.Whisper:
        """
        加载语音识别模型。

        参数:
            model_config (Dict[str, Any]): 语音模型配置。

        返回:
            whisper.Whisper: 加载的Whisper模型。
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = whisper.load_model(model_config['name'], device=device)
        
        # 设置批处理大小和计算精度
        model.batch_size = model_config.get('batch_size', 16)
        model.compute_type = model_config.get('compute_type', 'float32')
        
        return model

    def _load_text_model(self, model_config: Dict[str, Any]) -> tuple[Any, Any]:
        """
        加载文本处理模型。

        参数:
            model_config (Dict[str, Any]): 文本模型配置。

        返回:
            tuple[Any, Any]: 加载的文本模型及其分词器。
        """
        device = model_config['device'] if torch.cuda.is_available() else "cpu"
        
        # 加载BERT模型和分词器
        model = AutoModel.from_pretrained(model_config['name']).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_config['name'])
        
        # 启用自动混合精度训练
        if model_config.get('use_amp', False) and device == "cuda":
            model = torch.cuda.amp.autocast(enabled=True)(model.forward)
        
        # 创建带有批处理的pipeline
        batch_size = model_config.get('batch_size', 8)
        pipeline_model = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=batch_size
        )
        return model, tokenizer

    def get_model(self, model_type: str) -> Optional[Any]:
        """
        根据类型获取已加载的模型。

        参数:
            model_type (str): 要获取的模型类型。

        返回:
            Optional[Any]: 如果已加载则返回请求的模型，否则返回None。
        """
        return self.models.get(model_type)

    def get_tokenizer(self, model_type: str) -> Optional[Any]:
        """
        获取特定模型类型的分词器。

        参数:
            model_type (str): 要获取分词器的模型类型。

        返回:
            Optional[Any]: 如果可用则返回请求的分词器，否则返回None。
        """
        return self.tokenizers.get(model_type)

    def unload_model(self, model_type: str):
        """
        从内存中卸载模型。

        参数:
            model_type (str): 要卸载的模型类型。
        """
        if model_type in self.models:
            del self.models[model_type]
            if model_type in self.tokenizers:
                del self.tokenizers[model_type]
            torch.cuda.empty_cache()  # 清除CUDA缓存

    def reload_model(self, model_type: str):
        """
        重新加载特定模型。

        参数:
            model_type (str): 要重新加载的模型类型。
        """
        model_configs = self.config.get('models', {})
        if model_type in model_configs:
            self.unload_model(model_type)
            self._load_model(model_type, model_configs[model_type])
        else:
            raise ValueError(f"未找到模型类型的配置：{model_type}")

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        获取已加载模型的信息。

        参数:
            model_type (str): 要获取信息的模型类型。

        返回:
            Dict[str, Any]: 包含模型信息的字典。
        """
        if model_type not in self.models:
            return {"error": f"模型{model_type}未加载"}

        model = self.models[model_type]
        model_configs = self.config.get('models', {})
        
        info = {
            "name": model_type,
            "config": model_configs.get(model_type, {}),
            "loaded": True,
        }

        # 添加特定模型的额外信息
        if model_type == 'ocr':
            info.update({
                "language": model_configs[model_type]['language'],
                "gpu_enabled": model_configs[model_type]['use_gpu']
            })
        elif model_type == 'speech':
            info.update({
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "compute_type": model_configs[model_type]['compute_type']
            })
        elif model_type == 'text':
            info.update({
                "device": model_configs[model_type]['device'],
                "max_length": model_configs[model_type]['max_length']
            })

        return info
