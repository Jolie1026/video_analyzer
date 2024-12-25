"""
用于从视频中提取和分析文本内容的文本处理模块。
"""

import cv2
import numpy as np
import jieba
import jieba.analyse
from pathlib import Path
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer

class TextProcessor:
    """处理视频内容中的文本提取和分析。"""

    def __init__(self, model_loader):
        """
        初始化文本处理器及其必要的模型和配置。
        
        参数:
            model_loader: ModelLoader实例，用于加载和管理模型。
        """
        self.ocr_model = model_loader.get_model('ocr')
        self.text_model = model_loader.get_model('text')
        self.tokenizer = model_loader.get_tokenizer('text')
        
        # 初始化jieba分词
        jieba.initialize()
        # 加载停用词
        self.stopwords = set(self._load_stopwords())
        
        # 设置关键词提取器
        self.tfidf = TfidfVectorizer(
            tokenizer=jieba.cut,
            stop_words=self.stopwords,
            max_features=100
        )

    def process(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        处理视频帧以提取和分析文本内容。

        参数:
            frames (List[np.ndarray]): 视频帧列表。

        返回:
            Dict[str, Any]: 文本分析结果。
        """
        try:
            # 从视频帧中提取文本
            extracted_text = self._extract_text_from_frames(frames)
            
            # 分析提取的文本
            if extracted_text['text_segments']:
                analysis_results = self._analyze_text(extracted_text['text_segments'])
            else:
                analysis_results = {
                    'error': '视频中未找到文本内容'
                }

            return {
                'metadata': {
                    'total_segments': len(extracted_text['text_segments']),
                    'frames_processed': len(frames)
                },
                'text_segments': extracted_text['text_segments'],
                'analysis': analysis_results
            }

        except Exception as e:
            raise RuntimeError(f"处理文本时出错：{str(e)}")

    def _extract_text_from_frames(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """使用PaddleOCR从视频帧中提取文本。"""
        text_segments = []

        for frame_idx, frame in enumerate(frames):
            # 使用PaddleOCR进行文本检测和识别
            result = self.ocr_model.ocr(frame, cls=True)
            
            if result:
                frame_texts = []
                for line in result:
                    try:
                        # PaddleOCR返回格式：[[[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, confidence)]
                        if not isinstance(line, (list, tuple)) or len(line) != 2:
                            continue
                            
                        box = line[0]
                        if not isinstance(box, (list, tuple)) or len(box) != 4:
                            continue
                            
                        text_info = line[1]
                        if not isinstance(text_info, (list, tuple)) or len(text_info) != 2:
                            continue
                            
                        text, confidence = text_info
                        
                        # 确保text是字符串类型并且非空
                        if isinstance(text, (list, tuple)):
                            text = ' '.join(str(t) for t in text if str(t).strip())
                        else:
                            text = str(text).strip()
                            
                        if text:  # 只保存非空文本
                            # 确保box坐标是数值类型
                            processed_box = []
                            for point in box:
                                if isinstance(point, (list, tuple)) and len(point) == 2:
                                    x, y = point
                                    # 确保坐标是数值
                                    try:
                                        processed_box.append([float(x), float(y)])
                                    except (ValueError, TypeError):
                                        continue
                            
                            if len(processed_box) == 4:  # 确保有4个有效的坐标点
                                frame_texts.append({
                                    'text': text,
                                    'confidence': float(confidence),
                                    'position': {
                                        'box': processed_box,
                                        'center': self._calculate_box_center(processed_box)
                                    },
                                    'frame_number': frame_idx
                                })
                    except Exception as e:
                        # 记录错误但继续处理其他文本
                        print(f"处理OCR结果时出错：{str(e)}")
                        continue

                if frame_texts:
                    text_segments.append({
                        'frame_number': frame_idx,
                        'texts': frame_texts
                    })

        return {
            'text_segments': text_segments,
            'frames_processed': len(frames)
        }

    def _calculate_box_center(self, box: List[List[float]]) -> Dict[str, float]:
        """
        计算边界框的中心点。
        
        参数:
            box: 包含4个[x,y]坐标点的列表
            
        返回:
            包含x,y中心坐标的字典
        """
        try:
            if not box or len(box) != 4:
                raise ValueError("Invalid box coordinates")
                
            x_coords = []
            y_coords = []
            
            for point in box:
                if not isinstance(point, (list, tuple)) or len(point) != 2:
                    raise ValueError("Invalid point coordinates")
                x_coords.append(float(point[0]))
                y_coords.append(float(point[1]))
                
            if not x_coords or not y_coords:
                raise ValueError("No valid coordinates found")
                
            center_x = sum(x_coords) / len(x_coords)
            center_y = sum(y_coords) / len(y_coords)
            
            return {'x': float(center_x), 'y': float(center_y)}
            
        except (TypeError, ValueError) as e:
            print(f"计算边界框中心点时出错：{str(e)}")
            return {'x': 0.0, 'y': 0.0}  # 返回默认值而不是抛出异常

    def _analyze_text(self, text_segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        分析提取的文本片段。

        参数:
            text_segments (List[Dict[str, Any]]): 提取的文本片段列表。

        返回:
            Dict[str, Any]: 包含关键词和文本分析的结果。
        """
        # 合并所有文本进行整体分析
        all_text = ' '.join([
            text['text'] for segment in text_segments
            for text_item in segment['texts']
        ])

        # 使用jieba提取关键词
        keywords = self._extract_keywords(all_text)
        
        # 文本聚类分析
        clusters = self._cluster_text_segments(text_segments)
        
        # 计算文本统计数据
        statistics = self._calculate_text_statistics(all_text)

        return {
            'keywords': keywords,
            'text_clusters': clusters,
            'statistics': statistics,
            'summary': self._generate_summary(keywords, clusters, statistics)
        }

    def _extract_keywords(self, text: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        从文本中提取关键词。

        参数:
            text (str): 要分析的文本。
            top_k (int): 返回的关键词数量。

        返回:
            List[Dict[str, Any]]: 关键词列表，包含词语和权重。
        """
        # 使用jieba的TF-IDF提取关键词
        keywords = jieba.analyse.extract_tags(
            text,
            topK=top_k,
            withWeight=True,
            allowPOS=('ns', 'n', 'vn', 'v', 'nr')
        )
        
        return [
            {'word': word, 'weight': float(weight)}
            for word, weight in keywords
        ]

    def _cluster_text_segments(self, text_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        对文本片段进行聚类分析。

        参数:
            text_segments (List[Dict[str, Any]]): 文本片段列表。

        返回:
            List[Dict[str, Any]]: 聚类结果。
        """
        # 将文本按位置和内容相似度进行聚类
        clusters = []
        processed_indices = set()

        for i, segment in enumerate(text_segments):
            if i in processed_indices:
                continue

            cluster_texts = []
            cluster_frames = set()

            # 查找相似的文本片段
            for j, other_segment in enumerate(text_segments):
                if j in processed_indices:
                    continue

                # 检查文本相似度和位置接近度
                if self._are_segments_similar(segment, other_segment):
                    cluster_texts.extend(other_segment['texts'])
                    cluster_frames.add(other_segment['frame_number'])
                    processed_indices.add(j)

            if cluster_texts:
                clusters.append({
                    'texts': cluster_texts,
                    'frames': sorted(list(cluster_frames)),
                    'main_text': self._get_representative_text(cluster_texts)
                })

        return clusters

    def _are_segments_similar(self, segment1: Dict[str, Any], segment2: Dict[str, Any]) -> bool:
        """判断两个文本片段是否相似。"""
        # 简单实现：检查文本内容是否有重叠
        texts1 = set(t['text'] for t in segment1['texts'])
        texts2 = set(t['text'] for t in segment2['texts'])
        
        # 计算文本重叠率
        overlap = len(texts1.intersection(texts2))
        total = len(texts1.union(texts2))
        
        return overlap / total > 0.5 if total > 0 else False

    def _get_representative_text(self, texts: List[Dict[str, Any]]) -> str:
        """从一组文本中选择最具代表性的文本。"""
        # 选择置信度最高的文本
        if not texts:
            return ""
            
        best_text = max(texts, key=lambda x: x['confidence'])
        return best_text['text']

    def _calculate_text_statistics(self, text: str) -> Dict[str, Any]:
        """计算文本的统计数据。"""
        words = list(jieba.cut(text))
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'unique_words': len(set(words)),
            'average_word_length': sum(len(word) for word in words) / len(words) if words else 0
        }

    def _load_stopwords(self) -> List[str]:
        """加载中文停用词表。"""
        # 这里可以加载自定义的停用词表
        # 返回基本的停用词列表
        return ['的', '了', '和', '是', '就', '都', '而', '及', '与', '这', '那', '有', '在', '中']

    def _generate_summary(self, keywords: List[Dict[str, Any]],
                         clusters: List[Dict[str, Any]],
                         statistics: Dict[str, Any]) -> Dict[str, Any]:
        """生成文本分析摘要。"""
        return {
            'top_keywords': [kw['word'] for kw in keywords[:5]],
            'text_clusters': len(clusters),
            'total_characters': statistics['total_characters'],
            'total_words': statistics['total_words'],
            'unique_words': statistics['unique_words']
        }
