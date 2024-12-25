import os
import json
from typing import Dict, Optional

class I18n:
    """Internationalization helper class"""
    
    def __init__(self):
        self.translations: Dict[str, Dict[str, str]] = {}
        self.current_lang = "en"
        self._load_translations()

    def _load_translations(self):
        """Load all translation files"""
        translations_dir = os.path.join(os.path.dirname(__file__), "..", "translations")
        os.makedirs(translations_dir, exist_ok=True)
        
        # Default English translations
        self.translations["en"] = {
            "video_processing": "Processing video...",
            "audio_processing": "Processing audio...",
            "text_processing": "Processing text...",
            "processing_complete": "Processing complete",
            "error_file_not_found": "Error: File not found",
            "error_processing_failed": "Error: Processing failed",
            "error_loading_config": "Failed to load configuration: {}",
            "error_saving_results": "Failed to save results: {}",
            "warning_cannot_delete": "Warning: Cannot delete {}: {}",
            "success": "Success",
            "starting_analysis": "Starting analysis...",
            "saving_results": "Saving results...",
            "init_success": "System initialized, all caches, temporary files and logs have been cleared.",
            "init_failed": "System initialization failed: {}",
            "provide_video_path": "Please provide video file path, or use --init parameter to initialize system",
            "execution_failed": "Program execution failed: {}",
            "result_saved": "Results saved to: {}",
            "start_processing": "Start processing video: {}"
        }
        
        # Simplified Chinese translations
        self.translations["zh_CN"] = {
            "video_processing": "正在处理视频...",
            "audio_processing": "正在处理音频...",
            "text_processing": "正在处理文本...",
            "processing_complete": "处理完成",
            "error_file_not_found": "错误：文件未找到",
            "error_processing_failed": "错误：处理失败",
            "error_loading_config": "加载配置失败：{}",
            "error_saving_results": "保存结果失败：{}",
            "warning_cannot_delete": "警告：无法删除 {}: {}",
            "success": "成功",
            "starting_analysis": "开始分析...",
            "saving_results": "正在保存结果...",
            "init_success": "系统已初始化，所有缓存、临时文件和日志已清除。",
            "init_failed": "初始化系统失败：{}",
            "provide_video_path": "请提供视频文件路径，或使用 --init 参数初始化系统",
            "execution_failed": "程序执行失败：{}",
            "result_saved": "结果已保存到：{}",
            "start_processing": "开始处理视频：{}"
        }
        
        # Korean translations
        self.translations["ko"] = {
            "video_processing": "비디오 처리 중...",
            "audio_processing": "오디오 처리 중...",
            "text_processing": "텍스트 처리 중...",
            "processing_complete": "처리 완료",
            "error_file_not_found": "오류: 파일을 찾을 수 없습니다",
            "error_processing_failed": "오류: 처리 실패",
            "error_loading_config": "구성 로드 실패: {}",
            "error_saving_results": "결과 저장 실패: {}",
            "warning_cannot_delete": "경고: {} 삭제할 수 없음: {}",
            "success": "성공",
            "starting_analysis": "분석 시작...",
            "saving_results": "결과 저장 중...",
            "init_success": "시스템이 초기화되었으며, 모든 캐시, 임시 파일 및 로그가 삭제되었습니다.",
            "init_failed": "시스템 초기화 실패: {}",
            "provide_video_path": "비디오 파일 경로를 제공하거나 --init 매개변수를 사용하여 시스템을 초기화하십시오",
            "execution_failed": "프로그램 실행 실패: {}",
            "result_saved": "결과가 저장됨: {}",
            "start_processing": "비디오 처리 시작: {}"
        }

    def set_language(self, lang_code: str):
        """Set the current language"""
        if lang_code in self.translations:
            self.current_lang = lang_code
        else:
            raise ValueError(f"Unsupported language code: {lang_code}")

    def get(self, key: str, *args) -> str:
        """Get translation for the given key"""
        try:
            text = self.translations[self.current_lang][key]
            if args:
                return text.format(*args)
            return text
        except KeyError:
            return key

# Global i18n instance
_i18n = I18n()

def get_text(key: str, *args) -> str:
    """Get translation for the given key"""
    return _i18n.get(key, *args)

def set_language(lang_code: str):
    """Set the current language"""
    _i18n.set_language(lang_code)
