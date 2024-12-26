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
            "video_analyzer": "Video Analyzer",
            "language_selection": "Language",
            "file_selection": "File Selection",
            "video_file": "Video File:",
            "output_file": "Output File:",
            "browse": "Browse",
            "init_system": "Initialize System",
            "process_video": "Process Video",
            "select_video": "Select Video File",
            "select_output": "Select Output File",
            "init_confirm": "This will clear all caches and logs. Continue?",
            "processing": "Processing...",
            "process_complete": "Processing complete. Results saved to",
            "process_error": "Error processing video",
            "select_video_prompt": "Please select a video file",
            "select_output_prompt": "Please select an output file",
            "error": "Error",
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
            "video_analyzer": "视频分析器",
            "language_selection": "语言",
            "file_selection": "文件选择",
            "video_file": "视频文件：",
            "output_file": "输出文件：",
            "browse": "浏览",
            "init_system": "初始化系统",
            "process_video": "处理视频",
            "select_video": "选择视频文件",
            "select_output": "选择输出文件",
            "init_confirm": "这将清除所有缓存和日志。是否继续？",
            "processing": "处理中...",
            "process_complete": "处理完成。结果已保存至",
            "process_error": "处理视频时出错",
            "select_video_prompt": "请选择视频文件",
            "select_output_prompt": "请选择输出文件",
            "error": "错误",
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
            "video_analyzer": "비디오 분석기",
            "language_selection": "언어",
            "file_selection": "파일 선택",
            "video_file": "비디오 파일:",
            "output_file": "출력 파일:",
            "browse": "찾아보기",
            "init_system": "시스템 초기화",
            "process_video": "비디오 처리",
            "select_video": "비디오 파일 선택",
            "select_output": "출력 파일 선택",
            "init_confirm": "모든 캐시와 로그가 지워집니다. 계속하시겠습니까?",
            "processing": "처리 중...",
            "process_complete": "처리 완료. 결과 저장 위치:",
            "process_error": "비디오 처리 오류",
            "select_video_prompt": "비디오 파일을 선택하세요",
            "select_output_prompt": "출력 파일을 선택하세요",
            "error": "오류",
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
