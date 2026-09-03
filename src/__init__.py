"""
Dental Outpatient Record OCR Package (口腔门诊病历智能识别与结构化提取系统)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

from .parser import (
    clean_forbidden_text,
    parse_record_sections,
    record_text_to_json,
)
from .rule_based_ocr import (
    ConversionOutput,
    convert_outpatient_file_to_txt,
    convert_outpatient_pdf_to_txt,
    convert_outpatient_pdfs_parallel,
    get_cached_ocr,
    process_document,
    process_pdf,
)

__all__ = [
    "DentalOutpatientOCR",
    "ConversionOutput",
    "convert_outpatient_file_to_txt",
    "convert_outpatient_pdf_to_txt",
    "convert_outpatient_pdfs_parallel",
    "get_cached_ocr",
    "process_document",
    "process_pdf",
    "record_text_to_json",
    "parse_record_sections",
    "clean_forbidden_text",
]


class DentalOutpatientOCR:
    """
    口腔门诊病历 OCR 识别器高层开发接口。
    
    支持输入扫描 PDF 或高清图片（PNG/JPG/WEBP），通过图像分割、形态学牙科符号解算与
    文字 OCR 时空对齐，输出标准纯文字全文与规范结构化 JSON 对象。
    """

    def __init__(self, debug_timing: bool = False):
        self.debug_timing = debug_timing
        self.ocr = get_cached_ocr(debug_timing=debug_timing)

    def recognize(
        self,
        file_path: Union[str, Path],
        output_root: Union[str, Path] = "outputs",
        export_review_png: bool = True,
        export_debug_artifacts: bool = True,
    ) -> Dict[str, Any]:
        """
        对单份门诊病历执行端到端识别转换。

        参数:
            file_path: 输入文件路径（支持 .pdf, .png, .jpg, .jpeg, .webp 等）
            output_root: 输出保存根目录
            export_review_png: 是否导出包含行标与符号解析连线的复核审查图
            export_debug_artifacts: 是否导出逐页 JSON 与 txt 符号坐标调试数据

        返回:
            包含纯文本、结构化字典与输出文件路径的综合结果字典
        """
        result = convert_outpatient_file_to_txt(
            pdf_path=file_path,
            output_root=output_root,
            export_review_png=export_review_png,
            export_debug_artifacts=export_debug_artifacts,
            debug_timing=self.debug_timing,
        )
        record_text = Path(result.record_path).read_text(encoding="utf-8")
        structured_data = record_text_to_json(record_text)

        return {
            "text": record_text,
            "structured": structured_data,
            "record_path": str(result.record_path),
            "review_paths": [str(p) for p in result.review_paths],
            "case_dir": str(result.case_dir),
            "page_count": result.page_count,
        }
