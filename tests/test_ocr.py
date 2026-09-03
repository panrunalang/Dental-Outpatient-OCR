#!/usr/bin/env python3
"""
自动化测试与功能验证脚本 (Automated Test Suite)
用于快速自检口腔门诊病历 OCR 流水线、符号解析与结构化字段提取。
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

# 添加项目根目录到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import DentalOutpatientOCR, record_text_to_json


class TestDentalOutpatientOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_png = PROJECT_ROOT / "examples/inputs/sample_record.png"
        cls.sample_pdf = PROJECT_ROOT / "examples/inputs/sample_record.pdf"
        cls.output_dir = PROJECT_ROOT / "examples/outputs/test_run"
        cls.engine = DentalOutpatientOCR(debug_timing=False)

    def test_sample_files_exist(self):
        """验证示例文件是否完备"""
        self.assertTrue(self.sample_png.exists(), f"示例图片不存在: {self.sample_png}")
        self.assertTrue(self.sample_pdf.exists(), f"示例PDF不存在: {self.sample_pdf}")

    def test_pipeline_png_recognition(self):
        """验证 PNG 图像端到端识别流水线"""
        res = self.engine.recognize(
            file_path=self.sample_png,
            output_root=self.output_dir,
            export_review_png=True,
        )

        self.assertIsNotNone(res["text"])
        self.assertGreater(len(res["text"]), 100, "识别出的文本长度过短")

        # 检查关键牙科符号是否被正确解码并融入文本
        text = res["text"]
        self.assertIn("17", text, "未识别到17号牙")
        self.assertIn("PD", text, "未识别到牙周探诊深度PD标识")
        self.assertIn("FI", text, "未识别到根分叉病变FI标识")

        # 检查是否生成审查图
        self.assertTrue(len(res["review_paths"]) > 0, "未生成可视化审查图")
        for review_path in res["review_paths"]:
            self.assertTrue(Path(review_path).exists(), f"审查图文件不存在: {review_path}")

        # 检查结构化输出
        structured = res["structured"]
        self.assertEqual(structured["患者"], "张小明", "患者姓名解析不符合预期")
        self.assertIn("牙周牙髓联合病变", structured.get("诊断", ""), "诊断内容解析不符合预期")
        self.assertIn("17、37、48", text, "L18 未能完整识别出 17、37、48")

    def test_parser_structuring(self):
        """验证病历文本板块正则切分与标准化"""
        sample_txt_path = PROJECT_ROOT / "examples/outputs/sample_record.txt"
        if not sample_txt_path.exists():
            self.skipTest("sample_record.txt 尚未生成")

        raw_text = sample_txt_path.read_text(encoding="utf-8")
        parsed = record_text_to_json(raw_text)

        self.assertEqual(parsed["日期"], "2026-09-03")
        self.assertEqual(parsed["患者"], "张小明")
        self.assertIn("牙周科", parsed.get("科室", ""))
        self.assertIn("牙周牙髓联合病变", parsed.get("诊断", ""))
        self.assertIn("种植义齿", parsed.get("治疗计划", ""))
        self.assertIn("口腔颌面外科", parsed.get("处置", ""))

    def test_cli_execution(self):
        """验证命令行 CLI 工具运行是否正常"""
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "run_ocr.py"),
            "--input",
            str(self.sample_png),
            "--output-dir",
            str(self.output_dir),
            "--no-review-png",
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, f"CLI 执行失败: {result.stderr}")
        self.assertIn("全部任务已完成", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
