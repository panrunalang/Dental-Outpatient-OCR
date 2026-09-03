#!/usr/bin/env python3
"""
Dental Outpatient Record OCR - 命令行入口脚本 (CLI)
支持单文件/多文件/目录批量识别，输出纯文字、可视化审查图及结构化 JSON。
"""

import argparse
import json
import sys
import time
from pathlib import Path

# 确保 src 在导入路径中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.parser import record_text_to_json
from src.rule_based_ocr import (
    OUT_DIR,
    RAW_DIR,
    SUPPORTED_EXTENSIONS,
    convert_outpatient_file_to_txt,
    convert_outpatient_pdfs_parallel,
    get_cached_ocr,
    process_document,
    resolve_input_files,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="口腔门诊病历（非牙周大表）图文转纯文本识别系统 (Dental Outpatient EMR OCR)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 1. 识别默认示例图像并导出可视化审查图
  python run_ocr.py --input examples/inputs/sample_record.png

  # 2. 识别 PDF 并输出结构化 JSON
  python run_ocr.py --input examples/inputs/sample_record.pdf --json

  # 3. 批量处理目录下所有病历并指定输出目录
  python run_ocr.py --input examples/inputs/ --output-dir outputs/ --review-png --json
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        default=str(RAW_DIR / "sample_record.png"),
        help="输入文件路径（支持 .pdf, .png, .jpg, .jpeg, .webp）或包含文件的目录。默认: examples/inputs/sample_record.png",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=str(OUT_DIR),
        help="结果输出目录。默认: outputs/",
    )
    parser.add_argument(
        "--review-png",
        action="store_true",
        default=True,
        help="是否生成带有文字行与牙科符号引出线对照的可视化审查图 (默认开启)",
    )
    parser.add_argument(
        "--no-review-png",
        dest="review_png",
        action="store_false",
        help="关闭生成可视化审查图",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=True,
        help="是否额外输出结构化病历板块 JSON 文件 (默认开启)",
    )
    parser.add_argument(
        "--debug-artifacts",
        action="store_true",
        help="导出逐页符号坐标等调试用中间文件",
    )
    parser.add_argument(
        "--debug-timing",
        action="store_true",
        help="在终端打印各阶段流水线详细耗时日志",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="批量处理时启用多进程并行加速",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="并行处理的工作进程数",
    )
    return parser


def print_banner():
    banner = r"""
========================================================================
   ____             _        _    ___   ____ ____  
  |  _ \  ___ _ __ | |_ __ _| |  / _ \ / ___|  _ \ 
  | | | |/ _ \ '_ \| __/ _` | | | | | | |   | |_) |
  | |_| |  __/ | | | || (_| | | | |_| | |___|  _ < 
  |____/ \___|_| |_|\__\__,_|_|  \___/ \____|_| \_\
   Dental Outpatient Record Rule-based OCR System
========================================================================
    """
    print(banner)


def main() -> None:
    print_banner()
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    try:
        files_to_process = resolve_input_files(input_path)
    except Exception as e:
        print(f"[错误] 输入路径解析失败: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"[*] 待处理文件数量: {len(files_to_process)}")
    print(f"[*] 输出目录: {output_root.resolve()}")
    print(f"[*] 选项: 可视化审查图={'开启' if args.review_png else '关闭'}, 结构化JSON={'开启' if args.json else '关闭'}")
    print("-" * 72)

    total_start = time.perf_counter()

    if args.parallel and len(files_to_process) > 1:
        print(f"[*] 启动并行批处理模式 (workers={args.max_workers or '自动'})...")
        results = convert_outpatient_pdfs_parallel(
            pdf_paths=files_to_process,
            output_root=output_root,
            export_review_png=args.review_png,
            export_debug_artifacts=args.debug_artifacts,
            max_workers=args.max_workers,
            debug_timing=args.debug_timing,
        )
        for res in results:
            print(f"[✓] 完成: {res.pdf_path.name} -> {res.record_path}")
            if args.json:
                text = Path(res.record_path).read_text(encoding="utf-8")
                parsed_json = record_text_to_json(text)
                json_path = res.case_dir / "structured_record.json"
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(parsed_json, fh, ensure_ascii=False, indent=2)
    else:
        ocr = get_cached_ocr(debug_timing=args.debug_timing)
        for idx, file_item in enumerate(files_to_process, start=1):
            print(f"\n[{idx}/{len(files_to_process)}] 正在处理: {file_item.name} ...")
            start_t = time.perf_counter()
            res = process_document(
                ocr=ocr,
                pdf_path=file_item,
                output_root=output_root,
                export_review_png=args.review_png,
                export_debug_artifacts=args.debug_artifacts,
                debug_timing=args.debug_timing,
            )
            elapsed = time.perf_counter() - start_t
            print(f"[✓] 识别完成 (耗时: {elapsed:.2f}s, 共 {res.page_count} 页)")
            print(f"    ├─ 纯文字结果: {res.record_path}")
            if res.review_paths:
                for r_path in res.review_paths:
                    print(f"    ├─ 可视化审查: {r_path}")

            if args.json:
                text = Path(res.record_path).read_text(encoding="utf-8")
                parsed_json = record_text_to_json(text)
                json_path = res.case_dir / "structured_record.json"
                with open(json_path, "w", encoding="utf-8") as fh:
                    json.dump(parsed_json, fh, ensure_ascii=False, indent=2)
                print(f"    └─ 结构化JSON: {json_path}")

            # 打印部分纯文本预览
            lines = Path(res.record_path).read_text(encoding="utf-8").splitlines()
            preview = [l for l in lines if l.strip()][:5]
            if preview:
                print("    [纯文字片段预览]:")
                for p_line in preview:
                    print(f"      | {p_line[:80]}{'...' if len(p_line) > 80 else ''}")

    total_elapsed = time.perf_counter() - total_start
    print("-" * 72)
    print(f"[+] 全部任务已完成! 总耗时: {total_elapsed:.2f}s")


if __name__ == "__main__":
    main()
