"""
口腔门诊病历纯文本结构化字段解析模块
基于规则引擎与正则表达式，将 OCR 重建后的全文转化为标准临床病历 JSON 对象。
完全本地运行，不依赖外部大语言模型或在线 API。
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

FIELD_KEYS = [
    "日期",
    "患者",
    "出生日期",
    "性别",
    "科室",
    "主诉",
    "现病史",
    "既往史",
    "个人史",
    "家族史",
    "全身",
    "复诊",
    "检查",
    "诊断",
    "治疗计划",
    "处置",
]

DATE_PATTERN = re.compile(
    r"(?P<year>(?:19|20)\d{2})[-年./](?P<month>\d{1,2})[-月./](?P<day>\d{1,2})"
)

SECTION_PATTERN = re.compile(
    r"(日期|患者|出生日期|性别|科室|主诉|现病史|既往史|个人史|家族史|全身|复诊|检查|诊断|治疗计划|处置)\s*[:：.．]"
)

SECTION_HEADING_PATTERNS = [
    (
        key,
        re.compile(
            rf"(?P<prefix>^|[\s]){''.join(f'{re.escape(ch)}\\s*' for ch in key).rstrip('\\s*')}(?P<colon>[:：.．])",
            flags=re.MULTILINE,
        ),
    )
    for key in FIELD_KEYS
]

BARE_EMBEDDED_SECTION_KEYS = {"诊断", "治疗计划", "处置"}
TOOTH_NOISE_PATTERN = r"[1-8][1-8](?:、[1-8][1-8])*(?:[BLMDO])?"


def clean_forbidden_text(text: Optional[str]) -> Optional[str]:
    """
    清理不需要保留的敏感/冗余信息：
    主索引、签名、诊治医师、电话号码。
    """
    if text is None:
        return None
    text = str(text)
    patterns = [
        r"主索引\s*[:：]?\s*[^；;，,。.\n]*",
        r"签名\s*[:：]?\s*[^；;，,。.\n]*",
        r"诊治医师\s*[:：]?\s*[^；;，,。.\n]*",
        r"电话\s*[:：]?\s*[^；;，,。.\n]*",
    ]
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[，,]\s*(?:0\d{2,3}-?)?\d{8,}(?=\s*$)", "", text)
    return text


def build_section_heading_body(key: str) -> str:
    return "".join(f"{re.escape(ch)}\\s*" for ch in key).rstrip("\\s*")


def normalize_section_headings(text: str) -> str:
    normalized = str(text or "")
    for key, pattern in SECTION_HEADING_PATTERNS:
        normalized = pattern.sub(
            lambda m: f"{m.group('prefix')}{key}{m.group('colon')}", normalized
        )
    return normalized


def normalize_text(value: Optional[str]) -> Optional[str]:
    """标准化病历文本排版与标点符号。"""
    if value is None:
        return None
    text = clean_forbidden_text(str(value))
    if not text:
        return None
    # 常见 OCR 词汇纠偏
    text = text.replace("煎合", "愈合")
    text = text.replace("IIIII", "II–III°")
    text = text.replace("LII", "TM: II")
    text = text.replace("II=LL", "TM: II")
    text = text.replace("II:WL", "TM: II")
    text = text.replace("LI", "TM: I")
    text = text.replace("L工", "TM: I")
    text = text.replace("I:WL", "TM: I")
    text = text.replace(":LL", "TM: 0")
    text = text.replace("LTM", "TM")
    text = text.replace("JLL", "TM")
    text = text.replace("WL", "TM")
    text = text.replace("工", "I")
    text = text.replace(":O", ":0")
    text = text.replace("：14mm", "1–4mm ")
    text = text.replace("：15mm", "1–5mm ")
    text = text.replace("：16mm", "1–6mm ")
    text = text.replace("：17mm", "1–7mm ")
    text = text.replace("：18mm", "1–8mm ")
    text = text.replace("：19mm", "1–9mm ")
    text = text.replace("4%阿替卡因", " 4%阿替卡因")
    text = text.replace("余余", "余")
    text = text.replace("口口腔", "口腔")
    text = text.replace("。线示", "。X线示")

    text = text.replace("\u3000", " ").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff0-9])\s+(?=[\u4e00-\u9fff0-9])", "", text)
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[A-Za-z])", "", text)
    text = re.sub(r"(?<=[A-Za-z])\s+(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s+([，。；：、,.!?）)】》])", r"\1", text)
    text = re.sub(r"([（(【《])\s+", r"\1", text)
    text = re.sub(r"\s+([）)】》])", r"\1", text)
    text = text.strip(" \n\t；;，,。")
    return text or None


def clean_section_noise(value: Optional[str], field_key: str) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None

    text = re.sub(
        r"(?:0\d{2,3}-?)?\d{7,8}(?:、(?:0\d{2,3}-?)?\d{7,8})*(?=\s*$)",
        "",
        text,
    )

    if field_key == "检查":
        text = re.sub(
            rf"(?<=口腔卫生)\s*{TOOTH_NOISE_PATTERN}\s*(?=一般)",
            "",
            text,
        )

    if field_key == "处置":
        text = re.sub(
            rf"(?<=[；;，,\s]){TOOTH_NOISE_PATTERN}\s*(?=医嘱)",
            "",
            text,
        )
        text = re.sub(
            rf"{TOOTH_NOISE_PATTERN}\s*$",
            "",
            text,
        )

    if field_key in {"主诉", "现病史", "复诊"}:
        text = re.sub(
            rf"((?:无|未见|未诉)(?:明显)?不适)\s*{TOOTH_NOISE_PATTERN}\s*$",
            r"\1",
            text,
        )

    text = re.sub(r"\s{2,}", " ", text)
    text = text.strip(" \n\t；;，,。")
    return text or None


def merge_text(old_text: Optional[str], new_text: Optional[str]) -> Optional[str]:
    old_text = normalize_text(old_text)
    new_text = normalize_text(new_text)
    if not new_text:
        return old_text
    if not old_text:
        return new_text
    if new_text in old_text:
        return old_text
    if old_text in new_text:
        return new_text
    return f"{old_text} {new_text}".strip()


def normalize_record_text(record_text: str) -> str:
    text = str(record_text or "")
    text = clean_forbidden_text(text) or ""
    text = text.replace("\u3000", " ").replace("\r", "\n")
    text = normalize_section_headings(text)
    text = text.replace("门（急）诊病历", " ").replace("门(急)诊病历", " ")
    text = re.sub(r"(?m)^\s*广州医科大学附属口腔医院\s*$", " ", text)
    text = re.sub(r"(?m)^\s*0\d{8,}\s*$", " ", text)
    text = re.sub(r"(?m)^\s*\d{8,}\s*$", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def flatten_record_text(record_text: str) -> str:
    text = normalize_record_text(record_text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def extract_date(text: Optional[str], output_format: str = "visit") -> Optional[str]:
    match = DATE_PATTERN.search(str(text or ""))
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    if output_format == "visit":
        return f"{year:04d}-{month:02d}-{day:02d}"
    return f"{year:04d}年{month:02d}月{day:02d}日"


def extract_gender(text: Optional[str]) -> Optional[str]:
    match = re.search(r"[男女]", str(text or ""))
    return match.group(0) if match else None


def normalize_department(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    text = text.replace("（", "(").replace("）", ")")
    text = re.sub(r"\s+", "", text)
    return text or None


def normalize_numbered_section_text(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    text = re.sub(r"^\s*([1-9])[．、.]\s*", r"\1. ", text)
    text = re.sub(r"(?<!^)(?<!\d)\s*([2-9])[．、.]\s*", r" \1. ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip() or None


def normalize_diagnosis_text(value: Optional[str]) -> Optional[str]:
    text = normalize_numbered_section_text(value)
    if not text:
        return None
    text = re.sub(
        r"(?<=[\u4e00-\u9fff])(?P<num>[3-9])(?P=num)\.(?=\s*[1-8][1-8])",
        lambda m: f" {m.group('num')}. ",
        text,
    )
    text = re.sub(r"(?<!^)(?<!\d)([2-9])\.(?=\S)", r" \1. ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip() or None


def normalize_check_text(value: Optional[str]) -> Optional[str]:
    text = normalize_text(value)
    if not text:
        return None
    text = re.sub(r"(BI[:：=]\s*[0-4][–-][0-4])(?=[1-8][1-8])", r"\1，", text)
    return text.strip() or None


def parse_record_sections(record_text: str) -> Tuple[Dict[str, Optional[str]], str]:
    """把连续文本切分成各大病历板块。"""
    flat_text = flatten_record_text(record_text)
    sections: Dict[str, Optional[str]] = {key: None for key in FIELD_KEYS}
    matches = list(SECTION_PATTERN.finditer(flat_text))

    for idx, match in enumerate(matches):
        key = match.group(1)
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(flat_text)
        value = normalize_text(flat_text[start:end])
        if value:
            sections[key] = merge_text(sections.get(key), value)

    return sections, flat_text


def record_text_to_json(
    record_text: str,
) -> Dict[str, Any]:
    """
    将 OCR 纯文本直接转化为结构化病历字典。
    
    参数:
        record_text: OCR 重建后的全文纯文本
    
    返回:
        结构化病历数据字典
    """
    sections, flat_text = parse_record_sections(record_text)
    data: Dict[str, Any] = {key: None for key in FIELD_KEYS}

    data["日期"] = extract_date(sections.get("日期") or flat_text, output_format="visit")
    data["出生日期"] = extract_date(sections.get("出生日期") or flat_text, output_format="birth")
    data["性别"] = extract_gender(sections.get("性别") or flat_text)
    data["患者"] = normalize_text(sections.get("患者"))
    data["科室"] = normalize_department(sections.get("科室"))

    for key in [
        "主诉",
        "现病史",
        "既往史",
        "个人史",
        "家族史",
        "全身",
        "复诊",
    ]:
        data[key] = clean_section_noise(sections.get(key), key)

    data["检查"] = clean_section_noise(normalize_check_text(sections.get("检查")), "检查")
    data["诊断"] = normalize_diagnosis_text(sections.get("诊断"))
    data["治疗计划"] = normalize_numbered_section_text(sections.get("治疗计划"))
    data["处置"] = clean_section_noise(normalize_numbered_section_text(sections.get("处置")), "处置")

    return data
