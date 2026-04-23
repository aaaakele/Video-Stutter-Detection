from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import zipfile
import xml.etree.ElementTree as ET

from src.detectors.base_detector import StutterEvent


@dataclass
class EvalStats:
    precision: float
    recall: float
    f1_score: float
    tp: int
    fp: int
    fn: int


@dataclass
class GroundTruthEvent:
    start_sec: int
    end_sec: int


@dataclass
class EvaluationReport:
    detector: str
    stats: EvalStats
    matched_pairs: list[dict]


def _load_gt_from_json(gt_path: Path) -> list[GroundTruthEvent]:
    with gt_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if isinstance(payload, dict):
        items = payload.get("events", [])
    else:
        items = payload
    events: list[GroundTruthEvent] = []
    for item in items:
        if "start_sec" in item and "end_sec" in item:
            start_sec = int(item["start_sec"])
            end_sec = int(item["end_sec"])
            if end_sec <= start_sec:
                end_sec = start_sec + 1
            events.append(GroundTruthEvent(start_sec=start_sec, end_sec=end_sec))
        elif "start_ms" in item and "end_ms" in item:
            start_sec = int(float(item["start_ms"]) // 1000)
            end_sec = int(float(item["end_ms"]) // 1000)
            if end_sec <= start_sec:
                end_sec = start_sec + 1
            events.append(GroundTruthEvent(start_sec=start_sec, end_sec=end_sec))
    return events


def _column_index(cell_ref: str) -> int:
    letters = "".join(char for char in cell_ref if char.isalpha())
    value = 0
    for char in letters:
        value = value * 26 + (ord(char.upper()) - ord("A") + 1)
    return max(0, value - 1)


def _read_shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        raw = archive.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    namespace = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    values: list[str] = []
    for si in root.findall("ns:si", namespace):
        parts = [node.text or "" for node in si.findall(".//ns:t", namespace)]
        values.append("".join(parts))
    return values


def _read_first_sheet_rows(xlsx_path: Path) -> list[list[str]]:
    namespace = {
        "ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
        "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
    }
    with zipfile.ZipFile(xlsx_path) as archive:
        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        relations = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        shared_strings = _read_shared_strings(archive)

        first_sheet = workbook.find("ns:sheets/ns:sheet", namespace)
        if first_sheet is None:
            return []
        rel_id = first_sheet.attrib.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        if not rel_id:
            return []

        target = None
        for rel in relations.findall("pkg:Relationship", namespace):
            if rel.attrib.get("Id") == rel_id:
                target = rel.attrib.get("Target")
                break
        if not target:
            return []

        sheet_path = target if target.startswith("xl/") else f"xl/{target}"
        sheet_root = ET.fromstring(archive.read(sheet_path))

        rows: list[list[str]] = []
        for row in sheet_root.findall("ns:sheetData/ns:row", namespace):
            row_values: list[str] = []
            current_col = 0
            for cell in row.findall("ns:c", namespace):
                cell_ref = cell.attrib.get("r", "A1")
                col_idx = _column_index(cell_ref)
                while current_col < col_idx:
                    row_values.append("")
                    current_col += 1

                cell_type = cell.attrib.get("t")
                value_node = cell.find("ns:v", namespace)
                value = value_node.text if value_node is not None else ""
                if cell_type == "s" and value:
                    resolved = shared_strings[int(value)]
                elif cell_type == "inlineStr":
                    text_node = cell.find("ns:is/ns:t", namespace)
                    resolved = text_node.text if text_node is not None else ""
                else:
                    resolved = value or ""
                row_values.append(str(resolved).strip())
                current_col += 1
            if any(item != "" for item in row_values):
                rows.append(row_values)
        return rows


def _find_column(headers: list[str], aliases: list[str]) -> int | None:
    normalized = [header.strip().lower() for header in headers]
    for alias in aliases:
        alias_lower = alias.lower()
        for index, header in enumerate(normalized):
            if header == alias_lower:
                return index
    return None


def _parse_time_to_sec(value: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError("Empty time text")

    parts = text.split(":")
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = float(parts[1])
        total_seconds = minutes * 60 + seconds
    elif len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        total_seconds = hours * 3600 + minutes * 60 + seconds
    else:
        total_seconds = float(text)
    return int(total_seconds)


def _parse_time_range_to_event(value: str) -> GroundTruthEvent:
    normalized = value.strip().replace("：", ":").replace("—", "-").replace("–", "-").replace("~", "-")
    parts = [item.strip() for item in normalized.split("-") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"Unsupported time range format: {value}")
    start_sec = _parse_time_to_sec(parts[0])
    end_sec = _parse_time_to_sec(parts[1])
    if end_sec <= start_sec:
        end_sec = start_sec + 1
    return GroundTruthEvent(start_sec=start_sec, end_sec=end_sec)


def _load_gt_from_xlsx(gt_path: Path) -> list[GroundTruthEvent]:
    rows = _read_first_sheet_rows(gt_path)
    if len(rows) < 2:
        return []

    headers = rows[0]
    start_idx = _find_column(headers, ["start_ms", "start", "开始", "开始时间", "起始时间"])
    end_idx = _find_column(headers, ["end_ms", "end", "结束", "结束时间", "终止时间"])
    range_idx = _find_column(headers, ["卡顿时间", "时间区间", "时间间隔", "stutter_time", "time_range"])

    events: list[GroundTruthEvent] = []
    if range_idx is not None:
        for row in rows[1:]:
            if range_idx >= len(row):
                continue
            raw_value = row[range_idx].strip()
            if not raw_value:
                continue
            events.append(_parse_time_range_to_event(raw_value))
        return events

    if start_idx is None or end_idx is None:
        raise ValueError(f"Unsupported GT sheet columns: {headers}")

    for row in rows[1:]:
        if start_idx >= len(row) or end_idx >= len(row):
            continue
        start_raw = row[start_idx].strip()
        end_raw = row[end_idx].strip()
        if not start_raw or not end_raw:
            continue
        start_sec = int(float(start_raw))
        end_sec = int(float(end_raw))
        if end_sec <= start_sec:
            end_sec = start_sec + 1
        events.append(GroundTruthEvent(start_sec=start_sec, end_sec=end_sec))
    return events


def load_ground_truth(gt_path: str | Path) -> list[GroundTruthEvent]:
    path = Path(gt_path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        return _load_gt_from_json(path)
    if suffix == ".xlsx":
        return _load_gt_from_xlsx(path)
    raise ValueError(f"Unsupported ground truth format: {path}")


def has_overlap(pred: StutterEvent, gt: GroundTruthEvent) -> bool:
    return pred.start_sec < gt.end_sec and gt.start_sec < pred.end_sec


def evaluate_events(detector: str, predicted: list[StutterEvent], ground_truth: list[GroundTruthEvent]) -> EvaluationReport:
    matched_gt: set[int] = set()
    matched_pairs: list[dict] = []
    tp = 0
    fp = 0

    for pred in predicted:
        hit_index = None
        for index, gt_item in enumerate(ground_truth):
            if index in matched_gt:
                continue
            if has_overlap(pred, gt_item):
                hit_index = index
                break

        if hit_index is None:
            fp += 1
            continue

        tp += 1
        matched_gt.add(hit_index)
        gt_item = ground_truth[hit_index]
        matched_pairs.append(
            {
                "pred": asdict(pred),
                "gt": asdict(gt_item),
            }
        )

    fn = max(0, len(ground_truth) - len(matched_gt))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return EvaluationReport(
        detector=detector,
        stats=EvalStats(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            tp=tp,
            fp=fp,
            fn=fn,
        ),
        matched_pairs=matched_pairs,
    )
