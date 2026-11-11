#!/usr/bin/env python3
"""Evaluate a localization journal against an explicitly aligned reference CSV."""
from __future__ import annotations
import argparse
import bisect
import csv
import json
import math
from pathlib import Path
from localization_journal import wrap


def reference_csv(path):
    """Explicit contract: ts,x,y,yaw_rad; coordinates and clock match journal."""
    with Path(path).open() as stream:
        reader = csv.DictReader(stream)
        if not {"ts", "x", "y", "yaw_rad"} <= set(reader.fieldnames or []):
            raise ValueError("reference CSV requires ts,x,y,yaw_rad (metres/radians)")
        rows = []
        for row in reader:
            values = tuple(float(row[k]) for k in ("ts", "x", "y", "yaw_rad"))
            if not all(math.isfinite(v) for v in values):
                raise ValueError("reference values must be finite")
            if rows and values[0] <= rows[-1][0]:
                raise ValueError("reference timestamps must strictly increase")
            rows.append(values)
    if not rows:
        raise ValueError("reference is empty")
    return rows


def interpolate(rows, stamps, ts, max_gap):
    i = bisect.bisect_left(stamps, ts)
    if i < len(rows) and abs(stamps[i] - ts) < 1e-7:
        return rows[i][1:]
    if i == 0 or i == len(rows) or stamps[i] - stamps[i-1] > max_gap:
        return None
    a, b = rows[i-1], rows[i]
    f = (ts-a[0])/(b[0]-a[0])
    return (a[1]+f*(b[1]-a[1]), a[2]+f*(b[2]-a[2]), wrap(a[3]+f*wrap(b[3]-a[3])))


def weighted_stats(values):
    if not values or sum(w for _, w in values) <= 0:
        return None
    values = sorted(values)
    total = sum(w for _, w in values)
    def percentile(q):
        cumulative = 0
        for value, weight in values:
            cumulative += weight
            if cumulative >= total*q:
                return value
        return values[-1][0]
    return {"median": percentile(.5), "p95": percentile(.95), "max": max(v for v, _ in values),
            "rmse": math.sqrt(sum(v*v*w for v, w in values)/total)}


def episodes(intervals, minimum):
    merged = []
    for start, end, reason in intervals:
        if end <= start:
            continue
        if merged and abs(merged[-1]["end_s"] - start) < 1e-8 and merged[-1]["reason"] == reason:
            merged[-1]["end_s"] = end
        else:
            merged.append({"start_s": start, "end_s": end, "reason": reason})
    return [{**e, "duration_s": e["end_s"]-e["start_s"]} for e in merged
            if e["end_s"]-e["start_s"] >= minimum - 1e-8]


def evaluate(path, reference=None, reference_offset=0.0):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    starts = [r for r in records if r["kind"] == "run_start"]
    if len(starts) != 1 or any(r["run_id"] != starts[0]["run_id"] for r in records):
        raise ValueError("journal must contain exactly one run")
    cfg = starts[0]["scenario"]
    samples = [r for r in records if r["kind"] == "sample"]
    if len(samples) < 2:
        raise ValueError("at least two samples are required")
    if any(b["time_s"] < a["time_s"] for a, b in zip(samples, samples[1:])):
        raise ValueError("clock moved backwards: split the recording into separate runs")
    start = cfg.get("route_start_s", samples[0]["time_s"])
    ends = [r["time_s"] for r in records if r["kind"] == "run_end" and r.get("time_s") is not None]
    end = cfg.get("route_end_s", ends[-1] if ends else samples[-1]["time_s"])
    if end <= start:
        raise ValueError("empty route interval")
    refs = reference or []
    stamps = [r[0]+reference_offset for r in refs]
    refs = [(r[0]+reference_offset, *r[1:]) for r in refs]
    duration = end-start
    available = covered = position_good = orientation_good = 0.0
    pos_errors, angle_errors, signed_angles = [], [], []
    failures, loss_intervals, exceedances = [], [], []
    details = []
    # Include unobserved beginning and end of the declared route in availability.
    boundaries = [start] + sorted({r["time_s"] for r in samples if start < r["time_s"] < end}) + [end]
    sample_times = [r["time_s"] for r in samples]
    for a, b in zip(boundaries, boundaries[1:]):
        i = bisect.bisect_right(sample_times, a)-1
        sample = samples[i] if i >= 0 else None
        # A fresh pose only remains available until its age limit expires.
        valid_until = min(b, sample["stamp_s"]+cfg["max_pose_age_s"]) if sample and sample["fresh"] else a
        valid_until = max(a, valid_until)
        if valid_until < b:
            loss_intervals.append((valid_until, b, "unavailable"))
            failures.append((valid_until, b, "failure"))
        weight = valid_until-a
        if weight <= 0:
            continue
        available += weight
        ref = interpolate(refs, stamps, sample["stamp_s"], cfg["max_reference_gap_s"]) if refs else None
        if ref is None:
            continue
        covered += weight
        pose = sample["pose"]
        pe = math.hypot(pose[0]-ref[0], pose[1]-ref[1])
        ae_signed = math.degrees(wrap(pose[2]-ref[2]))
        ae = abs(ae_signed)
        pos_errors.append((pe, weight)); angle_errors.append((ae, weight)); signed_angles.append((ae_signed, weight))
        pok = pe <= cfg["position_error_m"]
        aok = cfg["orientation_error_deg"] is not None and ae <= cfg["orientation_error_deg"]
        position_good += weight if pok else 0
        orientation_good += weight if aok else 0
        exceeded = not pok or (cfg["orientation_error_deg"] is not None and not aok)
        if exceeded:
            failures.append((a, valid_until, "failure"))
        for metric, error, threshold in (("position_m", pe, cfg["position_error_m"]),
                                         ("orientation_deg", ae, cfg["orientation_error_deg"])):
            if threshold is not None and error > threshold:
                exceedances.append((a, valid_until, metric))
        details.append({"time_s": a, "end_s": valid_until, "stamp_s": sample["stamp_s"],
                        "position_error_m": pe, "orientation_error_deg": ae,
                        "position_threshold_m": cfg["position_error_m"],
                        "orientation_threshold_deg": cfg["orientation_error_deg"]})
    # Interleaved metric intervals must be merged independently.
    error_events = []
    for metric in ("position_m", "orientation_deg"):
        for e in episodes([v for v in exceedances if v[2] == metric], cfg["min_event_duration_s"]):
            matches = [d for d in details if d["time_s"] < e["end_s"] and d["end_s"] > e["start_s"]]
            field = "position_error_m" if metric == "position_m" else "orientation_error_deg"
            threshold = cfg["position_error_m"] if metric == "position_m" else cfg["orientation_error_deg"]
            error_events.append({**e, "event": "error_threshold_exceeded", "scenario_id": cfg["scenario_id"],
                                 "peak_error": max(d[field] for d in matches), "threshold": threshold})
    failures.sort()
    failure_events = episodes(failures, cfg["min_event_duration_s"])
    loss_events = episodes(loss_intervals, cfg["min_event_duration_s"])
    availability = available/duration
    coverage = covered/duration
    accuracy_ready = bool(refs) and coverage >= cfg["min_reference_coverage"] and cfg["orientation_error_deg"] is not None
    pf = position_good/covered if covered else None
    af = orientation_good/covered if covered and cfg["orientation_error_deg"] is not None else None
    threshold_fraction = cfg["required_within_threshold_fraction"]
    passed = bool(ends) and accuracy_ready and availability >= cfg["min_availability"] and pf >= threshold_fraction and af >= threshold_fraction and len(failure_events) <= cfg["max_failure_episodes"]
    if passed:
        classification = "successful_single_run"
    elif available == 0:
        classification = "unsuccessful"
    elif not accuracy_ready or not ends:
        classification = "insufficient_evidence"
    elif pf >= threshold_fraction or af >= threshold_fraction or position_good > 0 or orientation_good > 0:
        classification = "partially_successful"
    else:
        classification = "unsuccessful"
    # Circular dispersion of signed orientation residual; removes constant bias.
    stability = None
    if signed_angles:
        sx = sum(math.sin(math.radians(v))*w for v,w in signed_angles)/covered
        cx = sum(math.cos(math.radians(v))*w for v,w in signed_angles)/covered
        mean = math.atan2(sx, cx)
        stability = math.sqrt(sum(math.degrees(wrap(math.radians(v)-mean))**2*w for v,w in signed_angles)/covered)
    return {"schema": "fliftbot.localization_report.v1", "run_id": starts[0]["run_id"],
            "scenario": cfg, "metadata": starts[0].get("metadata", {}),
            "route_start_s": start, "route_end_s": end, "duration_s": duration,
            "availability": availability, "reference_coverage": coverage,
            "position_error_m": weighted_stats(pos_errors), "orientation_error_deg": weighted_stats(angle_errors),
            "orientation_residual_std_deg": stability,
            "position_within_threshold_fraction": pf, "orientation_within_threshold_fraction": af,
            "loss_count": len(loss_events), "failure_count": len(failure_events),
            "loss_duration_s": sum(e["duration_s"] for e in loss_events),
            "failure_duration_s": sum(e["duration_s"] for e in failure_events),
            "loss_events": loss_events, "failure_events": failure_events,
            "error_events": error_events, "comparison_samples": details,
            "diagnostic_events": [r for r in records if r["kind"] == "event"],
            "classification": classification, "single_run_passed": passed,
            "repeatability": "not_assessed_requires_repeated_route_runs",
            "reference": "external_reference_disagreement" if refs else None,
            "reference_time_offset_s": reference_offset,
            "complete": bool(ends)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--journal", required=True, type=Path)
    p.add_argument("--reference", type=Path, help="CSV ts,x,y,yaw_rad in journal clock/frame")
    p.add_argument("--reference-time-offset-s", type=float, default=0.0, help="added to reference timestamps")
    p.add_argument("--output", required=True, type=Path, help="new output directory")
    args = p.parse_args()
    if not math.isfinite(args.reference_time_offset_s):
        p.error("reference time offset must be finite")
    result = evaluate(args.journal, reference_csv(args.reference) if args.reference else None, args.reference_time_offset_s)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"report.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    fields = ["time_s", "end_s", "stamp_s", "position_error_m", "orientation_error_deg", "position_threshold_m", "orientation_threshold_deg"]
    with (args.output/"errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(result["comparison_samples"])
    with (args.output/"events.jsonl").open("w") as stream:
        for event in result["error_events"] + result["failure_events"]:
            stream.write(json.dumps(event, ensure_ascii=False)+"\n")
    lines = [f"# Сценарий: {result['scenario']['scenario_id']}",
             f"Классификация: {result['classification']}",
             f"Доступность: {result['availability']:.2%}; покрытие эталоном: {result['reference_coverage']:.2%}.",
             f"Срывы: {result['failure_count']}; длительность: {result['failure_duration_s']:.3f} с.",
             f"Ошибка положения, м: {result['position_error_m']}",
             f"Ошибка ориентации, градусы: {result['orientation_error_deg']}",
             f"Разброс ошибки ориентации, градусы: {result['orientation_residual_std_deg']}",
             "Повторяемость требует повторных проездов. Сравнение показывает расхождение с указанным эталоном."]
    (args.output/"report.md").write_text("\n\n".join(lines)+"\n")
    print(json.dumps({k: result[k] for k in ("classification", "availability", "failure_count")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
