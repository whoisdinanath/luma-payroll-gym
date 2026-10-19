import json
import math
import os
import re
import statistics
from pathlib import Path

from luma_gym.scoring import RUBRIC_VERSION
from luma_gym.skills import bootstrap, measure_evaluation, paired


RESULTS = Path(os.environ.get("LUMA_RESULTS", "results"))
ROOT = RESULTS.parent
MODELS = {
    "deepseek/deepseek-v4.1-flash": ("DeepSeek V4.1 Flash", "DeepSeek", "#2563eb"),
    "openai/gpt-5.6-luna": ("GPT 5.6 Luna", "Luna", "#0d9488"),
    "xiaomi/mimo-v2.6-pro": ("MiMo V2.6 Pro", "MiMo", "#d97706"),
    "z-ai/glm-5.3-flash": ("GLM 5.3 Flash", "GLM", "#7c3aed"),
    "google/gemini-3.1-flash-lite-preview": ("Gemini 3.1 Flash Lite", "Gemini", "#e11d48"),
}
BASELINES = [
    ("reference_with_empty_explanation", "Reference (no explanation)", "Ceiling"),
    ("right_decision_no_arithmetic", "Right decision, no arithmetic", "Decisions only"),
    ("refuse_payslips_perfect_traps", "Refuse payslips, perfect traps", "Refusal"),
    ("always_clarify_conflict", "Always clarify: conflict", "Constant"),
    ("format_only", "Always calculate: zeros", "Format only"),
    ("copy_example", "Copy prompt example", "Demo copy"),
]
TIERS = {1: "Retrieval", 2: "Calculation", 3: "Traps"}
TRAPS = {
    "conflict": "conflicting pay",
    "draft_amendment": "draft amendment",
    "future_amendment": "future amendment",
    "missing_gross": "missing gross pay",
    "missing_month": "missing month",
    "uncovered_month": "uncovered month",
}
FIELDS = {
    "employee_contribution": "contribution",
    "taxable_pay": "taxable pay",
    "income_tax": "income tax",
    "net_pay": "net pay",
    "value": "retrieved value",
    "allowed_deduction": "deduction",
    "gross_pay": "gross pay",
    "currency": "currency",
    "field": "field",
    "reason_code": "reason",
}
ANSWER_FIELDS = [
    "decision",
    "currency",
    "gross_pay",
    "employee_contribution",
    "allowed_deduction",
    "taxable_pay",
    "income_tax",
    "net_pay",
    "field",
    "value",
    "reason_code",
    "clauses",
]
REVIEWED = [
    {"title": "Invented a gross pay that no document contains", "tag": "Fabrication", "count": 19,
     "examples": [("v4-hosted-gemini-tools", "case4-4041-00012")]},
    {"title": "Guessed a rate row for a record with no month", "tag": "Calibration", "count": 31,
     "examples": [("v4-hosted-gemini-tools", "case4-4041-00011")]},
    {"title": "Used an amendment that takes effect later", "tag": "Rule order", "count": 14,
     "examples": [("v4-hosted-gemini-tools", "case4-4041-00035")]},
    {"title": "Right structure, wrong bracket arithmetic", "tag": "Arithmetic", "count": 69,
     "examples": [("v4-hosted-gemini-single", "case4-4041-00005"), ("v4-hosted-gemini-single", "case4-4041-00006")]},
    {"title": "Right answer wrapped in Markdown or prose", "tag": "Format", "count": 78,
     "examples": [("v4-hosted-glm-single", "case4-4041-00002"), ("v4-hosted-deepseek-tools", "case4-4041-00040"),
                  ("v4-hosted-glm-tools", "case4-4041-00001")]},
    {"title": "One cent off exactly at the tax-free line", "tag": "Boundary", "count": 2,
     "examples": [("v4-hosted-luna-default-single", "case4-4041-00102")]},
    {"title": "Income tax off by a factor of ten", "tag": "Arithmetic", "count": 1,
     "examples": [("v4-hosted-mimo-single", "case4-4041-00009")]},
]
SPLITS = {
    "heldout": ROOT / "data/training-splits/held_out.json",
    "development": ROOT / "data/training-splits/development.json",
}
V4 = RESULTS / "v4"
HOSTED = ["deepseek", "mimo", "gemini", "glm", "luna-default"]
STRATA_LABELS = {
    "bracket_zero": "below tax-free line", "bracket_middle": "middle bracket", "bracket_upper": "upper bracket",
    "cap_binding": "wage cap binds", "boundary_cap": "at the wage cap", "boundary_free": "at the tax-free line",
    "boundary_upper": "at the upper bracket line", "rounding_tie": "half-cent rounding tie",
}
REASON_SHORT = {"MISSING_GROSS_PAY": "missing gross pay", "MISSING_PAYROLL_MONTH": "missing month",
                "NO_APPLICABLE_REFERENCE": "month not covered", "CONFLICTING_GROSS_PAY": "conflicting pay"}
STAGES = [("raw-single", "Frozen, single-turn", "#a1a1aa"), ("raw-calculator", "Frozen + calculator", "#71717a"),
          ("sft-calculator", "SFT + calculator", "#0d9488"), ("rl-0.04-calculator", "SFT + RL, β 0.04", "#2563eb"),
          ("rl-0.001-calculator", "SFT + RL, β 0.001", "#e11d48"), ("rl-1-calculator", "SFT + RL, β 1", "#65a30d"),
          ("rl-10-calculator", "SFT + RL, β 10", "#0891b2"), ("rl-100-calculator", "SFT + RL, β 100", "#4f46e5"),
          ("rl-0.04-balanced-calculator", "SFT + RL, β 0.04, balanced", "#d97706"),
          ("rl-0.04-signal-calculator", "SFT + RL, β 0.04, more payslips", "#c026d3")]
HALF_LABELS = {"first-half": "January to June", "second-half": "July to December"}
STAGE_METRICS = [("exact_calculation", "Exact payslips", True), ("correct_clarification", "Correct clarifications", True),
                 ("false_clarification", "Refuses good payslips", False), ("false_calculation", "Calculates through traps", False),
                 ("invalid", "Malformed", False), ("tool_use_on_payslips", "Calculator used on payslips", True)]
FINAL_CURVE_SPECS = [
    ("reward", "Mean reward", "percent"),
    ("calculation", "Payslip pass rate", "percent"),
    ("false_clarification", "Refuses good payslips", "percent"),
    ("kl", "KL divergence (k3)", "number"),
    ("entropy", "Policy entropy", "number"),
    ("completion_tokens", "Completion length (tokens)", "number"),
    ("trap", "Trap pass rate", "percent"),
    ("false_calculation", "Calculates through traps", "percent"),
    ("signal", "Non-tied groups", "percent"),
]
CURVE_SPECS = [
    ("reward", "Mean reward", "percent"),
    ("kl", "KL divergence (k3)", "number"),
    ("entropy", "Policy entropy", "number"),
    ("completion_tokens", "Completion length (tokens)", "number"),
    ("trap", "Trap pass rate", "percent"),
    ("calculation", "Payslip pass rate", "percent"),
    ("correct_clarification", "Correct clarifications", "percent"),
    ("false_calculation", "Calculates through traps", "percent"),
    ("signal", "Non-tied groups", "percent"),
]
SKILL_METRICS = [
    ("correct_clarification", "Correct Clarification (Traps)", True),
    ("false_calculation", "Calculates on Traps", False),
    ("false_clarification", "Refuses good payslips", False),
    ("exact_calculation", "Exact Payslip Calculation", True),
    ("invalid", "Malformed JSON Output", False),
    ("reward", "Mean Rubric Reward", True),
]
EPISODES = [
    {
        "run": "rl-rubric-3-beta-0.04",
        "rule": "Rubric 3: credit for every field and the clauses",
        "exploit": "Calculated 39 of 40 traps with the worked example's values (50.00, 950.00)",
        "evidence": {"min_step": 11, "tier": 3, "decision": "calculate", "contains": "950.00"},
        "fix": "Rubric 4: credit only for computed values",
    },
    {
        "run": "rl-rubric-4-beta-0.04",
        "rule": "Rubric 4: credit only for computed values",
        "exploit": "Every group tied at 0 in 11 of 14 steps",
        "evidence": {"min_step": 1, "tier": 2, "decision": "retrieve"},
        "fix": "Rubric 5: 0.05 for the right decision",
    },
    {
        "run": "rl-beta-0.04",
        "rule": "Rubric 5: 0.05 for the right decision",
        "exploit": "Calculated on 99.5% of held-out traps",
        "evidence": {"min_step": 95, "tier": 3, "decision": "calculate"},
        "fix": "Rubric 6: only exact answers score; six worked examples",
    },
    {
        "run": "rl-rubric-6-beta-0.04",
        "rule": "Rubric 6: only exact answers score",
        "exploit": "Refused 99.0% of held-out payslips",
        "evidence": {"min_step": 18, "tier": 2, "decision": "clarify"},
        "fix": "Rubric 7: wrong decisions cost -0.5 both ways",
    },
    {
        "run": "rl-rubric-7-sample-weighted-beta-0.04",
        "rule": "Rubric 7, loss weighted per answer",
        "exploit": "Trap groups of 16 answers outweighed payslip groups 4.6 to 1",
        "evidence": {"min_step": 14, "tier": 2, "decision": "clarify"},
        "fix": "Every group weighted equally",
    },
]


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def read_lines(path):
    path = Path(path)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(value, digits=0):
    if value is None:
        return "-"
    return f"{value * 100:.{digits}f}%"


def signed(value, digits=1):
    if value is None:
        return "-"
    return f"{value * 100:+.{digits}f}"


def heat(value):
    if value is None:
        return "#e4e4e7"
    stops = [
        (-0.5, (225, 29, 72)),
        (0.0, (249, 115, 22)),
        (0.5, (234, 179, 8)),
        (0.95, (34, 197, 94)),
        (1.0, (16, 185, 129)),
    ]
    value = min(max(value, -0.5), 1.0)
    for (low, start), (high, end) in zip(stops, stops[1:]):
        if value <= high:
            share = (value - low) / (high - low) if high > low else 0
            return "#" + "".join(f"{round(a + (b - a) * share):02x}" for a, b in zip(start, end))
    return "#10b981"


def soft_heat(value):
    if value is None:
        return "#f4f4f5"
    if value >= 0.95:
        return "#dcfce7"
    if value >= 0.70:
        return "#fef9c3"
    if value >= 0.0:
        return "#ffedd5"
    return "#ffe4e6"


def text_heat(value):
    if value is None:
        return "#71717a"
    if value >= 0.95:
        return "#166534"
    if value >= 0.70:
        return "#854d0e"
    if value >= 0.0:
        return "#9a3412"
    return "#9f1239"


def ticks(low, high, count=4):
    high = max(high, low + 1e-6)
    rough = (high - low) / count
    base = 10 ** math.floor(math.log10(rough))
    step = next(base * factor for factor in (1, 2, 2.5, 5, 10) if base * factor >= rough * (1 - 1e-9))
    low = math.floor(low / step + 1e-9) * step
    high = math.ceil(high / step - 1e-9) * step
    return low, high, [round(low + step * index, 12) for index in range(round((high - low) / step) + 1)]


def model_info(model):
    fallback = model.split("/")[-1]
    return MODELS.get(model, (fallback, fallback, "#71717a"))


def load_tasks():
    tasks = {}
    for index, item in enumerate(read_json(ROOT / "data/cases.json", [])):
        detail = ""
        if item["difficulty"] == 1:
            detail = (item.get("retrieval_field") or "").replace("_", " ")
        elif item["difficulty"] == 3:
            detail = next((TRAPS[tag] for tag in item["tags"] if tag in TRAPS), "")
        else:
            parts = ["cap boundary"] if "boundary" in item["tags"] else []
            if "amendment" in item["record"]:
                parts.append("amended")
            detail = ", ".join(parts) or "standard"
        tasks[item["id"]] = {
            "id": item["id"],
            "number": index + 1,
            "tier": item["difficulty"],
            "detail": detail,
            "truth": item["ground_truth"],
            "record": item["record"],
        }
    for index, item in enumerate(read_json(ROOT / "data/v4/eval.json", [])):
        tags = item["tags"]
        if item["difficulty"] == 1:
            detail = (item.get("retrieval_field") or "").replace("_", " ")
        elif item["difficulty"] == 3:
            variant = next((tag[9:].replace("_", " ") for tag in tags if tag.startswith("conflict:")), "")
            detail = REASON_SHORT[item["ground_truth"]["reason_code"]] + (f" ({variant})" if variant and variant != "plain" else "")
        else:
            detail = STRATA_LABELS[next(tag[8:] for tag in tags if tag.startswith("stratum:"))]
            if "amendment-resolved" in tags:
                detail += ", resolved amendment"
        tasks[item["id"]] = {"id": item["id"], "number": index + 1, "tier": item["difficulty"], "detail": detail,
                             "truth": item["ground_truth"], "record": item["record"]}
    for index, item in enumerate(read_json(ROOT / "data/transfer-human.json", [])):
        tasks[item["id"]] = {"id": item["id"], "number": index + 1, "tier": item["difficulty"],
                             "detail": "hand-written transfer", "truth": item["ground_truth"], "record": item["record"]}
    return tasks


def failure(score, tier):
    if not score:
        return "error"
    status = score["status"]
    if status == "invalid_format":
        return "malformed"
    if status == "wrong_decision":
        return "answered trap" if tier == 3 else "wrong decision"
    if status == "wrong_clarification":
        return "wrong reason"
    if score["breakdown"]["exact"]["score"] < 1:
        wrong = [field for field in FIELDS if field in score["fields"] and not score["fields"][field]["correct"]]
        return f"wrong {FIELDS[wrong[0]]}" if wrong else "inexact"
    if score["breakdown"]["citations"]["score"] < 1:
        return "citations"
    explanation = score["breakdown"].get("explanation")
    if explanation and explanation["score"] < 0.5:
        return "explanation"
    return None


def summarize(rows):
    rows = [row for row in rows if row.get("score")]
    if not rows:
        return None
    by_task = {}
    for row in rows:
        by_task.setdefault(row["task_id"], []).append(row["score"]["score"])
    task_means = {task: statistics.mean(values) for task, values in by_task.items()}
    rewards = [row["score"]["score"] for row in rows]
    low, high = bootstrap(list(task_means.values()))
    tiers = {}
    for row in rows:
        tiers.setdefault(row["tier"], []).append(row["score"]["score"])
    scored = [row["score"] for row in rows]
    decided = [score for score in scored if score["diagnostics"]["correct_decision"]]
    judged = [score["breakdown"]["explanation"]["score"] for score in scored if "explanation" in score["breakdown"]]
    failures = {}
    for row in rows:
        kind = failure(row["score"], row["tier"])
        if kind:
            failures[kind] = failures.get(kind, 0) + 1
    return {
        "answers": len(rows),
        "tasks": task_means,
        "task_tiers": {row["task_id"]: row["tier"] for row in rows},
        "mean": statistics.mean(rewards),
        "sd": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "low": low,
        "high": high,
        "tiers": {tier: statistics.mean(values) for tier, values in tiers.items()},
        "exact": statistics.mean(score["breakdown"]["exact"]["score"] == 1 for score in scored),
        "fields": statistics.mean(score["breakdown"]["field_accuracy"]["score"] for score in decided) if decided else 0.0,
        "clauses": statistics.mean(
            score["breakdown"]["citations"]["score"] for score in decided if score["status"] == "scored"
        )
        if any(s["status"] == "scored" for s in decided)
        else 0.0,
        "explanation": statistics.mean(judged) if judged else 0.0,
        "malformed": statistics.mean(score["status"] == "invalid_format" for score in scored),
        "wrong_decision": statistics.mean(score["status"] == "wrong_decision" for score in scored),
        "failures": dict(sorted(failures.items(), key=lambda item: -item[1])),
    }


def load_run(directory, run_id=None):
    config = read_json(directory / "config.json", {})
    rows = read_json(directory / "scores.json", [])
    summary = summarize(rows) if config else None
    if not summary:
        return None
    costs = [row.get("cost_usd") or 0 for row in rows]
    latencies = [row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None]
    summary.update(
        id=run_id or directory.name,
        model=config["model"],
        mode=config["mode"],
        default_sampling=config.get("temperature") is None,
        rows=rows,
        cost_per_100=sum(costs) / len(rows) * 100,
        latency=statistics.median(latencies) if latencies else None,
    )
    return summary


def load_models():
    models = {}
    sources = []
    for name in HOSTED:
        for mode, key in (("single", "single"), ("tools", "tools"), ("tools-nocalc", "nocalc")):
            sources.append((V4 / f"hosted-{name}-{mode}", f"v4-hosted-{name}-{mode}", key))
        sources.append((V4 / f"transfer-human-{name}", f"v4-transfer-human-{name}", "transfer"))
    for directory, run_id, key in sources:
        run = load_run(directory, run_id) if directory.is_dir() else None
        if not run:
            continue
        name, short, color = model_info(run["model"])
        model = models.setdefault(
            run["model"],
            {"id": run["model"], "name": name, "short": short, "color": color, "default_sampling": False},
        )
        model[key] = run
        model["default_sampling"] = model["default_sampling"] or run["default_sampling"]
    ranked = sorted((model for model in models.values() if "single" in model), key=lambda model: -model["single"]["mean"])
    for rank, model in enumerate(ranked, 1):
        model["rank"] = rank
    return ranked


def load_baselines(tasks):
    results = read_json(ROOT / "reports/baselines-v4.json", {}).get("eval", {}).get("results", {})
    rows = []
    for key, name, note in BASELINES:
        scores = results.get(key)
        if not scores:
            continue
        summary = summarize(
            [
                {"task_id": score["task_id"], "tier": tasks[score["task_id"]]["tier"], "score": score}
                for score in scores
                if score["task_id"] in tasks
            ]
        )
        if not summary:
            continue
        summary.update(name=name, short=name, note=note, color="#71717a")
        rows.append(summary)
    return rows


def build_score_chart(entries):
    width, left, right = 340, 12, 12

    def x_pos(val):
        clamped = min(max(val, -0.5), 1.0)
        return round(left + (clamped - (-0.5)) / 1.5 * (width - left - right), 1)

    zero_x = x_pos(0.0)
    for row in entries:
        mean_x = x_pos(row["mean"])
        row["chart"] = {
            "zero_x": zero_x,
            "mean_x": mean_x,
            "low_x": x_pos(row["low"]),
            "high_x": x_pos(row["high"]),
            "bar_x": min(zero_x, mean_x),
            "bar_w": max(2.0, round(abs(mean_x - zero_x), 1)),
            "positive": row["mean"] >= 0,
        }
    axis = [{"x": x_pos(v), "label": pct(v, 0)} for v in (-0.5, 0.0, 0.5, 1.0)]
    return {"width": width, "zero_x": zero_x, "axis": axis}


def build_pareto_chart(models):
    width, height = 560, 260
    left, right, top, bottom = 52, 34, 24, 34
    costs = [m["single"]["cost_per_100"] for m in models]
    means = [m["single"]["mean"] for m in models]
    low_c, high_c, cost_marks = ticks(min(costs) * 0.8, max(costs) * 1.15, 4)
    low_m, high_m = min(0.6, min(means) - 0.05), 1.05

    def sx(cost):
        return round(left + (cost - low_c) / (high_c - low_c) * (width - left - right), 1)

    def sy(score):
        return round(top + (high_m - score) / (high_m - low_m) * (height - top - bottom), 1)

    points = []
    for m in models:
        s = m["single"]
        lat = s["latency"] or 5.0
        radius = round(6 + min(max(lat, 1.5), 11.0) * 0.65, 1)
        px, py = sx(s["cost_per_100"]), sy(s["mean"])
        points.append({
            "name": m["name"],
            "short": m["short"] + ("*" if m["default_sampling"] else ""),
            "color": m["color"],
            "x": px,
            "y": py,
            "low_y": sy(max(s["low"], low_m)),
            "high_y": sy(min(s["high"], high_m)),
            "r": radius,
            "cost": f"${s['cost_per_100']:.3f}",
            "mean": pct(s["mean"], 1),
            "latency": f"{lat:.1f}s",
        })
    frontier = sorted(points, key=lambda p: p["x"])
    pareto_pts = []
    best_y = 9999
    for pt in frontier:
        if pt["y"] <= best_y:
            pareto_pts.append(f"{pt['x']},{pt['y']}")
            best_y = pt["y"]
    x_ticks = [{"x": sx(c), "label": f"${c:.3f}"} for c in cost_marks]
    y_ticks = [{"y": sy(v), "label": pct(v, 0)} for v in (0.6, 0.7, 0.8, 0.9, 1.0) if v >= low_m]
    return {
        "width": width,
        "height": height,
        "left": left,
        "right": width - right,
        "top": top,
        "bottom": height - bottom,
        "points": points,
        "frontier": " ".join(pareto_pts),
        "x_ticks": x_ticks,
        "y_ticks": y_ticks,
    }


def build_tier_grouped_chart(models):
    categories = [
        ("Retrieval", lambda m: m["single"]["tiers"].get(1, 0)),
        ("Calculation", lambda m: m["single"]["tiers"].get(2, 0)),
        ("Traps", lambda m: m["single"]["tiers"].get(3, 0)),
        ("Tools + calculator", lambda m: m.get("tools", {}).get("mean", 0)),
        ("Transfer", lambda m: m.get("transfer", {}).get("mean", 0)),
    ]
    width, height = 560, 240
    left, right, top, bottom = 46, 16, 18, 32
    plot_w = width - left - right
    plot_h = height - top - bottom
    group_w = plot_w / len(categories)
    bar_w = 15

    def sy(val):
        clamped = max(0.0, min(1.0, val))
        return round(top + (1.0 - clamped) * plot_h, 1)

    groups = []
    for g_idx, (label, getter) in enumerate(categories):
        total_bars_w = len(models) * bar_w + (len(models) - 1) * 3
        gx = left + g_idx * group_w + (group_w - total_bars_w) / 2
        bars = []
        for m_idx, model in enumerate(models):
            val = getter(model)
            bx = round(gx + m_idx * (bar_w + 3), 1)
            by = sy(val)
            bh = max(2.0, round(height - bottom - by, 1))
            bars.append({
                "x": bx,
                "y": by,
                "w": bar_w,
                "h": bh,
                "color": model["color"],
                "model": model["short"] + ("*" if model["default_sampling"] else ""),
                "value": pct(val, 1),
            })
        groups.append({
            "label": label,
            "cx": round(left + (g_idx + 0.5) * group_w, 1),
            "bars": bars,
        })
    y_ticks = [{"y": sy(v), "label": pct(v, 0)} for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
    return {
        "width": width,
        "height": height,
        "left": left,
        "right": width - right,
        "bottom": height - bottom,
        "groups": groups,
        "y_ticks": y_ticks,
    }


def build_failure_stack(models):
    buckets = [
        ("passed", "Passed (≥95%)", "#10b981"),
        ("amounts", "Wrong amount", "#f59e0b"),
        ("malformed", "Malformed JSON", "#7c3aed"),
        ("answered trap", "Answered trap", "#e11d48"),
        ("wrong reason", "Wrong reason", "#f97316"),
        ("other", "Citations / explanation", "#94a3b8"),
    ]
    rows = []
    for model in models:
        single = model["single"]
        total = single["answers"]
        fails = single["failures"]
        failed_count = sum(fails.values())
        passed_count = total - failed_count
        amounts = sum(
            c for k, c in fails.items() if k.startswith("wrong ") and k not in ("wrong decision", "wrong reason")
        )
        other = sum(c for k, c in fails.items() if k in ("citations", "explanation", "inexact", "wrong decision"))
        counts = {
            "passed": passed_count,
            "amounts": amounts,
            "malformed": fails.get("malformed", 0),
            "answered trap": fails.get("answered trap", 0),
            "wrong reason": fails.get("wrong reason", 0),
            "other": other,
        }
        segments = []
        for key, label, color in buckets:
            c = counts[key]
            if c > 0:
                share = c / total * 100.0
                segments.append({
                    "key": key,
                    "label": label,
                    "color": color,
                    "count": c,
                    "w": round(share, 2),
                    "title": f"{model['short']}: {c}/{total} {label.lower()}",
                })
        rows.append({
            "model": model,
            "total": total,
            "failed": failed_count,
            "counts": counts,
            "segments": segments,
        })
    return {"buckets": [{"key": k, "label": l, "color": c} for k, l, c in buckets], "rows": rows}


def verdict(models):
    slices = [
        ("Overall", "single", None),
        ("Retrieval", "single", 1),
        ("Calculation", "single", 2),
        ("Traps", "single", 3),
        ("Tools + calculator", "tools", None),
        ("Transfer", "transfer", None),
    ]
    rows = []
    for label, key, tier in slices:
        entries = []
        for model in models:
            run = model.get(key)
            if not run:
                continue
            selected = [row for row in run["rows"] if row.get("score") and (tier is None or row["tier"] == tier)]
            if not selected:
                continue
            failures = {}
            for row in selected:
                kind = failure(row["score"], row["tier"])
                if kind:
                    failures[kind] = failures.get(kind, 0) + 1
            entries.append({
                "model": model,
                "mean": statistics.mean(row["score"]["score"] for row in selected),
                "cost": run["cost_per_100"],
                "answers": len(selected),
                "failures": failures,
            })
        if not entries:
            continue
        best = max(entries, key=lambda entry: (entry["mean"], -entry["cost"]))
        close = [entry for entry in entries if entry["mean"] >= best["mean"] - 0.01]
        value = min(close, key=lambda entry: entry["cost"])
        worst = min(entries, key=lambda entry: entry["mean"])
        main = max(worst["failures"].items(), key=lambda item: item[1]) if worst["failures"] else None
        rows.append({
            "label": label,
            "best": best,
            "value": value,
            "worst": worst if worst is not best else None,
            "failure": f"{main[0]} ({main[1]})" if main else "",
        })
    return rows


def heatmap(models, tasks, mode, threshold=0.95, only_failures=True):
    runs = [(model, model[mode]) for model in models if mode in model]
    if not runs:
        return None
    task_ids = sorted({task for model, run in runs for task in run["tasks"]}, key=lambda task: tasks[task]["number"])
    if only_failures:
        task_ids = [task for task in task_ids if any(run["tasks"].get(task, 1) < threshold for model, run in runs)]
    solved = len({task for model, run in runs for task in run["tasks"]}) - len(task_ids)
    groups = []
    for tier in (1, 2, 3):
        members = [task for task in task_ids if tasks[task]["tier"] == tier]
        members.sort(key=lambda task: statistics.mean(run["tasks"].get(task, 1) for model, run in runs))
        if members:
            groups.append((tier, members))
    left, top, cell, gap, row_height = 92, 26, 24, 5, 28
    x = left
    columns, bands = [], []
    for tier, members in groups:
        start = x
        for task in members:
            columns.append((task, x, tier))
            x += cell
        bands.append({"tier": tier, "label": TIERS[tier], "x": start, "width": x - start, "count": len(members)})
        x += gap * 3
    width = x - gap * 3 + 8
    cells = []
    for row, (model, run) in enumerate(runs):
        y = top + row * row_height
        for task, column, tier in columns:
            value = run["tasks"].get(task)
            if value is None:
                continue
            info = tasks[task]
            cells.append({
                "x": column,
                "y": y,
                "w": cell - 3,
                "h": row_height - 4,
                "tier": tier,
                "color": heat(value),
                "url": f"/transcript/{run['id']}/{task}",
                "title": f"Task {info['number']} ({TIERS[info['tier']]}, {info['detail']}): {model['short']} {pct(value, 0)}",
            })
    labels = [
        {
            "y": top + row * row_height + row_height / 2 + 1,
            "text": model["short"] + ("*" if model["default_sampling"] else ""),
            "color": model["color"],
        }
        for row, (model, run) in enumerate(runs)
    ]
    numbers = [{"x": column + (cell - 3) / 2, "text": tasks[task]["number"], "tier": tier} for task, column, tier in columns]
    height = top + len(runs) * row_height + 22
    return {
        "width": width,
        "height": height,
        "cells": cells,
        "labels": labels,
        "numbers": numbers,
        "bands": bands,
        "number_y": top + len(runs) * row_height + 14,
        "shown": len(task_ids),
        "solved": solved,
    }


def reviewed_cards(tasks):
    cards = []
    for item in REVIEWED:
        links = []
        for run, task in item["examples"]:
            model = read_json(RESULTS / run / "config.json", {}).get("model", "")
            links.append({
                "url": f"/transcript/{run}/{task}",
                "text": f"{model_info(model)[1]} #{tasks[task]['number']}",
            })
        cards.append({**item, "links": links})
    return cards


def line_chart(series, title, kind, steps, window=10):
    width, height, left, right, top, bottom = 380, 175, 44, 14, 12, 26
    lines = [(color, label, points) for color, label, points in series if points]
    if not lines:
        return None
    values = [value for color, label, points in lines for step, value in points]
    floor = 0.05 if kind == "percent" else 1e-3
    low, high, marks = ticks(min(0, min(values)), max(max(values) * 1.05, floor), 3)
    if kind == "percent" and high > 1:
        high, marks = 1, [mark for mark in marks if mark <= 1]

    def x(step):
        return round(left + step / max(steps, 1) * (width - left - right), 1)

    def y(value):
        return round(top + (high - value) / (high - low) * (height - top - bottom), 1)

    zero_y = y(0)
    paths = []
    for color, label_text, points in lines:
        smooth = []
        for index in range(len(points)):
            recent = [value for step, value in points[max(0, index - window + 1) : index + 1]]
            smooth.append(sum(recent) / len(recent))
        smooth_pts = [(step, val) for (step, _), val in zip(points, smooth)]
        poly = " ".join(f"{x(step)},{y(val)}" for step, val in smooth_pts)
        area = f"{x(smooth_pts[0][0])},{zero_y} " + poly + f" {x(smooth_pts[-1][0])},{zero_y}"
        last_step, last_val = smooth_pts[-1]
        paths.append({
            "color": color,
            "label": label_text,
            "raw": " ".join(f"{x(step)},{y(value)}" for step, value in points),
            "smooth": poly,
            "area": area,
            "end_x": x(last_step),
            "end_y": y(last_val),
            "dots": [{"x": x(step), "y": y(val)} for step, val in smooth_pts] if window == 1 else [],
        })
    fmt = (lambda value: pct(value, 1 if high <= 0.1 else 0)) if kind == "percent" else (lambda value: f"{value:g}")
    return {
        "title": title,
        "width": width,
        "height": height,
        "left": left,
        "right": width - right,
        "bottom": height - bottom,
        "paths": paths,
        "y_marks": [{"y": y(value), "label": fmt(value), "zero": value == 0} for value in marks],
        "x_marks": [
            {"x": x(0), "label": 0, "anchor": "start"},
            {"x": x(steps // 2), "label": steps // 2, "anchor": "middle"},
            {"x": x(steps), "label": steps, "anchor": "end"},
        ],
    }


def skill_checks():
    def count(name, tier):
        scores = read_json(V4 / name / "scores.json", [])
        rows = [row for row in scores if row["tier"] == tier]
        if not rows:
            return None
        exact = sum(row["score"]["breakdown"]["exact"]["score"] == 1.0 for row in rows)
        return f"{exact} of {len(rows)}"

    def share(name, tier):
        scores = read_json(V4 / name / "scores.json", [])
        rows = [row for row in scores if row["tier"] == tier]
        if not rows:
            return None
        return f"{sum(row['score']['breakdown']['exact']['score'] == 1.0 for row in rows) / len(rows):.1%}"

    models = (("raw", "Frozen"), ("sft", "SFT"), ("rl-0.04", "SFT + RL, β 0.04"), ("rl-0.04-signal", "SFT + RL, more payslips"))
    conditions = [
        ("Test set, six worked examples", "exact payslips", lambda run: share(f"test-{run}-calculator", 2)),
        ("Test set, no worked examples", "exact payslips", lambda run: share(f"test-{run}-calculator-noexamples", 2)),
        ("Hand-written requests", "exact payslips", lambda run: count(f"transfer-local-{run}-calculator", 2)),
        ("Hand-written requests", "traps right", lambda run: count(f"transfer-local-{run}-calculator", 3)),
        ("Same records, training instruction", "exact payslips", lambda run: count(f"transfer-fixed-line-{run}-calculator", 2)),
        ("Same records, training instruction", "traps right", lambda run: count(f"transfer-fixed-line-{run}-calculator", 3)),
        ("Hosted tool mode, no worked examples", "exact payslips", lambda run: count(f"eval-{run}-tools", 2)),
        ("Hosted tool mode, worked examples as tool calls", "exact payslips", lambda run: count(f"eval-{run}-tools-examples", 2)),
    ]
    rows = [{"condition": condition, "measure": measure, "cells": [value(run) for run, _ in models]}
            for condition, measure, value in conditions]
    return {"models": [label for _, label in models], "rows": rows}


def learning_curve():
    found = read_json(ROOT / "reports/protocol4-summary.json", {}).get("checkpoints", {})
    names = {key[:-len("-calculator")]: (color, label) for key, label, color in STAGES if key.startswith("rl-")}
    runs = [(run, *names.get(run, ("#71717a", run))) for run in found]
    charts = []
    for metric, title in (("exact_calculation", "Exact payslips"),
                          ("false_clarification", "Refuses good payslips"),
                          ("reward", "Mean reward")):
        series = [(color, label, [(point["step"], point[metric]) for point in found[run]]) for run, color, label in runs]
        chart = line_chart(series, title, "percent", 60, window=1)
        if chart:
            charts.append(chart)
    return {"charts": charts, "runs": [{"label": label, "color": color} for _, color, label in runs]} if charts else None


def curve_value(row, key):
    if key == "calculation":
        return (row.get("pass_rate_by_tier") or {}).get("2")
    if key == "trap":
        return (row.get("pass_rate_by_tier") or {}).get("3")
    if key == "signal":
        return 1 - row["tied_groups"]
    return row.get(key)


def run_label(config):
    if config.get("label"):
        return config["label"]
    rubric = config.get("rubric_version", "3")
    return f"β {config['beta']:g}, rubric {rubric}"


def load_training():
    runs = []
    for directory in sorted(RESULTS.glob("rl-*")) + sorted(V4.glob("rl-*")):
        config = read_json(directory / "config.json") if directory.is_dir() else None
        if not config or "beta" not in config or "smoke" in directory.name:
            continue
        steps = {row["step"]: row for row in read_lines(directory / "steps.jsonl")}
        rows = [steps[step] for step in sorted(steps)]
        current = config.get("protocol") == "4"
        more = config.get("calc_prompts", 2) != 2 or config.get("lr", 1e-5) != 1e-5
        label = (f"β {config['beta']:g}" + (", more payslips" if more else ", balanced" if "calc_sampling" in config else "")
                 + (f", seed {config['seed']}" if config.get("seed", 7) != 7 else ""))
        runs.append({
            "id": ("v4-" if directory.parent == V4 else "") + directory.name,
            "path": directory,
            "config": config,
            "steps": rows,
            "done": len(rows),
            "planned": config["steps"],
            "beta": config["beta"],
            "rubric": config.get("rubric_version", "3"),
            "current": current,
            "label": label if current else run_label(config),
            "title": f"SFT + RL, {label}" if current else run_label(config),
            "note": config.get("superseded", ""),
            "hours": sum(sum(row["seconds"].values()) for row in rows) / 3600,
        })
    final = sorted([run for run in runs if run["current"]], key=lambda run: (-run["beta"], run["label"]))
    colors = {key[:-len("-calculator")]: color for key, _, color in STAGES if key.startswith("rl-")}
    extra = iter(["#0d9488", "#9333ea", "#be123c", "#52525b"])
    for run in final:
        run["color"] = colors.get(run["path"].name) or next(extra, "#71717a")
    hist_palette = ["#7c3aed", "#d97706", "#0891b2", "#4f46e5", "#be123c", "#52525b"]
    h_idx = 0
    for run in runs:
        if "color" not in run:
            run["color"] = hist_palette[h_idx % len(hist_palette)]
            h_idx += 1
    return runs, final


def curves(runs, specs=CURVE_SPECS):
    if not runs:
        return []
    steps = max(run["planned"] for run in runs)
    charts = []
    for key, title, kind in specs:
        series = [
            (
                run["color"],
                run["title"],
                [(row["step"], curve_value(row, key)) for row in run["steps"] if curve_value(row, key) is not None],
            )
            for run in runs
        ]
        chart = line_chart(series, title, kind, steps)
        if chart:
            charts.append(chart)
    return charts


def evaluations(runs):
    measured = {}
    for split, path in SPLITS.items():
        for directory in sorted(RESULTS.glob(f"minicpm-{split}-*")):
            result = measure_evaluation(directory, path)
            if result:
                measured[(split, directory.name.removeprefix(f"minicpm-{split}-"))] = result

    groups = []
    for title, control, members in (
        ("6-example prompt, rubrics 6 and 7", "control", [run for run in runs if run["rubric"] in ("6", "7") and not run["note"]]),
        ("3-example prompt, rubric 5", "before", [run for run in runs if run["rubric"] == "5"]),
    ):
        base = measured.get(("heldout", control))
        if not base:
            continue
        rows = [
            {
                "label": "Frozen control",
                "color": "#71717a",
                "cells": [{"value": base[0][key]} for key, _, _ in SKILL_METRICS],
            }
        ]
        for run in sorted(members, key=lambda run: (not run["current"], -run["beta"], run["rubric"])):
            tag = run["id"].removeprefix("rl-")
            held = measured.get(("heldout", tag))
            development = measured.get(("development", tag))
            if not held and not development:
                rows.append({"label": run["title"], "color": run["color"], "pending": True, "url": f"/run/{run['id']}"})
                continue
            cells = []
            for key, label, good in SKILL_METRICS:
                if not held:
                    cells.append({"value": None})
                    continue
                change = paired(base[1][key], held[1][key])
                better = change and (change["difference"] > 0) == good and change["difference"] != 0
                clear = change and (change["interval_95"][0] > 0 or change["interval_95"][1] < 0)
                val = held[0][key]
                cells.append({
                    "value": val,
                    "change": change,
                    "tone": ("good" if better else "bad") if clear else "neutral",
                })
            rows.append({
                "label": run["title"],
                "color": run["color"],
                "cells": cells,
                "url": f"/run/{run['id']}",
                "steps": run["done"] if run["rubric"] == "6" else None,
            })
        groups.append({"title": title, "rows": rows})

    return groups


def protocol_four(split="test"):
    report = read_json(ROOT / "reports/protocol4-summary.json", {}).get("splits", {})
    chosen = split if report.get(split, {}).get("summaries") else "dev"
    summaries = report.get(chosen, {}).get("summaries", {})
    changes = report.get(chosen, {}).get("changes", {})
    stages = []
    for key, label, color in STAGES:
        summary = summaries.get(key)
        if not summary:
            continue
        base = "sft-calculator" if key.startswith("rl-") else "raw-calculator" if key == "sft-calculator" else None
        delta = changes.get(f"{base} -> {key}", {}) if base else {}
        cells = []
        for metric, title, good in STAGE_METRICS:
            value = summary.get(metric)
            change = delta.get(metric)
            clear = change and (change["interval_95"][0] > 0 or change["interval_95"][1] < 0)
            better = change and (change["difference"] > 0) == good
            cells.append({"value": value, "bar": round((value or 0) * 100, 1), "change": change,
                          "tone": ("good" if better else "bad") if clear else "neutral"})
        stages.append({"key": key, "label": label, "color": color, "cells": cells, "reward": summary.get("reward"),
                       "answers": summary.get("answers"), "strata": summary.get("exact_by_stratum", {}),
                       "reasons": summary.get("clarification_by_reason", {}),
                       "refusals": summary.get("refusal_by_contract", {}), "halves": summary.get("exact_by_half", {})})
    strata = [{"key": name, "label": label,
               "values": [{"color": stage["color"], "label": stage["label"], "value": stage["strata"].get(name),
                           "bar": round((stage["strata"].get(name) or 0) * 100, 1)}
                          for stage in stages if stage["key"] != "raw-single"]}
              for name, label in STRATA_LABELS.items()]
    halves = [{"key": name, "label": label,
               "values": [{"color": stage["color"], "label": stage["label"], "value": stage["halves"].get(name),
                           "bar": round((stage["halves"].get(name) or 0) * 100, 1)}
                          for stage in stages if not stage["key"].startswith("raw-")]}
              for name, label in HALF_LABELS.items()]
    errors = read_json(ROOT / "reports/protocol4-errors.json", {}).get(chosen, {})
    kinds = [("exact", "Exact", "#10b981"), ("refused", "Refused", "#7c3aed"),
             ("used the other half-year row", "Wrong half-year row", "#e11d48"), ("lines", "Wrong program line", "#f59e0b"),
             ("other", "No tool call, malformed, or copy error", "#94a3b8")]
    breakdown = []
    for stage in stages:
        found = errors.get(stage["key"])
        if not found:
            continue
        total = sum(found.values())
        counts = {"exact": found.get("exact", 0), "refused": found.get("refused", 0),
                  "used the other half-year row": found.get("used the other half-year row", 0),
                  "lines": sum(value for key, value in found.items() if key.startswith("wrong program line"))}
        counts["other"] = total - sum(counts.values())
        breakdown.append({"label": stage["label"], "color": stage["color"], "total": total,
                          "segments": [{"label": label, "color": color, "count": counts[key],
                                        "w": round(counts[key] / total * 100, 2)} for key, label, color in kinds if counts[key]]})
    return {"split": chosen, "stages": stages, "metrics": [title for _, title, _ in STAGE_METRICS], "strata": strata,
            "halves": halves,
            "breakdown": breakdown, "kinds": [{"label": label, "color": color} for _, label, color in kinds]}


def calculator_effect(models):
    rows = []
    for model in models:
        entry = {"model": model}
        for key in ("single", "nocalc", "tools"):
            run = model.get(key)
            if not run:
                continue
            payslips = [row["score"]["breakdown"]["exact"]["score"] == 1 for row in run["rows"] if row.get("score") and row["tier"] == 2]
            traps = [row["score"]["breakdown"]["exact"]["score"] == 1 for row in run["rows"] if row.get("score") and row["tier"] == 3]
            entry[key] = {"mean": run["mean"], "payslips": statistics.mean(payslips) if payslips else None,
                          "traps": statistics.mean(traps) if traps else None,
                          "cost": run["cost_per_100"], "answers": run["answers"]}
        rows.append(entry)
    return rows


def answer_decision(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return "invalid"
    return value.get("decision", "") if isinstance(value, dict) else "invalid"


def sample_answers(run_id):
    folder = V4 / run_id[3:] if run_id.startswith("v4-") else RESULTS / run_id
    return read_lines(folder / "samples.jsonl")


def episodes():
    rows = []
    for episode in EPISODES:
        evidence = episode["evidence"]
        candidates = [
            row
            for row in sample_answers(episode["run"])
            if row["step"] >= evidence["min_step"]
            and row["tier"] == evidence["tier"]
            and evidence.get("contains", "") in row["text"]
        ]
        sample = next(
            (row for row in candidates if (row.get("decision") or answer_decision(row["text"])) == evidence["decision"]),
            None,
        )
        if sample is None:
            sample = next(
                (
                    row
                    for row in sample_answers(episode["run"])
                    if row["step"] >= evidence["min_step"] and row["tier"] == evidence["tier"]
                ),
                None,
            )
        text = ""
        if sample:
            try:
                text = json.dumps(json.loads(sample["text"]))
            except ValueError:
                text = sample["text"]
        rows.append({
            **episode,
            "url": f"/run/{episode['run']}",
            "sample": text[:200] + ("..." if len(text) > 200 else ""),
            "sample_step": sample["step"] if sample else None,
        })
    return rows


def sweep(runs):
    rows = []
    for run in runs:
        if not (run["current"] or run["rubric"] in ("5", "7")) or run["note"] or not run["steps"]:
            continue
        last = run["steps"][-10:]
        kl_mean = statistics.mean(row["kl"] for row in last)
        max_kl = max(row["kl"] for row in run["steps"])
        ent_mean = statistics.mean(row["entropy"] for row in last)
        rew_mean = statistics.mean(row["reward"] for row in last)
        rows.append({
            "label": run["title"],
            "color": run["color"],
            "done": run["done"],
            "planned": run["planned"],
            "url": f"/run/{run['id']}",
            "kl": kl_mean,
            "kl_bar": round(min(kl_mean / 0.04, 1.0) * 100, 1),
            "max_kl": max_kl,
            "entropy": ent_mean,
            "reward": rew_mean,
        })
    return sorted(rows, key=lambda row: (not row["label"].startswith("SFT"), row["label"].endswith("rubric 5"), row["label"]))


def gym_facts(tasks):
    counts = {}
    for task in tasks.values():
        if task["id"].startswith("case4-4041"):
            counts[task["tier"]] = counts.get(task["tier"], 0) + 1
    audit = read_json(ROOT / "reports/payroll-task-audit.json", {})
    manifest = read_json(ROOT / "data/v4/manifest.json", {})
    original = read_json(ROOT / "reports/task-audit-original.json", {})
    crosscheck = read_json(ROOT / "reports/reference-crosscheck.json", {})
    hand = read_json(ROOT / "reports/fidelity-audit.json", {})
    judge_high = read_json(RESULTS / "judge-reliability-high-effort/metrics.json", {})
    transfer = read_json(ROOT / "reports/transfer.json", {})
    pairs = [pair["result"] for pair in (transfer.get("hand_written") or transfer).get("ranking_pairs", [])]
    gpu = sum(sum(row["seconds"].values()) for path in [*RESULTS.glob("rl-*/steps.jsonl"), *V4.glob("rl-*/steps.jsonl")]
              for row in read_lines(path))
    gpu += sum(row.get("seconds", 0) for path in [*RESULTS.glob("minicpm-*/generations.jsonl"), *V4.glob("*/generations.jsonl")]
               for row in read_lines(path))
    gpu += sum(row.get("seconds", 0) for row in read_lines(V4 / "sft/steps.jsonl"))
    sets = manifest.get("sets", {})
    return {
        "counts": counts,
        "sets": {name: item["tasks"] for name, item in sets.items()},
        "total_tasks": sum(item["tasks"] for item in sets.values()),
        "leaks": sum(item.get("prompt_leaks", 0) for item in sets.values()),
        "unique": sets.get("eval", {}).get("tasks"),
        "original_duplicates": original.get("semantic_duplicate_instances"),
        "crosscheck": crosscheck.get("divergence_rate"),
        "crosscheck_tasks": crosscheck.get("checked"),
        "hand_checked": hand.get("checked"),
        "hand_agreed": hand.get("agreed"),
        "judge_controlled": read_json(RESULTS / "judge-reliability-structured/metrics.json", {}),
        "judge_actual": read_json(RESULTS / "judge-actual-reliability/metrics.json", {}),
        "judge_high": judge_high,
        "transfer_concordant": pairs.count("concordant"),
        "transfer_ties": pairs.count("tie"),
        "transfer_discordant": pairs.count("discordant"),
        "gpu_hours": gpu / 3600,
        "spent": read_json(RESULTS / "spending.json", {}).get("spent_usd", 0),
    }


def load_home():
    tasks = load_tasks()
    models = load_models()
    baselines = load_baselines(tasks)
    entries = []
    for model in models:
        entries.append({
            **model["single"],
            "name": model["name"],
            "short": model["short"],
            "color": model["color"],
            "rank": model["rank"],
            "default_sampling": model["default_sampling"],
            "kind": "model",
            "tools": model.get("tools", {}).get("mean"),
            "transfer": model.get("transfer", {}).get("mean"),
        })
    for baseline in baselines:
        entries.append({**baseline, "kind": "baseline"})
    score_chart = build_score_chart(entries)
    runs, final = load_training()
    eval_groups = evaluations(runs)
    stages = protocol_four()
    by_key = {stage["key"]: stage for stage in stages["stages"]}
    best = by_key.get("rl-0.04-signal-calculator") or by_key.get("rl-0.04-calculator") or by_key.get("sft-calculator")
    leader = models[0] if models else None
    cheap = min((model for model in models if model["single"]["mean"] >= leader["single"]["mean"] - 0.01),
                key=lambda model: model["single"]["cost_per_100"]) if leader else None
    tool_leader = max((model for model in models if "tools" in model), key=lambda model: model["tools"]["mean"], default=None)
    kpis = {
        "leader": leader, "cheap": cheap, "tool_leader": tool_leader, "best_stage": best,
        "raw_exact": (by_key.get("raw-calculator") or {}).get("cells", [{}])[0].get("value"),
        "sft_exact": (by_key.get("sft-calculator") or {}).get("cells", [{}])[0].get("value"),
    }
    return {
        "kpis": kpis,
        "stages": stages,
        "calculator": calculator_effect(models),
        "models": models,
        "leaderboard": entries,
        "score_chart": score_chart,
        "pareto": build_pareto_chart(models),
        "tier_chart": build_tier_grouped_chart(models),
        "failure_stack": build_failure_stack(models),
        "verdict": verdict(models),
        "heatmap": heatmap(models, tasks, "single"),
        "tool_heatmap": heatmap(models, tasks, "tools"),
        "reviewed": reviewed_cards(tasks),
        "runs": runs,
        "final": final,
        "curves": curves(final, FINAL_CURVE_SPECS),
        "learning_curve": learning_curve(),
        "skill_checks": skill_checks(),
        "all_curves": curves(runs),
        "evaluations": eval_groups,
        "episodes": episodes(),
        "sweep": sweep(runs),
        "gym": gym_facts(tasks),
        "sandbox": read_json(ROOT / "reports/sandbox-judge-readiness.json", []),
        "answers": sum(model[key]["answers"] for model in models for key in ("single", "tools", "transfer") if key in model),
        "rubric": RUBRIC_VERSION,
    }


def load_run_page(run_id):
    runs, final = load_training()
    run = next((run for run in runs if run["id"] == run_id), None)
    if run is None:
        raise ValueError("Unknown run")
    run["color"] = run["color"] if run["current"] else "#2563eb"
    charts = curves([run])
    samples = read_lines(run["path"] / "samples.jsonl")
    shown = []
    for step in sorted({row["step"] for row in samples}):
        if step in (1, run["done"]) or (step == run["done"] // 2 and run["done"] > 4):
            groups = {}
            for row in samples:
                if row["step"] == step:
                    groups.setdefault(row["task_id"], []).append(row)
            shown.append({
                "step": step,
                "groups": [
                    {
                        "task": task,
                        "tier": rows[0]["tier"],
                        "reason": rows[0].get("reason"),
                        "answers": [
                            {
                                "reward": row["reward"],
                                "status": row["status"],
                                "decision": row.get("decision", ""),
                                "text": row["text"],
                            }
                            for row in rows
                        ],
                    }
                    for task, rows in groups.items()
                ],
            })
    config = run["config"]
    facts = [
        ("KL β", f"{run['beta']:g}"),
        ("Rubric", run["rubric"]),
        ("Steps", f"{run['done']} / {run['planned']}"),
        ("LR", config.get("lr")),
        ("Group size", f"{config.get('group', 4)}-{config.get('max_group', 4)}"),
        ("Epochs", config.get("epochs", 1)),
        ("Prompt", config.get("prompt", "standard")),
        ("GPU time", f"{run['hours']:.1f}h"),
    ]
    return {"run": run, "runs": runs, "charts": charts, "steps": shown, "facts": facts, "rubric": RUBRIC_VERSION}


def pretty(text):
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        return text
    if isinstance(value, dict) and value:
        return "{\n" + ",\n".join(f"  {json.dumps(key)}: {json.dumps(item)}" for key, item in value.items()) + "\n}"
    if isinstance(value, list) and value:
        return "[\n" + ",\n".join(f"  {json.dumps(item)}" for item in value) + "\n]"
    return text


def describe_call(call):
    if isinstance(call, str):
        try:
            call = json.loads(call)
        except ValueError:
            return call
    function = call.get("function") or call
    try:
        arguments = json.loads(function.get("arguments") or "{}")
        arguments = ", ".join(f"{key}={json.dumps(value)}" for key, value in arguments.items())
    except (TypeError, ValueError, AttributeError):
        arguments = function.get("arguments")
    return f"{function.get('name')}({arguments})"


def show(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def field_state(score, field):
    if not score or not score["diagnostics"]["valid_format"]:
        return ""
    if field == "decision":
        return "ok" if score["diagnostics"]["correct_decision"] else "bad"
    if field in score["fields"]:
        return "ok" if score["fields"][field]["correct"] else "bad"
    if field == "clauses" and score["status"] == "scored":
        return "ok" if score["breakdown"]["citations"]["score"] == 1 else "bad"
    return ""


def load_transcript(run, task_id):
    directory = V4 / run[3:] if run.startswith("v4-") else RESULTS / run
    parent = V4 if run.startswith("v4-") else RESULTS
    if (
        not run.startswith(("comparison-", "v4-hosted-", "v4-transfer-human-"))
        or directory.parent.resolve() != parent.resolve()
        or not directory.is_dir()
    ):
        raise ValueError("Unknown run")
    config = read_json(directory / "config.json", {})
    rows = sorted(
        (row for row in read_json(directory / "scores.json", []) if row["task_id"] == task_id),
        key=lambda row: row.get("rollout", 0),
    )
    if not rows:
        raise ValueError("Unknown task")
    outputs = {}
    with (directory / "transcript.jsonl").open() as handle:
        for line in handle:
            if task_id in line and '"output"' in line:
                event = json.loads(line)
                if event.get("event") == "output" and event.get("output"):
                    outputs[event["id"]] = event["output"]
    task = load_tasks().get(task_id)
    attempts = []
    for row in rows:
        output = outputs.get(row["id"]) or {}
        messages = output.get("completion") or []
        answer = None
        if messages and messages[-1].get("role") == "assistant":
            try:
                answer = json.loads(messages[-1].get("content"))
            except (TypeError, ValueError):
                answer = None
        score = row.get("score")
        breakdown = []
        if score:
            for name, part in score["breakdown"].items():
                breakdown.append({
                    "name": name.replace("_", " "),
                    "score": part.get("score"),
                    "clauses": " ".join(part.get("clauses", [])),
                })
        steps = [
            {
                "role": {"assistant": "Assistant", "tool": "Tool", "user": "Prompt"}.get(
                    message.get("role"), message.get("role")
                ),
                "text": pretty(message.get("content") or ""),
                "calls": [describe_call(call) for call in message.get("tool_calls") or []],
                "reasoning": message.get("reasoning_content") or "",
            }
            for message in messages
        ]
        attempts.append({
            "number": row.get("rollout", 0) + 1,
            "score": score,
            "reward": score["score"] if score else None,
            "failure": failure(score, row["tier"]) or "passed",
            "latency": row.get("latency_seconds"),
            "cost": row.get("cost_usd"),
            "answer": answer if isinstance(answer, dict) else None,
            "steps": steps,
            "breakdown": breakdown,
            "error": row.get("error"),
            "calls": sum(len(step["calls"]) for step in steps),
        })
    expected = task["truth"] if task else None
    fields = [
        field
        for field in ANSWER_FIELDS
        if (expected and field in expected) or any(attempt["answer"] and field in attempt["answer"] for attempt in attempts)
    ]
    comparison = [
        {
            "field": field.replace("_", " "),
            "expected": show(expected.get(field)) if expected else "",
            "cells": [
                {
                    "value": show(attempt["answer"].get(field)) if attempt["answer"] else "(malformed)",
                    "state": field_state(attempt["score"], field),
                }
                for attempt in attempts
            ],
        }
        for field in fields
    ]
    prompt = ""
    for output in outputs.values():
        users = [message for message in output.get("prompt") or [] if message.get("role") == "user"]
        if users:
            prompt = show(users[-1].get("content"))
            break
    name, short, color = model_info(config.get("model", run))
    rewards = [attempt["reward"] for attempt in attempts if attempt["reward"] is not None]
    tier = rows[0].get("tier")
    return {
        "run": run,
        "task_id": task_id,
        "number": task["number"] if task else None,
        "kind": TIERS.get(tier, ""),
        "detail": task["detail"] if task else "",
        "model_name": name,
        "color": color,
        "default_sampling": config.get("temperature", 0) is None,
        "mode": "Transfer" if run.startswith("v4-transfer-") else "Tools + calculator" if config.get("mode") == "tools" else "Single-turn",
        "mean": statistics.mean(rewards) if rewards else None,
        "comparison": comparison,
        "expected": expected is not None,
        "attempts": attempts,
        "prompt": prompt,
        "rubric": RUBRIC_VERSION,
    }


def render_markdown(paths):
    from markdown_it import MarkdownIt
    from mdit_py_plugins.dollarmath import dollarmath_plugin

    renderer = MarkdownIt("commonmark").enable("table").use(dollarmath_plugin, allow_space=False, allow_digits=False)
    sections, contents = [], []
    for path in paths:
        html = renderer.render(Path(path).read_text())

        def anchor(match):
            title = re.sub(r"<.*?>", "", match[2])
            slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
            if match[1] == "2" or (match[1] == "1" and len(paths) > 1):
                contents.append({"id": slug, "title": title})
            return f'<h{match[1]} id="{slug}">{match[2]}</h{match[1]}>'

        html = re.sub(r"<h([123])>(.*?)</h\1>", anchor, html)
        sections.append(html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>"))
    return {"content": "\n".join(sections), "contents": contents, "rubric": RUBRIC_VERSION}


def load_report():
    return render_markdown([ROOT / "reports/writeup.md"])


def load_rules():
    return render_markdown([ROOT / "rules/payroll.md", ROOT / "rules/scoring.md", ROOT / "rules/judge.md"])
