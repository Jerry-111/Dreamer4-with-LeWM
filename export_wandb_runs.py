#!/usr/bin/env python3
"""Export Dreamer4 W&B run histories to analysis-friendly CSV files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


DEFAULT_PROJECTS = ("dreamer4-tokenizer", "dreamer4-dynamics")
DEFAULT_METRICS = (
    "loss/total",
    "loss/mse",
    "loss/lpips",
    "loss/flow_mse",
    "loss/bootstrap_mse",
    "loss/loss_emp",
    "loss/loss_self",
    "stats/psnr",
    "stats/keep_prob",
    "stats/masked_frac",
    "stats/action_shuffle_loss_ratio",
    "stats/sigma_mean",
    "stats/B_self",
    "eval/mse_pred",
    "eval/mse_floor",
    "eval/mse_ratio_pred_over_floor",
    "eval/psnr_pred",
    "eval/psnr_floor",
    "eval/psnr_gain_over_floor_db",
    "eval/mse_pred_t1",
    "eval/mse_pred_tmid",
    "eval/mse_pred_tend",
    "eval/mse_floor_t1",
    "eval/mse_floor_tmid",
    "eval/mse_floor_tend",
    "debug/z_std",
    "amp/scale",
    "lr",
    "time/hrs",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Export W&B histories, summaries, configs, and run metadata for "
            "Dreamer4 tokenizer/dynamics evaluation."
        )
    )
    p.add_argument("--entity", default="jerrychsh-ucsd", help="W&B entity/team.")
    p.add_argument(
        "--projects",
        nargs="+",
        default=list(DEFAULT_PROJECTS),
        help="W&B projects to search.",
    )
    p.add_argument(
        "--run-name-regex",
        default="pusht",
        help="Regex matched against run display names. Use '' to disable.",
    )
    p.add_argument(
        "--run-id",
        action="append",
        default=[],
        help=(
            "Specific W&B run id/path to export. Repeat as needed. "
            "Bare ids are searched in every selected project; full paths look "
            "like entity/project/run_id."
        ),
    )
    p.add_argument(
        "--state",
        action="append",
        default=[],
        help="Optional run state filter, e.g. finished, running, crashed.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="History keys to export. Use --all-history to export every key.",
    )
    p.add_argument(
        "--all-history",
        action="store_true",
        help="Export all history keys instead of the default Dreamer4 metrics.",
    )
    p.add_argument(
        "--include-non-scalar-history",
        action="store_true",
        help="Keep media/list/dict history values as JSON strings.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=10_000,
        help="W&B scan_history page size.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("exports/wandb_pusht"),
        help="Directory where CSV files are written.",
    )
    return p.parse_args()


def is_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            return math.isfinite(value)
        return True
    return False


def csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if is_scalar(value):
        return value
    return json.dumps(value, sort_keys=True, default=str)


def flatten(prefix: str, value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, subvalue in value.items():
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(next_prefix, subvalue))
        return out
    return {prefix: csv_value(value)}


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], preferred: Iterable[str] = ()) -> int:
    materialized = list(rows)
    keys: list[str] = []
    seen: set[str] = set()
    for key in preferred:
        if key not in seen:
            keys.append(key)
            seen.add(key)
    for row in materialized:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in materialized:
            writer.writerow({key: csv_value(row.get(key)) for key in keys})
    return len(materialized)


def safe_name(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("._") or "run"


def wandb_filters(args: argparse.Namespace) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if args.run_name_regex:
        filters["display_name"] = {"$regex": args.run_name_regex}
    if args.state:
        filters["state"] = {"$in": args.state} if len(args.state) > 1 else args.state[0]
    return filters


def iter_project_runs(api: Any, args: argparse.Namespace) -> list[Any]:
    runs: list[Any] = []
    seen: set[str] = set()

    if args.run_id:
        for run_id in args.run_id:
            candidates = [run_id] if run_id.count("/") == 2 else [
                f"{args.entity}/{project}/{run_id}" for project in args.projects
            ]
            for path in candidates:
                try:
                    run = api.run(path)
                except Exception:
                    continue
                if run.path[-1] not in seen:
                    runs.append(run)
                    seen.add(run.path[-1])
        return runs

    filters = wandb_filters(args)
    for project in args.projects:
        path = f"{args.entity}/{project}"
        for run in api.runs(path, filters=filters, per_page=200):
            if run.path[-1] not in seen:
                runs.append(run)
                seen.add(run.path[-1])
    return runs


def run_base_row(run: Any) -> dict[str, Any]:
    return {
        "entity": run.entity,
        "project": run.project,
        "run_id": run.id,
        "run_name": run.name,
        "run_path": "/".join(run.path),
        "state": run.state,
        "created_at": getattr(run, "created_at", None),
        "updated_at": getattr(run, "updated_at", None),
        "url": run.url,
    }


def history_rows(run: Any, args: argparse.Namespace) -> list[dict[str, Any]]:
    keep_keys = None if args.all_history else set(args.metrics)
    rows: list[dict[str, Any]] = []
    for row in run.scan_history(page_size=args.page_size):
        base = run_base_row(run)
        kept_metric = False
        for key, value in row.items():
            if key.startswith("_") and key != "_step":
                continue
            if keep_keys is not None and key != "_step" and key not in keep_keys:
                continue
            if not args.include_non_scalar_history and not is_scalar(value):
                continue
            base[key] = csv_value(value)
            kept_metric = kept_metric or key != "_step"
        if not kept_metric and keep_keys is not None:
            continue
        rows.append(base)
    return rows


def long_rows(wide_rows: Iterable[Mapping[str, Any]]) -> Iterable[dict[str, Any]]:
    metadata = {
        "entity",
        "project",
        "run_id",
        "run_name",
        "run_path",
        "state",
        "created_at",
        "updated_at",
        "url",
        "_step",
    }
    for row in wide_rows:
        for key, value in row.items():
            if key in metadata or value in ("", None):
                continue
            yield {
                "project": row.get("project"),
                "run_id": row.get("run_id"),
                "run_name": row.get("run_name"),
                "step": row.get("_step"),
                "metric": key,
                "value": value,
            }


def latest_metric_tables(
    wide_rows: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata = {
        "entity",
        "project",
        "run_id",
        "run_name",
        "run_path",
        "state",
        "created_at",
        "updated_at",
        "url",
        "_step",
    }
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    base_by_run: dict[str, dict[str, Any]] = {}

    for row in wide_rows:
        run_id = str(row.get("run_id", ""))
        if not run_id:
            continue
        base_by_run[run_id] = {key: row.get(key) for key in metadata if key != "_step"}
        try:
            step = int(float(row.get("_step", 0)))
        except (TypeError, ValueError):
            step = 0

        for key, value in row.items():
            if key in metadata or value in ("", None):
                continue
            slot = (run_id, key)
            previous = latest.get(slot)
            if previous is None or step >= int(previous["step"]):
                latest[slot] = {
                    "project": row.get("project"),
                    "run_id": run_id,
                    "run_name": row.get("run_name"),
                    "step": step,
                    "metric": key,
                    "value": value,
                }

    latest_long = sorted(latest.values(), key=lambda item: (item["project"], item["run_name"], item["metric"]))

    latest_wide: list[dict[str, Any]] = []
    for run_id, base in sorted(base_by_run.items(), key=lambda item: (item[1].get("project"), item[1].get("run_name"))):
        row = dict(base)
        for (metric_run_id, metric), item in latest.items():
            if metric_run_id != run_id:
                continue
            clean_metric = metric.replace("/", ".")
            row[f"latest.{clean_metric}"] = item["value"]
            row[f"latest_step.{clean_metric}"] = item["step"]
        latest_wide.append(row)

    return latest_long, latest_wide


def main() -> int:
    args = parse_args()

    try:
        import wandb
    except ImportError:
        print(
            "wandb is not installed in this Python environment. Try:\n"
            "  conda run -n dreamer4 python export_wandb_runs.py ...",
            file=sys.stderr,
        )
        return 2

    api = wandb.Api()
    runs = iter_project_runs(api, args)
    if not runs:
        print("No W&B runs matched the requested filters.", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_history: list[dict[str, Any]] = []
    index_rows: list[dict[str, Any]] = []
    config_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for run in runs:
        print(f"Exporting {run.project}/{run.name} ({run.id})")
        base = run_base_row(run)
        index_rows.append(base)

        config_rows.append({**base, **flatten("config", dict(run.config))})
        summary_rows.append({**base, **flatten("summary", dict(run.summary))})

        run_history = history_rows(run, args)
        all_history.extend(run_history)

        run_csv = args.outdir / "runs" / f"{safe_name(run.project)}__{safe_name(run.name)}__{run.id}.csv"
        write_csv(run_csv, run_history, preferred=[*base.keys(), "_step", *args.metrics])

    preferred_history = [
        "entity",
        "project",
        "run_id",
        "run_name",
        "run_path",
        "state",
        "created_at",
        "updated_at",
        "url",
        "_step",
        *args.metrics,
    ]
    n_history = write_csv(args.outdir / "history_wide.csv", all_history, preferred=preferred_history)
    n_long = write_csv(
        args.outdir / "history_long.csv",
        long_rows(all_history),
        preferred=["project", "run_id", "run_name", "step", "metric", "value"],
    )
    latest_long, latest_wide = latest_metric_tables(all_history)
    write_csv(
        args.outdir / "metrics_latest.csv",
        latest_long,
        preferred=["project", "run_id", "run_name", "step", "metric", "value"],
    )
    write_csv(
        args.outdir / "metrics_latest_wide.csv",
        latest_wide,
        preferred=[
            "entity",
            "project",
            "run_id",
            "run_name",
            "run_path",
            "state",
            "created_at",
            "updated_at",
            "url",
        ],
    )
    write_csv(args.outdir / "run_index.csv", index_rows)
    write_csv(args.outdir / "run_config.csv", config_rows)
    write_csv(args.outdir / "run_summary.csv", summary_rows)

    print(f"Wrote {len(runs)} runs, {n_history} wide rows, {n_long} long rows to {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
