import os
import glob
from typing import Optional, List, Union, Any

from .reader import read_df
from .timeline import (
    plot_kernel_timeline,
    plot_scope_timeline,
    plot_memory_timeline,
    plot_utilization_timeline,
    plot_host_timeline,
    _explode_device_samples,
    _explode_host_samples
)

# --- Global State ---
_GLOBAL_DF = None

def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as e:
        raise ImportError("viz requires matplotlib. Install with: pip install gpufl[viz]") from e

def _check_init():
    if _GLOBAL_DF is None:
        raise RuntimeError("Global data not loaded. Call viz.init('path/to/logs') first.")

def init(log_pattern: Union[str, List[str]]):
    """
    Load log files into memory for fast interactive visualization.

    Args:
        log_pattern: File path, directory, or glob pattern (e.g., "logs/*.log").
                     If directory, defaults to "logs/**/*.log" (recursive) or similar.
    """
    global _GLOBAL_DF

    # Handle directory vs glob
    if isinstance(log_pattern, str) and os.path.isdir(log_pattern):
        # Look for all .log files in that directory
        pattern = os.path.join(log_pattern, "*.log")
    else:
        pattern = log_pattern

    print(f"Loading logs from: {pattern} ...")
    df = read_df(pattern)

    if len(df) == 0:
        print("[Warn] No events found. Check your path pattern.")
        _GLOBAL_DF = None
    else:
        # Pre-convert timestamps to numeric to save time later
        import pandas as pd
        for col in ["ts_ns", "start_ns", "end_ns"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        _GLOBAL_DF = df

        # Print summary
        apps = df["app"].unique() if "app" in df.columns else []
        print(f"Loaded {len(df)} events.")
        print(f"Apps found: {list(apps)}")

def get_data() -> Any:
    """Return the raw global DataFrame."""
    _check_init()
    return _GLOBAL_DF

def _filter_df(df, **filters):
    """Apply generic k=v filtering to the dataframe."""
    d = df
    for k, v in filters.items():
        if k not in d.columns:
            print(f"[Warn] Column '{k}' not found in logs; ignoring filter {k}={v}.")
            continue

        # Support filtering by list (isin) or single value (==)
        if isinstance(v, (list, tuple)):
            d = d[d[k].isin(v)]
        else:
            d = d[d[k] == v]
    return d

def show(**filters):
    """
    Filter the loaded data and show timelines.

    Usage:
        viz.show(app="MyApp")
        viz.show(tag="loading_phase")
        viz.show(pid=1234)
    """
    _check_init()
    plt = _require_matplotlib()

    # 1. Filter Data
    df = _filter_df(_GLOBAL_DF, **filters)

    if len(df) == 0:
        print(f"No events match filters: {filters}")
        return

    # Generate a title based on filters
    title_suffix = ", ".join([f"{k}={v}" for k, v in filters.items()])

    figs = []

    # 2. Host Metrics (CPU/RAM)
    # Check if we have host data in this slice
    if "host" in df.columns and not df[df["host"].notna()].empty:
        f = plot_host_timeline(df, title=f"Host Metrics ({title_suffix})")
        if f: figs.append(f)

    # 3. GPU Metrics
    # Only try plotting if we have sample events
    if not df[df["event_type"].str.contains("sample", na=False)].empty:
        f = plot_utilization_timeline(df, title=f"GPU Utilization ({title_suffix})")
        if f: figs.append(f)

        f = plot_memory_timeline(df, title=f"GPU Memory ({title_suffix})")
        if f: figs.append(f)

    # 4. Kernels
    if not df[df["event_type"] == "kernel"].empty:
        f = plot_kernel_timeline(df, title=f"Kernels ({title_suffix})")
        if f: figs.append(f)

    # 5. Scopes
    # (Requires simpler scope logic or robust pairing)
    try:
        f = plot_scope_timeline(df, title=f"Scopes ({title_suffix})")
        if f: figs.append(f)
    except Exception:
        # Scope plotting can fail if begin/end are disjoint in the filtered slice
        pass

    if not figs:
        print("Data found, but no charts could be generated (missing specific event types).")

    plt.show()

def compare(group_by="app", metric="cpu", **filters):
    """
    Compare a specific metric across different groups (e.g. apps).

    Args:
        group_by: Column to group by (default: "app").
                  Example: compare(group_by="tag", metric="cpu", app="MyApp")
        metric: "cpu", "gpu", "ram"
        **filters: Additional filters to narrow down the dataset first.
    """
    _check_init()
    plt = _require_matplotlib()

    # 1. Filter first (e.g., only look at 'phase_1')
    df = _filter_df(_GLOBAL_DF, **filters)
    if len(df) == 0:
        print("No data matches filters.")
        return

    if group_by not in df.columns:
        print(f"Cannot group by '{group_by}': column not found.")
        return

    groups = df[group_by].unique()
    if len(groups) == 0:
        print("No groups found.")
        return

    fig = plt.figure(figsize=(10, 5))
    has_plot = False

    # 2. Plot lines for each group
    for g in groups:
        # Slice for this group
        sub_df = df[df[group_by] == g]
        label = str(g)

        if metric == "cpu":
            s = _explode_host_samples(sub_df)
            if not s.empty:
                plt.plot(s["t_s"], s["cpu_pct"], label=label)
                has_plot = True

        elif metric == "ram":
            s = _explode_host_samples(sub_df)
            if not s.empty:
                plt.plot(s["t_s"], s["ram_used_mib"], label=label)
                has_plot = True

        elif metric == "gpu":
            s = _explode_device_samples(sub_df, sample_types=("scope_sample", "system_sample"))
            if not s.empty:
                # Naive: just take GPU 0 for comparison
                g0 = s[s["gpu_id"] == 0]
                if not g0.empty:
                    plt.plot(g0["t_s"], g0["util_gpu"], label=label)
                    has_plot = True

    if has_plot:
        plt.title(f"Comparison: {metric.upper()} by {group_by}")
        plt.xlabel("Time (s) [Relative start of each group]")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    else:
        print(f"No data found for metric '{metric}' in the selected groups.")
        plt.close(fig)