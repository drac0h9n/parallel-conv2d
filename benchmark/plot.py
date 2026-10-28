#!/usr/bin/env python3
"""Draw the charts and summary tables from the benchmark CSVs.

Usage (from the repository root):

    python3 benchmark/plot.py                         # reads and writes results/
    python3 benchmark/plot.py --results results --size 4096 --kernels 3,5 --formats png,pdf

Inputs   <results>/samples.csv        every timed call (benchmark/run_experiments.py)
         <results>/end_to_end.csv     optional, whole-process timings
         <results>/environment.json   optional, hardware record (P/E-core split)
         <results>/run_info.json      optional, run parameters
Outputs  <results>/strong_scaling.png, speedup_vs_baseline.png, throughput_vs_size.png,
         efficiency.png, kernel_size.png, gflops.png, speedup_vs_size.png,
         end_to_end.png (300 dpi; add --formats png,pdf for vector copies)
         <results>/summary.md         the tables behind every chart

Statistics. Every plotted point is the median of the timed runs of one
configuration; error bars span the interquartile range (25th-75th percentile).
Throughput is images processed per second, 1000 / median time per image in ms;
its error bars convert the 75th and 25th percentile times the same way.
Strong-scaling speedup is self-relative, S(p) = T_impl(1) / T_impl(p), so the
ideal line is S = p. Figure 1b and the tables also give the speedup over the
sequential baseline, T_serial / T_impl(p). Parallel efficiency is
E(p) = S(p) / p. Compute throughput counts one multiply and one add per kernel
tap: FLOP = 2 K^2 (N-K+1)^2 per image.

Requires matplotlib (python3 -m pip install matplotlib).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter, PercentFormatter  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
Key = Tuple[str, int, int, int]  # (implementation, image size N, kernel K, threads p)

# ---------------------------------------------------------------------------
# Visual system (light, print-friendly). The two parallel implementations use
# the first two slots of a colour-vision-deficiency-validated categorical
# palette; the sequential baseline is neutral ink. Every series also has its
# own marker, so identity survives greyscale printing.
# ---------------------------------------------------------------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SHADE = "#f4f3ef"
SERIES = {
    "serial": {"color": "#52514e", "marker": "D", "size": 4.6, "label": "Sequential baseline"},
    "basic": {"color": "#eb6834", "marker": "s", "size": 5.2, "label": "Basic OpenMP"},
    "optimized": {"color": "#2a78d6", "marker": "o", "size": 5.6, "label": "Optimized OpenMP"},
}
OVERHEAD_COLORS = ("#898781", "#d3d2cb")  # first-call overhead, process + input set-up
IDEAL_STYLE = {"color": MUTED, "linewidth": 1.0, "linestyle": (0, (4, 3))}
LABEL_PRIORITY = ("optimized", "serial", "basic")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "normal",
            "axes.titlecolor": INK,
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.labelcolor": INK_2,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "xtick.color": AXIS,
            "ytick.color": AXIS,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "text.color": INK,
            "pdf.fonttype": 42,  # keep text editable in the PDF
            "svg.fonttype": "none",
        }
    )


def style_axes(ax, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=3, width=0.8)
    ax.grid(True, axis=grid_axis, which="major", color=GRID, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class Stat:
    def __init__(self, values: Sequence[float]) -> None:
        ordered = sorted(values)
        self.n = len(ordered)
        self.median = statistics.median(ordered)
        if self.n >= 2:
            q1, _, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
        else:
            q1 = q3 = self.median
        self.q1, self.q3 = q1, q3
        self.mean = statistics.fmean(ordered)
        self.std = statistics.stdev(ordered) if self.n >= 2 else 0.0
        self.minimum, self.maximum = ordered[0], ordered[-1]


class Results:
    def __init__(self, samples: Dict[Key, List[float]]) -> None:
        if not samples:
            raise SystemExit("error: no timing samples found")
        self.stats: Dict[Key, Stat] = {key: Stat(values) for key, values in samples.items()}

    @classmethod
    def load(cls, path: Path) -> "Results":
        samples: Dict[Key, List[float]] = defaultdict(list)
        with path.open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (
                    row["implementation"],
                    int(row["image_size"]),
                    int(row["kernel"]),
                    int(row["threads"]),
                )
                samples[key].append(float(row["runtime_ms"]))
        return cls(samples)

    def sizes(self) -> List[int]:
        return sorted({key[1] for key in self.stats})

    def kernels(self) -> List[int]:
        return sorted({key[2] for key in self.stats})

    def threads(self, implementation: str, size: int, kernel: int) -> List[int]:
        return sorted(
            key[3]
            for key in self.stats
            if key[0] == implementation and key[1] == size and key[2] == kernel
        )

    def has(self, key: Key) -> bool:
        return key in self.stats

    def stat(self, key: Key) -> Stat:
        if key not in self.stats:
            raise KeyError(f"no measurements for {key}")
        return self.stats[key]

    def median(self, key: Key) -> float:
        return self.stat(key).median

    def speedup(self, key: Key, baseline: str = "self") -> Tuple[float, float, float]:
        """Speedup (median) and its IQR band.

        baseline="self": T_impl(1) / T_impl(p); baseline="serial": T_serial / T_impl(p).
        The band divides the baseline median by the 75th/25th-percentile time.
        """
        implementation, size, kernel, _ = key
        if baseline == "serial":
            base = self.median(("serial", size, kernel, 1))
        else:
            base = self.median((implementation, size, kernel, 1))
        stat = self.stat(key)
        return base / stat.median, base / stat.q3, base / stat.q1

    @staticmethod
    def flops(size: int, kernel: int) -> float:
        out = size - kernel + 1
        return 2.0 * kernel * kernel * out * out

    def gflops(self, key: Key) -> float:
        return self.flops(key[1], key[2]) / (self.median(key) * 1e6)

    def throughput(self, key: Key) -> Tuple[float, float, float]:
        """Images per second (1000 / median ms) and its IQR band (from the 75th/25th-percentile time)."""
        stat = self.stat(key)
        return 1000.0 / stat.median, 1000.0 / stat.q3, 1000.0 / stat.q1


def load_end_to_end(path: Path) -> Dict[Key, List[Tuple[float, float]]]:
    rows: Dict[Key, List[Tuple[float, float]]] = defaultdict(list)
    if not path.is_file():
        return rows
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = (
                row["implementation"],
                int(row["image_size"]),
                int(row["kernel"]),
                int(row["threads"]),
            )
            rows[key].append((float(row["end_to_end_ms"]), float(row["cold_kernel_ms"])))
    return rows


def core_split(environment: dict) -> Tuple[Optional[int], Optional[int]]:
    sysctl = (environment.get("macos") or {}).get("sysctl") or {}
    try:
        performance = int(sysctl["hw.perflevel0.physicalcpu"])
        efficiency = int(sysctl.get("hw.perflevel1.physicalcpu", 0))
    except (KeyError, ValueError):
        return None, None
    return performance, efficiency


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------


def series_label(implementation: str, threads: Optional[int] = None) -> str:
    label = SERIES[implementation]["label"]
    if threads is not None and implementation != "serial":
        label += f", p = {threads}"
    return label


def series_handle(implementation: str, label: Optional[str] = None) -> Line2D:
    spec = SERIES[implementation]
    return Line2D(
        [0], [0],
        color=spec["color"], linewidth=1.8, marker=spec["marker"], markersize=spec["size"],
        markerfacecolor=spec["color"], markeredgecolor=SURFACE, markeredgewidth=1.0,
        label=label or spec["label"],
    )


def ideal_handle(label: str) -> Line2D:
    return Line2D([0], [0], label=label, **IDEAL_STYLE)


def shared_legend(fig, handles: Sequence, columns: Optional[int] = None) -> None:
    fig.legend(
        handles=list(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=columns or len(handles),
        frameon=False,
        handlelength=2.4,
        columnspacing=1.6,
    )


def plot_series(ax, xs, ys, lows, highs, implementation: str) -> None:
    spec = SERIES[implementation]
    yerr = [
        [max(0.0, y - low) for y, low in zip(ys, lows)],
        [max(0.0, high - y) for y, high in zip(ys, highs)],
    ]
    ax.errorbar(
        xs, ys, yerr=yerr,
        color=spec["color"], ecolor=spec["color"], linewidth=1.8,
        marker=spec["marker"], markersize=spec["size"], markerfacecolor=spec["color"],
        markeredgecolor=SURFACE, markeredgewidth=1.0,
        elinewidth=0.9, capsize=2.2, capthick=0.9, zorder=3,
        solid_capstyle="round", solid_joinstyle="round",
    )


def end_labels(ax, points: Sequence[Tuple[str, float, float, str]], min_gap_pt: float = 10.0) -> None:
    """Direct-label series ends in priority order; skip a label rather than collide."""
    order = {name: index for index, name in enumerate(LABEL_PRIORITY)}
    placed: List[float] = []
    for implementation, x, y, text in sorted(points, key=lambda item: order.get(item[0], 99)):
        _, y_display = ax.transData.transform((x, y))
        y_pt = y_display * 72.0 / ax.figure.dpi
        if any(abs(y_pt - other) < min_gap_pt for other in placed):
            continue
        placed.append(y_pt)
        ax.annotate(text, (x, y), xytext=(6, 0), textcoords="offset points", va="center", ha="left",
                    fontsize=8, color=INK, zorder=4)


def mark_core_types(ax, performance: Optional[int], efficiency: Optional[int], p_max: int) -> None:
    """Shade thread counts that must spill onto efficiency cores (Apple silicon)."""
    if not performance or not efficiency or p_max <= performance:
        return
    ax.axvspan(performance, p_max + 0.6, color=SHADE, zorder=0, linewidth=0)
    ax.axvline(performance, color=AXIS, linewidth=0.8, zorder=1)
    ax.text(performance + 0.3, 0.03, f"p > {performance}: threads\nalso on E-cores",
            transform=ax.get_xaxis_transform(), ha="left", va="bottom", fontsize=7.5,
            color=MUTED, linespacing=1.15)


def rate_text(value: float) -> str:
    """Images per second with about three significant digits."""
    if value >= 100:
        return f"{value:,.0f}"
    if value >= 10:
        return f"{value:.1f}"
    if value >= 1:
        return f"{value:.2f}"
    return f"{value:.3f}"


def plain_number(value: float, _position=None) -> str:
    if value <= 0:
        return ""
    if value >= 1:
        return f"{value:,.0f}"
    return f"{value:g}"


def log_y(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_number))
    ax.yaxis.set_minor_formatter(NullFormatter())


def log2_sizes(ax, sizes: Sequence[int]) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(sizes))
    ax.set_xticklabels([str(n) for n in sizes])
    ax.xaxis.set_minor_formatter(NullFormatter())


def suptitle(fig, text: str) -> None:
    fig.suptitle(text, x=0.0, ha="left", fontsize=10, fontweight="bold")


def save(fig, directory: Path, stem: str, formats: Sequence[str]) -> List[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for extension in formats:
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, dpi=300 if extension == "png" else None, bbox_inches="tight", pad_inches=0.05)
        written.append(path)
    plt.close(fig)
    return written


def panel_axes(count: int, width_per_panel: float = 3.45, height: float = 2.9, sharey: bool = True):
    fig, axes = plt.subplots(1, count, figsize=(width_per_panel * count, height), sharey=sharey,
                             constrained_layout=True, squeeze=False)
    return fig, list(axes[0])


def letter(index: int) -> str:
    return "(" + "abcdefghijklmnopqrstuvwxyz"[index] + ")"


def kernel_title(index: int, kernel: int) -> str:
    return f"{letter(index)} {kernel}×{kernel} kernel"


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def figure_speedup(results: Results, size: int, kernels: Sequence[int], cores, out: Path, formats,
                   baseline: str):
    fig, axes = panel_axes(len(kernels))
    top = 0.0
    for index, (ax, kernel) in enumerate(zip(axes, kernels)):
        style_axes(ax)
        threads = results.threads("optimized", size, kernel)
        p_max = max(threads)
        mark_core_types(ax, cores[0], cores[1], p_max)
        ax.plot([1, p_max], [1, p_max], zorder=2, **IDEAL_STYLE)
        labels = []
        for implementation in ("basic", "optimized"):
            values = [results.speedup((implementation, size, kernel, p), baseline) for p in threads]
            ys = [value[0] for value in values]
            top = max(top, max(value[2] for value in values), p_max)
            plot_series(ax, threads, ys, [v[1] for v in values], [v[2] for v in values], implementation)
            labels.append((implementation, threads[-1], ys[-1], f"{ys[-1]:.1f}×"))
        end_labels(ax, labels)
        ax.set_xticks(threads)
        ax.set_xlim(0.4, p_max + 1.5)
        ax.set_title(kernel_title(index, kernel))
        ax.set_xlabel("OpenMP threads, p")
    for ax in axes:
        ax.set_ylim(0, math.ceil(top + 0.6))
    if baseline == "serial":
        axes[0].set_ylabel("Speedup over baseline  T$_{serial}$ / T(p)")
        ideal = ideal_handle("Linear, S = p")
        title = f"Speedup over the sequential baseline at {size}×{size}"
        stem = "speedup_vs_baseline"
    else:
        axes[0].set_ylabel("Speedup  S(p) = T(1) / T(p)")
        ideal = ideal_handle("Ideal linear speedup, S = p")
        title = f"Strong scaling at {size}×{size}: speedup vs. threads"
        stem = "strong_scaling"
    shared_legend(fig, [series_handle("basic"), series_handle("optimized"), ideal])
    suptitle(fig, title)
    return save(fig, out, stem, formats)


def figure_throughput_vs_size(results: Results, kernels: Sequence[int], out: Path, formats):
    fig, axes = panel_axes(len(kernels))
    sizes = results.sizes()
    p_max = 1
    for index, (ax, kernel) in enumerate(zip(axes, kernels)):
        style_axes(ax)
        p_max = max(results.threads("optimized", sizes[-1], kernel))
        labels = []
        for implementation, threads in (("serial", 1), ("basic", p_max), ("optimized", p_max)):
            bands = [results.throughput((implementation, n, kernel, threads)) for n in sizes]
            ys = [band[0] for band in bands]
            plot_series(ax, sizes, ys, [band[1] for band in bands], [band[2] for band in bands], implementation)
            labels.append((implementation, sizes[-1], ys[-1], f"{rate_text(ys[-1])} img/s"))
        log2_sizes(ax, sizes)
        log_y(ax)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 3.2)
        end_labels(ax, labels)
        ax.set_title(kernel_title(index, kernel))
        ax.set_xlabel("Image size N (N×N input)")
    axes[0].set_ylabel("Throughput (images/s, log scale)")
    shared_legend(fig, [series_handle("serial"), series_handle("basic", series_label("basic", p_max)),
                        series_handle("optimized", series_label("optimized", p_max))])
    suptitle(fig, "Throughput vs. problem size")
    return save(fig, out, "throughput_vs_size", formats)


def figure_efficiency(results: Results, size: int, kernels: Sequence[int], cores, out: Path, formats):
    fig, axes = panel_axes(len(kernels))
    top = 105.0
    for index, (ax, kernel) in enumerate(zip(axes, kernels)):
        style_axes(ax)
        threads = results.threads("optimized", size, kernel)
        p_max = max(threads)
        mark_core_types(ax, cores[0], cores[1], p_max)
        ax.axhline(100, zorder=2, **IDEAL_STYLE)
        labels = []
        for implementation in ("basic", "optimized"):
            values = [results.speedup((implementation, size, kernel, p)) for p in threads]
            ys = [100.0 * v[0] / p for v, p in zip(values, threads)]
            lows = [100.0 * v[1] / p for v, p in zip(values, threads)]
            highs = [100.0 * v[2] / p for v, p in zip(values, threads)]
            top = max(top, max(highs) + 5)
            plot_series(ax, threads, ys, lows, highs, implementation)
            labels.append((implementation, threads[-1], ys[-1], f"{ys[-1]:.0f}%"))
        end_labels(ax, labels)
        ax.set_xticks(threads)
        ax.set_xlim(0.4, p_max + 1.5)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_title(kernel_title(index, kernel))
        ax.set_xlabel("OpenMP threads, p")
    for ax in axes:
        ax.set_ylim(0, top)
    axes[0].set_ylabel("Parallel efficiency  E(p) = S(p) / p")
    shared_legend(fig, [series_handle("basic"), series_handle("optimized"), ideal_handle("Ideal, E = 100%")])
    suptitle(fig, f"Parallel efficiency at {size}×{size}")
    return save(fig, out, "efficiency", formats)


def figure_kernel_size(results: Results, size: int, out: Path, formats):
    kernels = [k for k in results.kernels() if results.has(("serial", size, k, 1))]
    p_max = max(results.threads("optimized", size, kernels[0]))
    fig, axes = panel_axes(2, sharey=False)
    ticks = [f"{k}×{k}" for k in kernels]

    ax = axes[0]
    style_axes(ax)
    labels = []
    for implementation, threads in (("serial", 1), ("basic", p_max), ("optimized", p_max)):
        bands = [results.throughput((implementation, size, k, threads)) for k in kernels]
        ys = [band[0] for band in bands]
        plot_series(ax, kernels, ys, [band[1] for band in bands], [band[2] for band in bands], implementation)
        labels.append((implementation, kernels[-1], ys[-1], f"{rate_text(ys[-1])} img/s"))
    log_y(ax)
    ax.set_xticks(kernels)
    ax.set_xticklabels(ticks)
    ax.set_xlim(kernels[0] - 1, kernels[-1] + 4.2)
    end_labels(ax, labels)
    ax.set_title(f"{letter(0)} Throughput")
    ax.set_xlabel("Kernel size K")
    ax.set_ylabel("Throughput (images/s, log scale)")

    ax = axes[1]
    style_axes(ax)
    labels = []
    for implementation in ("basic", "optimized"):
        values = [results.speedup((implementation, size, k, p_max)) for k in kernels]
        ys = [v[0] for v in values]
        plot_series(ax, kernels, ys, [v[1] for v in values], [v[2] for v in values], implementation)
        labels.append((implementation, kernels[-1], ys[-1], f"{ys[-1]:.1f}×"))
    ax.axhline(p_max, zorder=2, **IDEAL_STYLE)
    end_labels(ax, labels)
    ax.set_xticks(kernels)
    ax.set_xticklabels(ticks)
    ax.set_xlim(kernels[0] - 1, kernels[-1] + 2.6)
    ax.set_ylim(0, p_max + 1)
    ax.set_title(f"{letter(1)} Speedup at p = {p_max}")
    ax.set_xlabel("Kernel size K")
    ax.set_ylabel(f"S({p_max}) = T(1) / T({p_max})")
    shared_legend(fig, [series_handle("serial"), series_handle("basic", series_label("basic", p_max)),
                        series_handle("optimized", series_label("optimized", p_max)),
                        ideal_handle(f"Ideal, S = {p_max}")])
    suptitle(fig, f"Effect of kernel size at {size}×{size}")
    return save(fig, out, "kernel_size", formats)


def figure_gflops(results: Results, size: int, out: Path, formats):
    kernels = [k for k in results.kernels() if results.has(("serial", size, k, 1))]
    columns = 2 if len(kernels) > 1 else 1
    rows = math.ceil(len(kernels) / columns)
    fig, grid = plt.subplots(rows, columns, figsize=(3.45 * columns, 2.45 * rows + 0.25), sharex=True,
                             sharey=True, constrained_layout=True, squeeze=False)
    axes = [ax for row in grid for ax in row]
    top = 0.0
    for index, kernel in enumerate(kernels):
        ax = axes[index]
        style_axes(ax)
        threads = results.threads("optimized", size, kernel)
        serial = results.gflops(("serial", size, kernel, 1))
        ax.axhline(serial, color=SERIES["serial"]["color"], linewidth=1.2, zorder=2)
        ax.annotate(f"sequential {serial:.1f}", (threads[-1], serial), xytext=(0, 3), textcoords="offset points",
                    ha="right", va="bottom", fontsize=7.5, color=INK_2)
        labels = []
        flop_millions = results.flops(size, kernel) / 1e6
        for implementation in ("basic", "optimized"):
            keys = [(implementation, size, kernel, p) for p in threads]
            ys = [results.gflops(key) for key in keys]
            lows = [flop_millions / results.stat(key).q3 for key in keys]
            highs = [flop_millions / results.stat(key).q1 for key in keys]
            top = max(top, max(highs))
            plot_series(ax, threads, ys, lows, highs, implementation)
            labels.append((implementation, threads[-1], ys[-1], f"{ys[-1]:.1f}"))
        end_labels(ax, labels)
        ax.set_xticks(threads)
        ax.set_xlim(0.4, max(threads) + 2.0)
        ax.set_title(kernel_title(index, kernel))
        if index % columns == 0:
            ax.set_ylabel("GFLOP/s")
        if index >= len(kernels) - columns:
            ax.set_xlabel("OpenMP threads, p")
    for ax in axes[: len(kernels)]:
        ax.set_ylim(0, top * 1.08)
    for ax in axes[len(kernels):]:
        ax.set_visible(False)
    serial_line = Line2D([0], [0], color=SERIES["serial"]["color"], linewidth=1.2,
                         label="Sequential baseline (p = 1)")
    shared_legend(fig, [serial_line, series_handle("basic"), series_handle("optimized")])
    suptitle(fig, f"Compute throughput at {size}×{size} (GFLOP/s = 2K²(N−K+1)² / median time)")
    return save(fig, out, "gflops", formats)


def figure_speedup_vs_size(results: Results, kernels: Sequence[int], out: Path, formats):
    sizes = results.sizes()
    fig, axes = panel_axes(len(kernels))
    p_max = 1
    for index, (ax, kernel) in enumerate(zip(axes, kernels)):
        style_axes(ax)
        p_max = max(results.threads("optimized", sizes[-1], kernel))
        ax.axhline(p_max, zorder=2, **IDEAL_STYLE)
        labels = []
        for implementation in ("basic", "optimized"):
            values = [results.speedup((implementation, n, kernel, p_max)) for n in sizes]
            ys = [v[0] for v in values]
            plot_series(ax, sizes, ys, [v[1] for v in values], [v[2] for v in values], implementation)
            labels.append((implementation, sizes[-1], ys[-1], f"{ys[-1]:.1f}×"))
        end_labels(ax, labels)
        log2_sizes(ax, sizes)
        ax.set_xlim(sizes[0] / 1.35, sizes[-1] * 2.1)
        ax.set_ylim(0, p_max + 1)
        ax.set_title(kernel_title(index, kernel))
        ax.set_xlabel("Image size N (N×N input)")
    axes[0].set_ylabel(f"S({p_max}) = T(1) / T({p_max})")
    shared_legend(fig, [series_handle("basic"), series_handle("optimized"), ideal_handle(f"Ideal, S = {p_max}")])
    suptitle(fig, f"Scalability with problem size: speedup at p = {p_max}")
    return save(fig, out, "speedup_vs_size", formats)


def end_to_end_rows(results: Results, e2e: Dict[Key, List[Tuple[float, float]]]) -> List[dict]:
    rows = []
    order = {"serial": 0, "basic": 1, "optimized": 2}
    for key in sorted(e2e, key=lambda k: (order.get(k[0], 9), k[2], k[1], k[3])):
        if not results.has(key):
            continue
        totals = [value[0] for value in e2e[key]]
        colds = [value[1] for value in e2e[key]]
        warm = results.median(key)
        total = statistics.median(totals)
        cold = statistics.median(colds)
        # The breakdown must add up to the end-to-end total: the kernel part is the
        # steady-state time, capped at the cold call measured in the same process.
        kernel_part = min(warm, cold)
        rows.append(
            {
                "implementation": key[0],
                "image_size": key[1],
                "kernel": key[2],
                "threads": key[3],
                "repeats": len(totals),
                "end_to_end_ms": total,
                "cold_kernel_ms": cold,
                "warm_kernel_ms": warm,
                "kernel_part_ms": kernel_part,
                "first_call_overhead_ms": cold - kernel_part,
                "process_and_input_ms": max(0.0, total - cold),
                "kernel_share_pct": 100.0 * kernel_part / total,
            }
        )
    return rows


def figure_end_to_end(rows: List[dict], kernel: int, out: Path, formats):
    sizes = sorted({row["image_size"] for row in rows if row["kernel"] == kernel})
    if not sizes:
        return []
    p_max = max(row["threads"] for row in rows if row["kernel"] == kernel and row["implementation"] != "serial")
    panels = [("serial", 1), ("optimized", p_max)]
    fig, axes = panel_axes(len(panels), height=3.0)
    for index, (ax, (implementation, threads)) in enumerate(zip(axes, panels)):
        style_axes(ax)
        lookup = {
            row["image_size"]: row
            for row in rows
            if row["kernel"] == kernel and row["implementation"] == implementation and row["threads"] == threads
        }
        positions = list(range(len(sizes)))
        parts = [
            ("kernel_part_ms", SERIES[implementation]["color"], SURFACE),
            ("first_call_overhead_ms", OVERHEAD_COLORS[0], SURFACE),
            ("process_and_input_ms", OVERHEAD_COLORS[1], INK),
        ]
        bottoms = [0.0] * len(sizes)
        for field, color, text_color in parts:
            shares = []
            for n in sizes:
                row = lookup.get(n)
                shares.append(100.0 * row[field] / row["end_to_end_ms"] if row else 0.0)
            bars = ax.bar(positions, shares, bottom=bottoms, width=0.58, color=color, edgecolor=SURFACE,
                          linewidth=1.0, zorder=3)
            for bar, share, bottom in zip(bars, shares, bottoms):
                if share >= 11:
                    ax.text(bar.get_x() + bar.get_width() / 2, bottom + share / 2, f"{share:.0f}%",
                            ha="center", va="center", fontsize=7.5, color=text_color, zorder=4)
            bottoms = [b + s for b, s in zip(bottoms, shares)]
        for position, n in zip(positions, sizes):
            row = lookup.get(n)
            if row:
                total = row["end_to_end_ms"]
                ax.text(position, 101.5, f"{total:,.0f} ms" if total >= 10 else f"{total:.1f} ms",
                        ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_xticks(positions)
        ax.set_xticklabels([str(n) for n in sizes])
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
        ax.set_title(f"{letter(index)} {series_label(implementation, threads)}")
        ax.set_xlabel("Image size N (N×N input)")
    axes[0].set_ylabel("Share of end-to-end wall time")
    handles = [
        Patch(facecolor=SERIES["serial"]["color"], label="Kernel, sequential (warm)"),
        Patch(facecolor=SERIES["optimized"]["color"], label=f"Kernel, optimized p = {p_max} (warm)"),
        Patch(facecolor=OVERHEAD_COLORS[0], label="First-call overhead (thread start-up, page faults)"),
        Patch(facecolor=OVERHEAD_COLORS[1], label="Process launch + loading the image"),
    ]
    shared_legend(fig, handles, columns=2)
    suptitle(fig, f"End-to-end time per process run, {kernel}×{kernel} kernel (total above each bar)")
    return save(fig, out, "end_to_end", formats)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


def environment_lines(environment: dict) -> List[str]:
    if not environment:
        return ["- (environment.json not found)"]
    lines = []
    macos = environment.get("macos") or {}
    linux = environment.get("linux") or {}
    metadata = environment.get("benchmark_metadata") or {}
    cache = environment.get("cmake_cache") or {}
    if macos:
        sysctl = macos.get("sysctl", {})
        overview = macos.get("hardware_overview", {})
        lines.append(f"- CPU: {sysctl.get('machdep.cpu.brand_string', overview.get('Chip', '?'))}, "
                     f"{sysctl.get('hw.physicalcpu', '?')} physical cores "
                     f"({sysctl.get('hw.perflevel0.physicalcpu', '?')} performance + "
                     f"{sysctl.get('hw.perflevel1.physicalcpu', '?')} efficiency), one hardware thread per core")
        if overview.get("Model Name"):
            lines.append(f"- Machine: {overview.get('Model Name')} ({overview.get('Model Identifier', '?')})")
        lines.append(f"- Memory: {macos.get('memory_gib', '?')} GiB unified memory")
        lines.append(f"- OS: macOS {macos.get('macos_version', '?')} (build {macos.get('macos_build', '?')})")
        lines.append(f"- Power: {macos.get('power_source', '?')}")
        libomp = macos.get("libomp") or {}
        if libomp:
            lines.append(f"- OpenMP runtime: LLVM libomp {libomp.get('version', '?')} (Homebrew)")
    if linux:
        lscpu = linux.get("lscpu", {})
        lines.append(f"- CPU: {lscpu.get('Model name', '?')} ({lscpu.get('CPU(s)', '?')} logical CPUs)")
        lines.append(f"- Memory: {linux.get('memory_gib', '?')} GiB; OS: {linux.get('os_pretty_name', '?')}")
    flags = (cache.get("CMAKE_CXX_FLAGS", "") + " " + cache.get("CMAKE_CXX_FLAGS_RELEASE", "")).strip()
    lines.append(f"- Compiler: {environment.get('compiler_version_banner') or metadata.get('compiler_id')}; "
                 f"flags `{flags}` plus OpenMP; CMake build type {metadata.get('build_type', '?')}")
    lines.append(f"- OpenMP: `_OPENMP` = {metadata.get('openmp_version', '?')}; "
                 f"omp_get_num_procs() = {metadata.get('openmp_processors', '?')}")
    lines.append(f"- {environment.get('cmake_version', 'cmake ?')}; Python {environment.get('python', '?')}")
    git = environment.get("git") or {}
    if environment.get("provenance"):
        lines.append(f"- Provenance: {environment['provenance']}")
    elif git.get("commit"):
        state = "plus uncommitted changes to tracked files" if git.get("tracked_files_modified") else "clean"
        lines.append(f"- Code: git {git['commit'][:12]} ({state}); per-file source SHA-256 in environment.json")
    load = environment.get("load_average_1_5_15")
    if load:
        lines.append(f"- Load average at start (1/5/15 min): {load[0]:.2f} / {load[1]:.2f} / {load[2]:.2f}")
    lines.append(f"- Measured: {environment.get('captured_at_local', '?')}")
    return lines


def markdown_table(header: Sequence[str], rows: Sequence[Sequence[str]], align: Optional[str] = None) -> List[str]:
    align = align or ("l" + "r" * (len(header) - 1))
    rule = ["---:" if a == "r" else "---" for a in align]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(rule) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return lines


def write_summary(results: Results, size: int, kernels: Sequence[int], environment: dict,
                  e2e: List[dict], run_info: dict, path: Path) -> None:
    sizes = results.sizes()
    lines = ["# Benchmark summary", ""]
    lines += ["## Test environment", ""] + environment_lines(environment) + [""]
    params = run_info.get("parameters") or {}
    if params:
        lines += [
            "## Parameters",
            "",
            f"- Image sizes N: {params.get('image_sizes')}; kernel sizes K: {params.get('kernel_sizes')}; "
            f"OpenMP threads p: {params.get('thread_counts')}",
            f"- {params.get('warmup_runs')} untimed warm-up + {params.get('measured_runs')} timed runs per "
            f"configuration ({params.get('matrix_configurations')} configurations); medians reported",
        ]
        dataset = params.get("dataset") or {}
        if dataset.get("images"):
            ids = [entry.get("id") for entry in dataset.get("images", [])]
            lines.append(f"- Dataset: {dataset.get('name')} ({dataset.get('url')}), {len(ids)} images "
                         f"({ids[0]} to {ids[-1]}); each timed call uses the next image in turn")
            lines.append(f"- Preprocessing: {dataset.get('preprocessing')}")
        if params.get("kernel"):
            lines.append(f"- Kernels: {params.get('kernel')}")
        lines.append("")
    lines += ["Throughput is images processed per second, 1000 / median time per image in ms. "
              "S(p) = T_impl(1) / T_impl(p) (self-relative, the same as the throughput ratio); "
              "\"vs serial\" = T_serial / T_impl(p); E(p) = S(p) / p. The end-to-end section stays in "
              "milliseconds because it splits one process run into parts.", ""]
    for kernel in kernels:
        threads = results.threads("optimized", size, kernel)
        serial = results.throughput(("serial", size, kernel, 1))[0]
        lines += [f"## Strong scaling, {size}×{size}, {kernel}×{kernel} kernel "
                  f"(sequential baseline {rate_text(serial)} images/s)", ""]
        rows = []
        for p in threads:
            cells = [str(p)]
            for implementation in ("basic", "optimized"):
                key = (implementation, size, kernel, p)
                s = results.speedup(key)[0]
                cells += [rate_text(results.throughput(key)[0]), f"{s:.2f}", f"{100 * s / p:.0f}%",
                          f"{results.speedup(key, 'serial')[0]:.2f}"]
            rows.append(cells)
        lines += markdown_table(
            ["p", "Basic img/s", "Basic S(p)", "Basic E(p)", "Basic vs serial",
             "Opt. img/s", "Opt. S(p)", "Opt. E(p)", "Opt. vs serial"], rows)
        lines.append("")
    p_max = max(results.threads("optimized", size, kernels[0]))
    for kernel in kernels:
        lines += [f"## Throughput vs image size, {kernel}×{kernel} kernel (OpenMP at p = {p_max})", ""]
        rows = []
        for n in sizes:
            serial = results.throughput(("serial", n, kernel, 1))[0]
            basic = results.throughput(("basic", n, kernel, p_max))[0]
            optimized = results.throughput(("optimized", n, kernel, p_max))[0]
            rows.append([str(n), rate_text(serial), rate_text(basic), rate_text(optimized),
                         f"{basic / serial:.2f}", f"{optimized / serial:.2f}",
                         f"{results.speedup(('optimized', n, kernel, p_max))[0]:.2f}"])
        lines += markdown_table(["N", "Serial img/s", "Basic img/s", "Opt. img/s", "Basic vs serial",
                                 "Opt. vs serial", f"Opt. S({p_max})"], rows)
        lines.append("")
    all_kernels = [k for k in results.kernels() if results.has(("serial", size, k, 1))]
    lines += [f"## Kernel size at {size}×{size} (OpenMP at p = {p_max})", ""]
    rows = []
    for kernel in all_kernels:
        serial_key = ("serial", size, kernel, 1)
        cells = [f"{kernel}×{kernel}", rate_text(results.throughput(serial_key)[0]),
                 f"{results.gflops(serial_key):.1f}"]
        for implementation in ("basic", "optimized"):
            key = (implementation, size, kernel, p_max)
            s = results.speedup(key)[0]
            cells += [rate_text(results.throughput(key)[0]), f"{s:.2f}", f"{100 * s / p_max:.0f}%",
                      f"{results.speedup(key, 'serial')[0]:.2f}", f"{results.gflops(key):.1f}"]
        rows.append(cells)
    lines += markdown_table(["K", "Serial img/s", "Serial GFLOP/s", "Basic img/s", "Basic S", "Basic E",
                             "Basic vs serial", "Basic GFLOP/s", "Opt. img/s", "Opt. S", "Opt. E", "Opt. vs serial",
                             "Opt. GFLOP/s"], rows)
    lines.append("")
    best = []
    for kernel in all_kernels:
        candidates = [((implementation, size, kernel, p), results.throughput((implementation, size, kernel, p))[0])
                      for implementation in ("basic", "optimized")
                      for p in results.threads(implementation, size, kernel)]
        (implementation, _, _, p), highest = max(candidates, key=lambda item: item[1])
        serial = results.throughput(("serial", size, kernel, 1))[0]
        best.append([f"{kernel}×{kernel}", implementation, str(p), rate_text(highest), f"{highest / serial:.2f}"])
    lines += [f"## Highest throughput per kernel at {size}×{size}", ""]
    lines += markdown_table(["K", "Implementation", "p", "Images/s", "Speedup vs serial"], best, align="llrrr")
    lines.append("")
    if e2e:
        lines += ["## End-to-end (whole process, cold) vs kernel time", "",
                  "End-to-end = process launch + loading one dataset image + OpenMP start-up + one cold "
                  "convolution + exit, timed from the parent process (median of repeats). "
                  "Warm kernel = median steady-state call from the main matrix.", ""]
        rows = []
        p_e2e = max(r["threads"] for r in e2e)
        for row in e2e:
            if row["implementation"] == "serial" or (row["threads"] == p_e2e):
                rows.append([row["implementation"], str(row["image_size"]), str(row["kernel"]), str(row["threads"]),
                             f"{row['end_to_end_ms']:,.2f}", f"{row['cold_kernel_ms']:,.3f}",
                             f"{row['warm_kernel_ms']:,.3f}", f"{row['kernel_share_pct']:.0f}%"])
        lines += markdown_table(["Impl.", "N", "K", "p", "End-to-end ms", "Cold kernel ms", "Warm kernel ms",
                                 "Kernel share"], rows, align="lrrrrrrr")
        lines.append("")
        lookup = {(r["implementation"], r["image_size"], r["kernel"], r["threads"]): r for r in e2e}
        rows = []
        for (implementation, n, kernel, threads), row in sorted(lookup.items(),
                                                                 key=lambda item: (item[0][2], item[0][1])):
            serial_row = lookup.get(("serial", n, kernel, 1))
            if implementation != "optimized" or threads != p_e2e or serial_row is None:
                continue
            rows.append([str(n), str(kernel), f"{serial_row['warm_kernel_ms'] / row['warm_kernel_ms']:.2f}",
                         f"{serial_row['end_to_end_ms'] / row['end_to_end_ms']:.2f}"])
        if rows:
            lines += [f"Kernel-only vs end-to-end speedup of Optimized OpenMP (p = {p_e2e}) over the sequential "
                      "baseline. Process start-up and image loading stay serial, so Amdahl's law caps "
                      "the end-to-end gain:", ""]
            lines += markdown_table(["N", "K", "Kernel speedup", "End-to-end speedup"], rows)
            lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Make report figures and tables from a benchmark run.")
    parser.add_argument("--results", type=Path, default=REPOSITORY_ROOT / "results")
    parser.add_argument("--size", type=int, help="image size for the strong-scaling figures (default: largest)")
    parser.add_argument("--kernels", default="3,5", help="kernel sizes for the two-panel figures")
    parser.add_argument("--formats", default="png", help="comma-separated, e.g. png,pdf")
    args = parser.parse_args()

    results_dir = args.results.resolve()
    results = Results.load(results_dir / "samples.csv")
    size = args.size or max(results.sizes())
    kernels = [int(k) for k in args.kernels.split(",") if k.strip()]
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    environment_path = results_dir / "environment.json"
    environment = json.loads(environment_path.read_text(encoding="utf-8")) if environment_path.is_file() else {}
    run_path = results_dir / "run_info.json"
    run_info = json.loads(run_path.read_text(encoding="utf-8")) if run_path.is_file() else {}
    cores = core_split(environment)

    apply_style()
    figures_dir = results_dir
    written: List[Path] = []
    written += figure_speedup(results, size, kernels, cores, figures_dir, formats, baseline="self")
    written += figure_speedup(results, size, kernels, cores, figures_dir, formats, baseline="serial")
    written += figure_throughput_vs_size(results, kernels, figures_dir, formats)
    written += figure_efficiency(results, size, kernels, cores, figures_dir, formats)
    written += figure_kernel_size(results, size, figures_dir, formats)
    written += figure_gflops(results, size, figures_dir, formats)
    written += figure_speedup_vs_size(results, kernels, figures_dir, formats)

    e2e = end_to_end_rows(results, load_end_to_end(results_dir / "end_to_end.csv"))
    if e2e:
        e2e_kernel = kernels[0] if any(row["kernel"] == kernels[0] for row in e2e) else e2e[0]["kernel"]
        written += figure_end_to_end(e2e, e2e_kernel, figures_dir, formats)

    write_summary(results, size, kernels, environment, e2e, run_info, results_dir / "summary.md")
    for path in written:
        print(f"wrote {path.relative_to(results_dir)}")
    print("wrote summary.md")


if __name__ == "__main__":
    main()
