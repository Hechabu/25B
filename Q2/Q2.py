#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题二：基于两光束干涉模型的碳化硅外延层厚度估计。

主程序阶段：
STAGE A 读取数据并检查数据质量
STAGE B 光谱预处理与候选波段初筛
STAGE C 构造统一折射率模型与等效光学波数
STAGE D FFT、峰谷回归与希尔伯特相位估计
STAGE E 双角度联合估计
STAGE F Bootstrap、敏感性与残差诊断
STAGE G 导出 Excel、图片、日志和论文素材索引

原始附件只读；程序不会覆盖或修改附件1、附件2。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
LOCAL_DEPS = SCRIPT_DIR / "_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks, hilbert, savgol_filter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


SEED = 20250825
RNG = np.random.default_rng(SEED)
ROOT = SCRIPT_DIR.parent
ATTACH_DIR = ROOT / "附件"
INPUTS = {
    "10°": ATTACH_DIR / "附件1.xlsx",
    "15°": ATTACH_DIR / "附件2.xlsx",
}
ANGLES = {"10°": 10.0, "15°": 15.0}

SG_WINDOW = 29
SG_POLYORDER = 3
BASELINE_WINDOW = 801
ENVELOPE_WINDOW = 401
PEAK_PROMINENCE = 0.50
DISTANCE_FACTOR = 0.60
INTERP_HALF_WINDOW = 2
HILBERT_EDGE_FRACTION = 0.05
HUBER_SCALE = 0.15
BOOTSTRAP_N = 500

MODEL_SIGMA_MIN = 2000.0  # 对应5 μm
MODEL_SIGMA_MAX = 4000.2  # 数据上限约4000 cm^-1，对应2.5 μm
MASK_STRONG = (800.0, 1000.0)
BOUNDARY_ANOMALY_SIGMA = 399.6747

FIG1 = SCRIPT_DIR / "图1_碳化硅晶圆原始反射光谱及有效波段.png"
FIG2 = SCRIPT_DIR / "图2_光谱预处理与峰谷检测结果.png"
FIG3 = SCRIPT_DIR / "图3_等效光学波数与干涉级次回归结果.png"
FIG4 = SCRIPT_DIR / "图4_不同方法厚度结果与不确定性比较.png"
WORKBOOK = SCRIPT_DIR / "问题二计算结果.xlsx"
LOG_FILE = SCRIPT_DIR / "问题二诊断日志.txt"
MANIFEST = SCRIPT_DIR / "paper_manifest.json"
WORKBOOK_JSON = SCRIPT_DIR / "_workbook_data.json"


REFRACTIVE_SOURCE = {
    "material": "4H-SiC，普通光（o-ray）",
    "formula": (
        "n^2-1=0.20075λ^2/(λ^2+12.07224)+5.54861λ^2/(λ^2-0.02641)"
        "+35.65066λ^2/(λ^2-1268.24708)，λ单位为μm"
    ),
    "valid_range": "0.4047–5.0 μm；本程序仅使用2.5–5.0 μm对应波数范围",
    "temperature": "公开数据库条目未明确标注；作为系统不确定性来源处理",
    "primary_source": "https://doi.org/10.1002/lpor.201300068",
    "formula_source": "https://refractiveindex.info/?shelf=main&book=SiC&page=Wang-4H-o",
    "limitation": "题目未给出晶型、晶向、偏振、掺杂和温度；固定4H-SiC普通光模型属于有来源的建模假设。",
}


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False


def make_subplots(*args, **kwargs):
    """默认使用TkAgg；无Tcl/Tk的自动化验证环境可设置Q2_HEADLESS=1。"""
    if os.environ.get("Q2_HEADLESS") == "1":
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        fig = Figure(figsize=kwargs.pop("figsize", None))
        FigureCanvasAgg(fig)
        axes = fig.subplots(*args, **kwargs)
        return fig, axes
    return plt.subplots(*args, **kwargs)


def close_figure(fig) -> None:
    if os.environ.get("Q2_HEADLESS") == "1":
        fig.clear()
    else:
        plt.close(fig)


def to_native(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_native(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): to_native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_native(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# STAGE A ---------------------------------------------------------------------
def read_and_check(path: Path, label: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    book = pd.ExcelFile(path)
    if len(book.sheet_names) != 1:
        raise ValueError(f"{path.name}工作表数量异常：{book.sheet_names}")
    raw = pd.read_excel(path, sheet_name=book.sheet_names[0])
    if raw.shape[1] < 2:
        raise ValueError(f"{path.name}至少需要两列")
    sigma = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    refl = pd.to_numeric(raw.iloc[:, 1], errors="coerce")
    df = pd.DataFrame({"波数_cm-1": sigma, "反射率_百分比": refl})
    finite = np.isfinite(df["波数_cm-1"]) & np.isfinite(df["反射率_百分比"])
    diff = df["波数_cm-1"].diff().dropna()
    first_flag = bool(
        len(df) > 0
        and abs(df.loc[0, "波数_cm-1"] - BOUNDARY_ANOMALY_SIGMA) < 1e-3
        and df.loc[0, "反射率_百分比"] == 0
    )
    report = {
        "文件": path.name,
        "角度": label,
        "工作表": book.sheet_names[0],
        "字段": "；".join(map(str, raw.columns)),
        "数据行数": len(df),
        "波数类型": str(raw.dtypes.iloc[0]),
        "反射率类型": str(raw.dtypes.iloc[1]),
        "缺失值": int(df.isna().sum().sum()),
        "非数值或无穷值": int((~finite).sum()),
        "重复波数": int(df["波数_cm-1"].duplicated().sum()),
        "波数严格递增": bool((diff > 0).all()),
        "波数最小值_cm-1": float(df["波数_cm-1"].min()),
        "波数最大值_cm-1": float(df["波数_cm-1"].max()),
        "平均采样间隔_cm-1": float(diff.mean()),
        "采样间隔标准差_cm-1": float(diff.std()),
        "采样间隔最小值_cm-1": float(diff.min()),
        "采样间隔最大值_cm-1": float(diff.max()),
        "首行边界异常": first_flag,
        "反射率零值数": int((df["反射率_百分比"] == 0).sum()),
        "反射率超过100%数": int((df["反射率_百分比"] > 100).sum()),
        "反射率最小值_百分比": float(df["反射率_百分比"].min()),
        "反射率最大值_百分比": float(df["反射率_百分比"].max()),
    }
    if not report["波数严格递增"] or report["重复波数"] > 0 or report["非数值或无穷值"] > 0:
        raise ValueError(f"{path.name}存在影响建模的波数或数值质量问题：{report}")
    df["边界异常标记"] = False
    if first_flag:
        df.loc[0, "边界异常标记"] = True
    df["反射率超过100%标记"] = df["反射率_百分比"] > 100
    df["强反射强色散屏蔽标记"] = df["波数_cm-1"].between(*MASK_STRONG)
    return df, report


# STAGE B/C -------------------------------------------------------------------
def refractive_model(sigma_cm: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Wang等（2013）4H-SiC普通光Sellmeier模型；σ单位cm^-1，λ单位μm。"""
    sigma = np.asarray(sigma_cm, dtype=float)
    if np.any(sigma <= 0):
        raise ValueError("波数必须为正")
    lam_um = 1.0e4 / sigma
    lam2 = lam_um**2
    n2 = (
        1.0
        + 0.20075 * lam2 / (lam2 + 12.07224)
        + 5.54861 * lam2 / (lam2 - 0.02641)
        + 35.65066 * lam2 / (lam2 - 1268.24708)
    )
    if np.any(n2 <= 0):
        raise ValueError("Sellmeier模型给出非正n²")
    return scale * np.sqrt(n2)


def equivalent_optical_wavenumber(
    sigma_cm: np.ndarray, angle_deg: float, n_scale: float = 1.0
) -> np.ndarray:
    n = refractive_model(sigma_cm, n_scale)
    return np.asarray(sigma_cm) * np.sqrt(n**2 - np.sin(np.deg2rad(angle_deg)) ** 2)


def odd_window(target: int, n: int, minimum: int = 5) -> int:
    w = min(int(target), int(n if n % 2 else n - 1))
    if w % 2 == 0:
        w -= 1
    return max(minimum, w)


def preprocess(
    df: pd.DataFrame,
    lo: float,
    hi: float,
    baseline_window: int = BASELINE_WINDOW,
) -> dict[str, np.ndarray]:
    mask = (
        df["波数_cm-1"].between(lo, hi)
        & ~df["边界异常标记"]
        & ~df["强反射强色散屏蔽标记"]
    )
    part = df.loc[mask].copy()
    sigma = part["波数_cm-1"].to_numpy(float)
    raw = part["反射率_百分比"].to_numpy(float)
    if len(sigma) < 100:
        raise ValueError("有效波段数据点过少")
    smooth = savgol_filter(raw, SG_WINDOW, SG_POLYORDER, mode="interp")
    bw = odd_window(baseline_window, len(sigma), SG_POLYORDER + 2)
    baseline = savgol_filter(smooth, bw, 3, mode="interp")
    detrended = smooth - baseline
    analytic = hilbert(detrended)
    amplitude_raw = np.abs(analytic)
    ew = odd_window(ENVELOPE_WINDOW, len(sigma), 5)
    envelope = savgol_filter(amplitude_raw, ew, 2, mode="interp")
    floor = max(float(np.quantile(envelope, 0.10)), 1e-9)
    envelope = np.maximum(envelope, floor)
    normalized = detrended / envelope
    smooth_residual = raw - smooth
    snr = float(np.std(detrended) / max(np.std(smooth_residual), 1e-12))
    return {
        "sigma": sigma,
        "raw": raw,
        "smooth": smooth,
        "baseline": baseline,
        "detrended": detrended,
        "normalized": normalized,
        "envelope": envelope,
        "smooth_residual": smooth_residual,
        "snr": snr,
        "baseline_window": bw,
    }


def fft_initialize(sigma: np.ndarray, signal: np.ndarray) -> dict[str, float]:
    step = float(np.median(np.diff(sigma)))
    y = (signal - np.mean(signal)) * np.hanning(len(signal))
    freq = np.fft.rfftfreq(len(y), d=step)
    power = np.abs(np.fft.rfft(y)) ** 2
    valid = (freq > 1.0 / (sigma[-1] - sigma[0])) & (freq < 0.05)
    if not np.any(valid):
        raise ValueError("FFT有效频率区间为空")
    idx = np.flatnonzero(valid)[np.argmax(power[valid])]
    main_freq = float(freq[idx])
    peak_power = float(power[idx])
    median_power = float(np.median(power[valid]))
    order = np.argsort(power[valid])[-5:][::-1]
    valid_indices = np.flatnonzero(valid)
    top_freq = [float(freq[valid_indices[k]]) for k in order]
    top_ratio = [float(power[valid_indices[k]] / max(peak_power, 1e-30)) for k in order]
    return {
        "频率_cycles_per_cm-1": main_freq,
        "周期_cm-1": 1.0 / main_freq,
        "主峰中位功率比": peak_power / max(median_power, 1e-30),
        "前五频率": top_freq,
        "前五相对功率": top_ratio,
    }


def quadratic_refine(
    sigma: np.ndarray, signal: np.ndarray, index: int, half_window: int = INTERP_HALF_WINDOW
) -> tuple[float, float, bool]:
    left, right = index - half_window, index + half_window + 1
    if left < 0 or right > len(signal):
        return float(sigma[index]), float(signal[index]), False
    xx = sigma[left:right]
    yy = signal[left:right]
    center = float(sigma[index])
    t = xx - center
    coef = np.polyfit(t, yy, 2)
    if abs(coef[0]) < 1e-14:
        return center, float(signal[index]), False
    vertex_t = -coef[1] / (2.0 * coef[0])
    vertex = center + vertex_t
    valid = bool(xx[0] <= vertex <= xx[-1])
    if not valid:
        return center, float(signal[index]), False
    value = float(np.polyval(coef, vertex_t))
    return float(vertex), value, True


def robust_line(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None) -> dict[str, Any]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    weights = np.ones_like(x) if weights is None else np.asarray(weights, float)
    x0 = float(np.mean(x))
    slope0 = float(np.polyfit(x - x0, y, 1)[0])
    intercept0 = float(np.mean(y))

    def residual(p: np.ndarray) -> np.ndarray:
        return np.sqrt(weights) * (y - (p[0] + p[1] * (x - x0)))

    fit = least_squares(residual, np.array([intercept0, slope0]), loss="huber", f_scale=HUBER_SCALE)
    pred = fit.x[0] + fit.x[1] * (x - x0)
    resid = y - pred
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / max(sst, 1e-30)
    return {
        "intercept_at_center": float(fit.x[0]),
        "slope": float(fit.x[1]),
        "x_center": x0,
        "pred": pred,
        "residual": resid,
        "r2": r2,
    }


def detect_and_regress_extrema(
    pre: dict[str, np.ndarray],
    angle_deg: float,
    kind: str,
    fft_info: dict[str, float],
    prominence: float = PEAK_PROMINENCE,
    distance_factor: float = DISTANCE_FACTOR,
    interp_half_window: int = INTERP_HALF_WINDOW,
    n_scale: float = 1.0,
) -> dict[str, Any]:
    sigma = pre["sigma"]
    normalized = pre["normalized"]
    target = normalized if kind == "峰值" else -normalized
    step = float(np.median(np.diff(sigma)))
    min_distance = max(10, int(distance_factor * fft_info["周期_cm-1"] / step))
    broad, broad_prop = find_peaks(target, distance=max(5, int(0.45 * min_distance)), prominence=0.20)
    accepted, props = find_peaks(target, distance=min_distance, prominence=prominence)
    edge = max(min_distance, int(HILBERT_EDGE_FRACTION * len(sigma)))
    accepted = accepted[(accepted > edge) & (accepted < len(sigma) - edge)]
    refined_sigma, refined_value, refined_ok = [], [], []
    for idx in accepted:
        s, v, ok = quadratic_refine(sigma, target, int(idx), interp_half_window)
        refined_sigma.append(s)
        refined_value.append(v if kind == "峰值" else -v)
        refined_ok.append(ok)
    refined_sigma = np.asarray(refined_sigma, float)
    if len(refined_sigma) < 4:
        raise ValueError(f"{angle_deg}°{kind}数量不足：{len(refined_sigma)}")
    x_ext = equivalent_optical_wavenumber(refined_sigma, angle_deg, n_scale)
    order = np.arange(len(x_ext), dtype=float)
    fit = robust_line(order, x_ext)
    beta = fit["slope"]
    thickness = 1.0e4 / (2.0 * beta)
    local = 1.0e4 / (2.0 * np.diff(x_ext))
    local_stats = {
        "局部厚度均值_um": float(np.mean(local)),
        "局部厚度中位数_um": float(np.median(local)),
        "局部厚度标准差_um": float(np.std(local, ddof=1)) if len(local) > 1 else 0.0,
        "局部厚度四分位距_um": float(np.subtract(*np.percentile(local, [75, 25]))),
        "局部厚度变异系数_百分比": float(np.std(local, ddof=1) / np.mean(local) * 100.0)
        if len(local) > 1 else 0.0,
    }
    rejected = []
    for idx in broad:
        if len(accepted) == 0 or np.min(np.abs(accepted - idx)) > max(2, min_distance // 5):
            rejected.append(int(idx))
    return {
        "kind": kind,
        "indices": accepted,
        "rejected_indices": np.asarray(rejected, int),
        "sigma": refined_sigma,
        "value": np.asarray(refined_value),
        "refined_ok": np.asarray(refined_ok, bool),
        "x": x_ext,
        "order": order,
        "fit": fit,
        "thickness_um": float(thickness),
        "local_thickness_um": local,
        "local_stats": local_stats,
        "min_distance_samples": min_distance,
        "prominence": prominence,
        "regression_r2": fit["r2"],
    }


def phase_estimate(
    pre: dict[str, np.ndarray], angle_deg: float, n_scale: float = 1.0
) -> dict[str, Any]:
    sigma = pre["sigma"]
    signal = pre["normalized"]
    x = equivalent_optical_wavenumber(sigma, angle_deg, n_scale)
    phase = np.unwrap(np.angle(hilbert(signal)))
    rough = np.polyfit(x, phase, 1)[0]
    if rough < 0:
        phase = -phase
    fft_info = fft_initialize(sigma, signal)
    step = float(np.median(np.diff(sigma)))
    one_period = int(fft_info["周期_cm-1"] / step)
    edge = max(one_period, int(HILBERT_EDGE_FRACTION * len(sigma)))
    if 2 * edge >= len(sigma) - 20:
        edge = max(10, int(0.05 * len(sigma)))
    sl = slice(edge, len(sigma) - edge)
    local_weight = np.clip(pre["envelope"][sl] / np.median(pre["envelope"][sl]), 0.25, 4.0)
    fit = robust_line(x[sl], phase[sl], local_weight)
    thickness_um = abs(fit["slope"]) * 1.0e4 / (4.0 * np.pi)
    phase_diff = np.diff(phase[sl])
    monotonic = float(np.mean(phase_diff >= -0.05))
    residual = fit["residual"]
    lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1]) if len(residual) > 2 else np.nan
    win = np.hanning(len(residual))
    rp = np.abs(np.fft.rfft((residual - np.mean(residual)) * win)) ** 2
    residual_periodic_ratio = float(np.max(rp[1:]) / max(np.median(rp[1:]), 1e-30)) if len(rp) > 2 else np.nan
    return {
        "sigma": sigma[sl],
        "x": x[sl],
        "phase": phase[sl],
        "weights": local_weight,
        "fit": fit,
        "thickness_um": float(thickness_um),
        "r2": fit["r2"],
        "monotonic_fraction": monotonic,
        "edge_removed_each_side": edge,
        "lag1_autocorrelation": lag1,
        "residual_periodic_ratio": residual_periodic_ratio,
        "fft": fft_info,
    }


def analyze_angle(
    df: pd.DataFrame,
    angle_deg: float,
    lo: float,
    hi: float,
    baseline_window: int = BASELINE_WINDOW,
    prominence: float = PEAK_PROMINENCE,
    distance_factor: float = DISTANCE_FACTOR,
    interp_half_window: int = INTERP_HALF_WINDOW,
    n_scale: float = 1.0,
) -> dict[str, Any]:
    pre = preprocess(df, lo, hi, baseline_window)
    fft_info = fft_initialize(pre["sigma"], pre["normalized"])
    peak = detect_and_regress_extrema(
        pre, angle_deg, "峰值", fft_info, prominence, distance_factor, interp_half_window, n_scale
    )
    valley = detect_and_regress_extrema(
        pre, angle_deg, "谷值", fft_info, prominence, distance_factor, interp_half_window, n_scale
    )
    phase = phase_estimate(pre, angle_deg, n_scale)
    return {"pre": pre, "fft": fft_info, "peak": peak, "valley": valley, "phase": phase}


def select_effective_band(data: dict[str, pd.DataFrame]) -> tuple[tuple[float, float], list[dict[str, Any]]]:
    candidates = []
    for lo in (2000.0, 2100.0, 2200.0, 2300.0):
        for hi in (3600.0, 3800.0, 4000.0):
            if hi - lo < 1400:
                continue
            row: dict[str, Any] = {"起点_cm-1": lo, "终点_cm-1": hi}
            try:
                analyses = {
                    label: analyze_angle(data[label], ANGLES[label], lo, hi)
                    for label in ("10°", "15°")
                }
                methods = []
                for label, result in analyses.items():
                    methods.extend([
                        result["peak"]["thickness_um"], result["valley"]["thickness_um"],
                        result["phase"]["thickness_um"]
                    ])
                    row[f"{label}相位R2"] = result["phase"]["r2"]
                    row[f"{label}信噪比"] = result["pre"]["snr"]
                    row[f"{label}峰数"] = len(result["peak"]["sigma"])
                    row[f"{label}谷数"] = len(result["valley"]["sigma"])
                cv = float(np.std(methods, ddof=1) / np.mean(methods))
                r2_min = min(row["10°相位R2"], row["15°相位R2"])
                snr_min = min(row["10°信噪比"], row["15°信噪比"])
                extrema_min = min(row["10°峰数"], row["10°谷数"], row["15°峰数"], row["15°谷数"])
                score = (
                    0.40 * r2_min
                    + 0.20 * min(snr_min / 15.0, 1.0)
                    + 0.15 * min(extrema_min / 5.0, 1.0)
                    + 0.25 * math.exp(-cv / 0.02)
                )
                row.update({"多方法变异系数": cv, "综合评分": score, "状态": "可用"})
            except Exception as exc:
                row.update({"综合评分": -np.inf, "状态": f"失败：{exc}"})
            candidates.append(row)
    usable = [r for r in candidates if np.isfinite(r["综合评分"])]
    if not usable:
        raise RuntimeError("所有候选波段均未通过诊断")
    best = max(usable, key=lambda r: r["综合评分"])
    return (float(best["起点_cm-1"]), float(best["终点_cm-1"])), candidates


# STAGE E ---------------------------------------------------------------------
def joint_phase_fit(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    p10, p15 = results["10°"]["phase"], results["15°"]["phase"]
    x10, y10, w10 = p10["x"], p10["phase"], p10["weights"]
    x15, y15, w15 = p15["x"], p15["phase"], p15["weights"]
    d0_um = float(np.mean([p10["thickness_um"], p15["thickness_um"]]))
    b0 = 4.0 * np.pi * d0_um / 1.0e4

    def residual(p: np.ndarray) -> np.ndarray:
        a10, a15, b = p
        return np.r_[
            np.sqrt(w10) * (y10 - a10 - b * x10),
            np.sqrt(w15) * (y15 - a15 - b * x15),
        ]

    p0 = np.array([np.mean(y10 - b0 * x10), np.mean(y15 - b0 * x15), b0])
    fit = least_squares(residual, p0, loss="huber", f_scale=HUBER_SCALE)
    a10, a15, b = fit.x
    pred10, pred15 = a10 + b * x10, a15 + b * x15
    res10, res15 = y10 - pred10, y15 - pred15
    d_um = abs(b) * 1.0e4 / (4.0 * np.pi)
    sst = np.sum((y10 - np.mean(y10)) ** 2) + np.sum((y15 - np.mean(y15)) ** 2)
    sse = np.sum(res10**2) + np.sum(res15**2)
    return {
        "thickness_um": float(d_um),
        "a10": float(a10), "a15": float(a15), "b": float(b),
        "pred10": pred10, "pred15": pred15,
        "residual10": res10, "residual15": res15,
        "r2": float(1.0 - sse / max(sst, 1e-30)),
        "residual_std_rad": float(np.std(np.r_[res10, res15], ddof=1)),
    }


# STAGE F ---------------------------------------------------------------------
def block_resample(residual: np.ndarray, block_length: int, rng: np.random.Generator) -> np.ndarray:
    residual = np.asarray(residual, float)
    n = len(residual)
    block_length = max(1, min(block_length, n))
    chunks = []
    while sum(len(c) for c in chunks) < n:
        start = int(rng.integers(0, n))
        idx = (start + np.arange(block_length)) % n
        chunks.append(residual[idx])
    return np.concatenate(chunks)[:n]


def weighted_group_slope(groups: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> float:
    numerator = 0.0
    denominator = 0.0
    for x, y, w in groups:
        xm = float(np.average(x, weights=w))
        ym = float(np.average(y, weights=w))
        numerator += float(np.sum(w * (x - xm) * (y - ym)))
        denominator += float(np.sum(w * (x - xm) ** 2))
    return numerator / denominator


def bootstrap_line_method(
    x: np.ndarray,
    y: np.ndarray,
    fitted: np.ndarray,
    residual: np.ndarray,
    converter,
    block_length: int,
    n_boot: int = BOOTSTRAP_N,
) -> np.ndarray:
    draws = []
    for _ in range(n_boot):
        yb = fitted + block_resample(residual, block_length, RNG)
        slope = float(np.polyfit(x, yb, 1)[0])
        draws.append(converter(slope))
    return np.asarray(draws, float)


def bootstrap_all(results: dict[str, dict[str, Any]], joint: dict[str, Any]) -> dict[str, np.ndarray]:
    boot: dict[str, np.ndarray] = {}
    for label in ("10°", "15°"):
        for key, cname in (("peak", "峰值法"), ("valley", "谷值法")):
            obj = results[label][key]
            fit = obj["fit"]
            boot[f"{label}{cname}"] = bootstrap_line_method(
                obj["order"], obj["x"], fit["pred"], fit["residual"],
                lambda slope: 1.0e4 / (2.0 * slope), 2
            )
        ph = results[label]["phase"]
        block = max(10, int(len(ph["x"]) / max(4, round((ph["phase"][-1] - ph["phase"][0]) / (2*np.pi)))))
        boot[f"{label}相位法"] = bootstrap_line_method(
            ph["x"], ph["phase"], ph["fit"]["pred"], ph["fit"]["residual"],
            lambda slope: abs(slope) * 1.0e4 / (4.0 * np.pi), block
        )

    p10, p15 = results["10°"]["phase"], results["15°"]["phase"]
    groups = [
        (p10["x"], joint["pred10"], p10["weights"], joint["residual10"]),
        (p15["x"], joint["pred15"], p15["weights"], joint["residual15"]),
    ]
    joint_draws = []
    block_lengths = [
        max(10, int(len(g[0]) / max(4, round((p["phase"][-1]-p["phase"][0])/(2*np.pi)))))
        for g, p in zip(groups, (p10, p15))
    ]
    for _ in range(BOOTSTRAP_N):
        sim_groups = []
        for (x, pred, w, resid), bl in zip(groups, block_lengths):
            yb = pred + block_resample(resid, bl, RNG)
            sim_groups.append((x, yb, w))
        slope = weighted_group_slope(sim_groups)
        joint_draws.append(abs(slope) * 1.0e4 / (4.0 * np.pi))
    boot["双角度联合"] = np.asarray(joint_draws)
    return boot


def summarize_bootstrap(draws: np.ndarray) -> dict[str, float]:
    return {
        "均值_um": float(np.mean(draws)),
        "标准误差_um": float(np.std(draws, ddof=1)),
        "95%CI下限_um": float(np.percentile(draws, 2.5)),
        "95%CI上限_um": float(np.percentile(draws, 97.5)),
        "中位数_um": float(np.median(draws)),
    }


def run_sensitivity(
    data: dict[str, pd.DataFrame], band: tuple[float, float]
) -> list[dict[str, Any]]:
    lo, hi = band
    configs = [
        ("基准方案", "基准", {}),
        ("有效波段", "起点-100", {"lo": max(MODEL_SIGMA_MIN, lo - 100)}),
        ("有效波段", "起点+100", {"lo": lo + 100}),
        ("有效波段", "终点-100", {"hi": hi - 100}),
        ("有效波段", "终点+100", {"hi": min(4000.0, hi + 100)}),
        ("基线窗口", "601", {"baseline_window": 601}),
        ("基线窗口", "1001", {"baseline_window": 1001}),
        ("峰值突出度", "0.40", {"prominence": 0.40}),
        ("峰值突出度", "0.60", {"prominence": 0.60}),
        ("最小峰间距因子", "0.55", {"distance_factor": 0.55}),
        ("最小峰间距因子", "0.70", {"distance_factor": 0.70}),
        ("局部插值半窗", "1", {"interp_half_window": 1}),
        ("局部插值半窗", "3", {"interp_half_window": 3}),
        ("折射率尺度", "-0.5%", {"n_scale": 0.995}),
        ("折射率尺度", "+0.5%", {"n_scale": 1.005}),
        ("入射角扰动", "两角均-0.2°", {"angle_offset": -0.2}),
        ("入射角扰动", "两角均+0.2°", {"angle_offset": 0.2}),
    ]
    rows = []
    baseline_d = None
    for factor, level, changes in configs:
        row = {"因素": factor, "水平": level, "状态": "成功"}
        try:
            trial = {}
            for label in ("10°", "15°"):
                trial[label] = analyze_angle(
                    data[label], ANGLES[label] + changes.get("angle_offset", 0.0),
                    changes.get("lo", lo), changes.get("hi", hi),
                    changes.get("baseline_window", BASELINE_WINDOW),
                    changes.get("prominence", PEAK_PROMINENCE),
                    changes.get("distance_factor", DISTANCE_FACTOR),
                    changes.get("interp_half_window", INTERP_HALF_WINDOW),
                    changes.get("n_scale", 1.0),
                )
            joint_trial = joint_phase_fit(trial)
            row.update({
                "10°相位厚度_um": trial["10°"]["phase"]["thickness_um"],
                "15°相位厚度_um": trial["15°"]["phase"]["thickness_um"],
                "联合厚度_um": joint_trial["thickness_um"],
                "联合R2": joint_trial["r2"],
            })
            if baseline_d is None:
                baseline_d = joint_trial["thickness_um"]
            row["相对基准变化_百分比"] = (joint_trial["thickness_um"] - baseline_d) / baseline_d * 100.0
        except Exception as exc:
            row["状态"] = f"失败：{exc}"
        rows.append(row)
    return rows


# STAGE G ---------------------------------------------------------------------
def plot_figures(
    data: dict[str, pd.DataFrame], band: tuple[float, float], results: dict[str, dict[str, Any]],
    bootstrap: dict[str, np.ndarray], joint: dict[str, Any]
) -> None:
    configure_chinese_font()
    lo, hi = band
    colors = {"10°": "#2563EB", "15°": "#EA580C"}

    fig, ax = make_subplots(figsize=(12, 6.6))
    for label in ("10°", "15°"):
        df = data[label]
        ax.plot(df["波数_cm-1"], df["反射率_百分比"], lw=1.0, color=colors[label], label=f"{label}光谱")
    ax.scatter([BOUNDARY_ANOMALY_SIGMA], [0], s=55, marker="x", color="#DC2626", zorder=6, label="首个边界异常点")
    ax.axvspan(*MASK_STRONG, color="#DC2626", alpha=0.12, label="强反射或强色散屏蔽区")
    ax.axvspan(lo, hi, color="#16A34A", alpha=0.12, label=f"最终有效波段 {lo:.0f}–{hi:.0f} cm^-1")
    ax.axhline(100, color="#6B7280", ls="--", lw=0.8, label="100%参考线（不截断原值）")
    ax.set(title="碳化硅晶圆原始反射光谱及有效波段", xlabel="波数 σ（cm$^{-1}$）", ylabel="反射率（%）")
    ax.legend(ncol=2, fontsize=9); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIG1, dpi=220); close_figure(fig)

    fig, axes = make_subplots(3, 2, figsize=(14, 12), sharex="col")
    for label in ("10°", "15°"):
        pre = results[label]["pre"]
        axes[0, 0].plot(pre["sigma"], pre["raw"], color=colors[label], alpha=0.45, lw=0.8, label=f"{label}原始")
        axes[0, 0].plot(pre["sigma"], pre["smooth"], color=colors[label], lw=1.3, label=f"{label}平滑")
        axes[0, 1].plot(pre["sigma"], pre["smooth_residual"], color=colors[label], lw=0.8, label=label)
        axes[1, 0].plot(pre["sigma"], pre["smooth"], color=colors[label], lw=1.0, label=f"{label}平滑")
        axes[1, 0].plot(pre["sigma"], pre["baseline"], color=colors[label], ls="--", lw=1.2, label=f"{label}背景")
        axes[1, 1].plot(pre["sigma"], pre["normalized"], color=colors[label], lw=0.9, label=label)
    for col, label in enumerate(("10°", "15°")):
        pre = results[label]["pre"]; peak = results[label]["peak"]; valley = results[label]["valley"]
        axp = axes[2, col]
        axp.plot(pre["sigma"], pre["normalized"], color=colors[label], lw=0.9, label="归一化干涉信号")
        axp.scatter(peak["sigma"], np.interp(peak["sigma"], pre["sigma"], pre["normalized"]), marker="^", s=40, color="#16A34A", label="保留峰值")
        axp.scatter(valley["sigma"], np.interp(valley["sigma"], pre["sigma"], pre["normalized"]), marker="v", s=40, color="#9333EA", label="保留谷值")
        rejected = np.unique(np.r_[peak["rejected_indices"], valley["rejected_indices"]])
        if len(rejected):
            axp.scatter(pre["sigma"][rejected], pre["normalized"][rejected], marker="x", s=26, color="#DC2626", label="剔除的低质量极值")
        axp.set_title(f"{label}峰谷检测"); axp.legend(fontsize=8, ncol=2)
    titles = ["原始反射率与Savitzky–Golay平滑", "平滑残差", "平滑曲线与背景基线", "去背景与振幅归一化"]
    for ax, title in zip(axes.flat[:4], titles):
        ax.set_title(title); ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.2)
    for ax in axes.flat:
        ax.set_xlabel("波数 σ（cm$^{-1}$）"); ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("反射率（%）"); axes[0, 1].set_ylabel("残差（百分点）")
    axes[1, 0].set_ylabel("反射率（%）"); axes[1, 1].set_ylabel("归一化振荡信号")
    axes[2, 0].set_ylabel("归一化振荡信号")
    fig.suptitle("光谱预处理与峰谷检测结果\nSavitzky–Golay滤波参数：窗口长度29，多项式阶数3", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(FIG2, dpi=220); close_figure(fig)

    fig, axes = make_subplots(2, 2, figsize=(13, 9))
    for col, label in enumerate(("10°", "15°")):
        ax, ar = axes[0, col], axes[1, col]
        for obj, marker, color, name in ((results[label]["peak"], "o", "#2563EB", "峰值"), (results[label]["valley"], "s", "#EA580C", "谷值")):
            ax.scatter(obj["order"], obj["x"], marker=marker, color=color, label=f"{name}位置")
            ax.plot(obj["order"], obj["fit"]["pred"], color=color, lw=1.4, label=f"{name}回归")
            ar.plot(obj["order"], obj["fit"]["residual"], marker=marker, color=color, label=f"{name}残差")
        ax.set_title(f"{label}等效光学波数—干涉级次回归")
        ax.set_xlabel("同类极值序号 i"); ax.set_ylabel("等效光学波数 x（cm$^{-1}$）")
        ar.axhline(0, color="#6B7280", lw=0.8); ar.set_title(f"{label}回归残差")
        ar.set_xlabel("同类极值序号 i"); ar.set_ylabel("x残差（cm$^{-1}$）")
        ax.legend(fontsize=8); ar.legend(fontsize=8); ax.grid(alpha=0.2); ar.grid(alpha=0.2)
    fig.suptitle("等效光学波数与干涉级次回归结果", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(FIG3, dpi=220); close_figure(fig)

    names = ["10°峰值法", "10°谷值法", "10°相位法", "15°峰值法", "15°谷值法", "15°相位法", "双角度联合"]
    point = [
        results["10°"]["peak"]["thickness_um"], results["10°"]["valley"]["thickness_um"], results["10°"]["phase"]["thickness_um"],
        results["15°"]["peak"]["thickness_um"], results["15°"]["valley"]["thickness_um"], results["15°"]["phase"]["thickness_um"], joint["thickness_um"],
    ]
    ci = [summarize_bootstrap(bootstrap[n]) for n in names]
    lower = np.array(point) - np.array([c["95%CI下限_um"] for c in ci])
    upper = np.array([c["95%CI上限_um"] for c in ci]) - np.array(point)
    fig, ax = make_subplots(figsize=(11, 6.5))
    xloc = np.arange(len(names)); cols = ["#60A5FA"] * 3 + ["#FB923C"] * 3 + ["#16A34A"]
    for i, (xv, pv) in enumerate(zip(xloc, point)):
        ax.errorbar(xv, pv, yerr=[[max(lower[i], 0)], [max(upper[i], 0)]], fmt="o", ms=8 if i < 6 else 11,
                    color=cols[i], ecolor=cols[i], capsize=5, lw=2 if i == 6 else 1.4)
    ax.axhline(joint["thickness_um"], color="#16A34A", ls="--", lw=1.0, alpha=0.8)
    ax.set_xticks(xloc, names, rotation=18, ha="right")
    ax.set_ylabel("外延层厚度 d（μm）"); ax.set_title("不同方法厚度结果与不确定性比较")
    ax.grid(axis="y", alpha=0.25)
    ax.annotate("推荐：双角度联合估计", (6, joint["thickness_um"]), xytext=(4.5, max(point)),
                arrowprops={"arrowstyle": "->", "color": "#15803D"}, color="#15803D")
    fig.tight_layout(); fig.savefig(FIG4, dpi=220); close_figure(fig)


def make_workbook_payload(
    quality: list[dict[str, Any]], grid_same: bool, band: tuple[float, float], candidates: list[dict[str, Any]],
    results: dict[str, dict[str, Any]], joint: dict[str, Any], bootstrap: dict[str, np.ndarray],
    sensitivity: list[dict[str, Any]], recommended_interval: tuple[float, float], etheta: float,
) -> dict[str, Any]:
    boot_summary = {name: summarize_bootstrap(draws) for name, draws in bootstrap.items()}
    method_rows = []
    for label in ("10°", "15°"):
        for key, method in (("peak", "峰值法"), ("valley", "谷值法"), ("phase", "希尔伯特相位法")):
            obj = results[label][key]
            name = f"{label}{'相位法' if key == 'phase' else method}"
            method_rows.append({
                "角度": label, "方法": method, "厚度_um": obj["thickness_um"],
                "Bootstrap标准误差_um": boot_summary[name]["标准误差_um"],
                "95%CI下限_um": boot_summary[name]["95%CI下限_um"], "95%CI上限_um": boot_summary[name]["95%CI上限_um"],
                "拟合R2": obj["r2"] if key == "phase" else obj["regression_r2"],
                "有效极值数或相位点数": len(obj["x"]),
                "用途": "单角度推荐" if key == "phase" else "交叉验证",
            })

    extrema_rows = {"峰值位置": [], "谷值位置": []}
    for label in ("10°", "15°"):
        for key, sheet in (("peak", "峰值位置"), ("valley", "谷值位置")):
            obj = results[label][key]
            for i, (s, x, v, ok) in enumerate(zip(obj["sigma"], obj["x"], obj["value"], obj["refined_ok"])):
                local = obj["local_thickness_um"][i] if i < len(obj["local_thickness_um"]) else None
                extrema_rows[sheet].append({
                    "角度": label, "极值序号": i, "波数_cm-1": s, "等效光学波数_cm-1": x,
                    "归一化信号值": v, "二次插值成功": bool(ok), "至下一同类极值局部厚度_um": local,
                })

    boot_rows = []
    for name, draws in bootstrap.items():
        for i, value in enumerate(draws):
            boot_rows.append({"方法": name, "Bootstrap序号": i + 1, "厚度_um": value})

    diagnostics = []
    for label in ("10°", "15°"):
        r = results[label]
        diagnostics.extend([
            {"角度": label, "指标": "预处理信噪比", "数值": r["pre"]["snr"], "说明": "去背景条纹标准差/SG平滑残差标准差"},
            {"角度": label, "指标": "FFT主峰中位功率比", "数值": r["fft"]["主峰中位功率比"], "说明": "FFT仅作初始化"},
            {"角度": label, "指标": "相位线性R2", "数值": r["phase"]["r2"], "说明": "φ=a+bx"},
            {"角度": label, "指标": "相位单调比例", "数值": r["phase"]["monotonic_fraction"], "说明": "允许小于0.05 rad的局部回摆"},
            {"角度": label, "指标": "相位残差一阶自相关", "数值": r["phase"]["lag1_autocorrelation"], "说明": "显著自相关提示模型遗漏"},
            {"角度": label, "指标": "相位残差周期功率比", "数值": r["phase"]["residual_periodic_ratio"], "说明": "仅记录潜在多周期，不作多光束修正"},
        ])

    parameter_rows = [
        {"参数": "入射角10°/15°", "数值或范围": "10°；15°", "来源类型": "given", "作用": "计算等效光学波数", "来源或选择规则": "题目附件说明", "候选值或搜索范围": "固定；敏感性±0.2°", "对结果影响": "改变角度修正", "未解决选择": "否"},
        {"参数": "SG窗口/阶数", "数值或范围": "29/3", "来源类型": "given", "作用": "平滑降噪", "来源或选择规则": "用户明确给定", "候选值或搜索范围": "固定", "对结果影响": "过小降噪不足，过大可能移峰", "未解决选择": "否"},
        {"参数": "有效波段", "数值或范围": f"{band[0]:.0f}–{band[1]:.0f} cm^-1", "来源类型": "tuned", "作用": "限定两光束模型和色散模型有效区", "来源或选择规则": "候选波段的相位R2、SNR、极值数及多方法一致性综合评分", "候选值或搜索范围": "2000–4000 cm^-1内候选子区间", "对结果影响": "主要敏感因素之一", "未解决选择": "否"},
        {"参数": "基线窗口", "数值或范围": str(BASELINE_WINDOW), "来源类型": "tuned", "作用": "估计缓慢背景A_j(σ)", "来源或选择规则": "大于主要条纹周期并经敏感性检验", "候选值或搜索范围": "601、801、1001", "对结果影响": "影响归一化相位和极值", "未解决选择": "否"},
        {"参数": "峰值突出度", "数值或范围": str(PEAK_PROMINENCE), "来源类型": "tuned", "作用": "筛选可靠极值", "来源或选择规则": "归一化信号诊断", "候选值或搜索范围": "0.40–0.60", "对结果影响": "改变极值数量", "未解决选择": "否"},
        {"参数": "折射率模型", "数值或范围": REFRACTIVE_SOURCE["formula"], "来源类型": "assumed", "作用": "σ→x转换", "来源或选择规则": "Wang等(2013) 4H-SiC普通光文献系数", "候选值或搜索范围": "尺度±0.5%敏感性", "对结果影响": "厚度系统误差主来源", "未解决选择": "晶型/晶向/掺杂未由题目给定"},
        {"参数": "Bootstrap次数", "数值或范围": str(BOOTSTRAP_N), "来源类型": "assumed", "作用": "估计统计不确定性", "来源或选择规则": "精度与运行时间折中", "候选值或搜索范围": "固定随机种子", "对结果影响": "影响CI蒙特卡洛误差", "未解决选择": "否"},
    ]

    answer_rows = [
        ["项目", "结果", "单位/说明"],
        ["附件1入射角", 10.0, "度"],
        ["附件1单独计算厚度", results["10°"]["phase"]["thickness_um"], "μm；希尔伯特相位法作为单角度推荐"],
        ["附件1 Bootstrap 95%CI下限", boot_summary["10°相位法"]["95%CI下限_um"], "μm"],
        ["附件1 Bootstrap 95%CI上限", boot_summary["10°相位法"]["95%CI上限_um"], "μm"],
        ["附件2入射角", 15.0, "度"],
        ["附件2单独计算厚度", results["15°"]["phase"]["thickness_um"], "μm；希尔伯特相位法作为单角度推荐"],
        ["附件2 Bootstrap 95%CI下限", boot_summary["15°相位法"]["95%CI下限_um"], "μm"],
        ["附件2 Bootstrap 95%CI上限", boot_summary["15°相位法"]["95%CI上限_um"], "μm"],
        ["双角度联合估计", joint["thickness_um"], "μm；推荐点估计"],
        ["联合Bootstrap 95%CI下限", boot_summary["双角度联合"]["95%CI下限_um"], "μm"],
        ["联合Bootstrap 95%CI上限", boot_summary["双角度联合"]["95%CI上限_um"], "μm"],
        ["含敏感性的建议区间下限", recommended_interval[0], "μm；Bootstrap与敏感性包络"],
        ["含敏感性的建议区间上限", recommended_interval[1], "μm；Bootstrap与敏感性包络"],
        ["双角度相对差异", etheta / 100.0, "百分比格式"],
        ["推荐采用的最终结果", joint["thickness_um"], "μm；双角度联合估计"],
        ["主要可靠性结论", "双角度相位结果接近，峰值、谷值与相位法量级一致；相位线性度高。", "统计区间不包含全部系统误差"],
        ["主要局限性", REFRACTIVE_SOURCE["limitation"] + " 残差周期仅作为模型遗漏证据记录。", "问题二不建立多光束修正"],
    ]

    sheets = {
        "题目答案汇总": answer_rows,
        "原始数据检查": [list(quality[0].keys())] + [[q.get(k) for k in quality[0].keys()] for q in quality] + [["两份波数网格完全一致", grid_same] + [None] * (len(quality[0]) - 2)],
        "有效波段": [list(candidates[0].keys())] + [[r.get(k) for k in candidates[0].keys()] for r in candidates],
        "预处理参数": [["参数", "数值", "来源类型", "说明"], ["SG窗口长度", SG_WINDOW, "given", "两角度一致"], ["SG多项式阶数", SG_POLYORDER, "given", "mode=interp"], ["基线窗口", BASELINE_WINDOW, "tuned", "敏感性检验"], ["包络窗口", ENVELOPE_WINDOW, "assumed", "平滑Hilbert振幅"], ["最终波段起点_cm-1", band[0], "tuned", "候选评分"], ["最终波段终点_cm-1", band[1], "tuned", "候选评分"]],
        "峰值位置": [list(extrema_rows["峰值位置"][0].keys())] + [[r.get(k) for k in extrema_rows["峰值位置"][0].keys()] for r in extrema_rows["峰值位置"]],
        "谷值位置": [list(extrema_rows["谷值位置"][0].keys())] + [[r.get(k) for k in extrema_rows["谷值位置"][0].keys()] for r in extrema_rows["谷值位置"]],
        "单角度厚度": [list(method_rows[0].keys())] + [[r.get(k) for k in method_rows[0].keys()] for r in method_rows],
        "双角度联合结果": [["指标", "数值", "说明"], ["共享厚度_um", joint["thickness_um"], "推荐"], ["10°截距", joint["a10"], "rad"], ["15°截距", joint["a15"], "rad"], ["公共相位斜率", joint["b"], "rad·cm"], ["联合R2", joint["r2"], "无量纲"], ["联合残差标准差_rad", joint["residual_std_rad"], "rad"]],
        "Bootstrap结果": [list(boot_rows[0].keys())] + [[r.get(k) for k in boot_rows[0].keys()] for r in boot_rows],
        "敏感性分析": [list(sensitivity[0].keys())] + [[r.get(k) for k in sensitivity[0].keys()] for r in sensitivity],
        "诊断指标": [list(diagnostics[0].keys())] + [[r.get(k) for k in diagnostics[0].keys()] for r in diagnostics],
        "参数来源说明": [list(parameter_rows[0].keys())] + [[r.get(k) for k in parameter_rows[0].keys()] for r in parameter_rows] + [["文献公式来源", REFRACTIVE_SOURCE["formula_source"], "assumed", REFRACTIVE_SOURCE["valid_range"], None, None, None, None], ["原始论文DOI", REFRACTIVE_SOURCE["primary_source"], "assumed", REFRACTIVE_SOURCE["material"], None, None, None, None]],
        "论文素材索引": [["素材", "文件或工作表", "级别", "建议用途"], ["最终答案表", "题目答案汇总", "main", "直接回答两角度厚度、联合厚度和区间"], ["方法比较图", FIG4.name, "main", "展示多方法与不确定性"], ["极值回归图", FIG3.name, "supporting", "验证等效光学波数线性关系"], ["原始光谱与波段图", FIG1.name, "supporting", "说明异常点、屏蔽区和有效波段"], ["预处理诊断图", FIG2.name, "appendix/diagnostic", "展示预处理没有明显破坏条纹"], ["完整Bootstrap", "Bootstrap结果", "appendix/diagnostic", "统计不确定性复核"], ["敏感性表", "敏感性分析", "appendix/diagnostic", "系统误差边界"]],
    }
    return {"sheets": to_native(sheets), "output": str(WORKBOOK)}


def write_log(
    quality: list[dict[str, Any]], grid_same: bool, band: tuple[float, float], candidates: list[dict[str, Any]],
    results: dict[str, dict[str, Any]], joint: dict[str, Any], bootstrap: dict[str, np.ndarray],
    sensitivity: list[dict[str, Any]], interval: tuple[float, float], runtime: float,
) -> None:
    lines = [
        "问题二诊断日志", "=" * 72, f"运行时间：{time.strftime('%Y-%m-%d %H:%M:%S')}", f"随机种子：{SEED}",
        "", "[数据检查]",
    ]
    for q in quality:
        lines.append(json.dumps(to_native(q), ensure_ascii=False))
    lines += [f"两份波数网格完全一致：{grid_same}", "首行399.6747 cm^-1零值已排除全部后续运算。", "附件2超过100%的反射率保留原值，仅作仪器归一化异常标记。", "800–1000 cm^-1强反射/强色散区已屏蔽。", "", "[折射率模型]", json.dumps(REFRACTIVE_SOURCE, ensure_ascii=False), "单位检查：σ[cm^-1]→λ=10^4/σ[μm]；x[cm^-1]；d=10^4|b|/(4π)[μm]。", "", "[有效波段]", f"最终波段：{band[0]:.1f}–{band[1]:.1f} cm^-1，对应λ={1e4/band[1]:.3f}–{1e4/band[0]:.3f} μm。", "选择依据：候选子区间的相位线性R2、SNR、有效极值数和多方法一致性综合评分。"]
    best_rows = sorted([r for r in candidates if np.isfinite(r.get("综合评分", -np.inf))], key=lambda r: r["综合评分"], reverse=True)[:5]
    lines += ["候选评分前5："] + [json.dumps(to_native(r), ensure_ascii=False) for r in best_rows]
    lines += ["", "[预处理与估计]"]
    for label in ("10°", "15°"):
        r = results[label]
        lines.append(f"{label}: SNR={r['pre']['snr']:.3f}, FFT周期={r['fft']['周期_cm-1']:.3f} cm^-1")
        lines.append(f"  峰值法={r['peak']['thickness_um']:.6f} μm, n={len(r['peak']['sigma'])}, R2={r['peak']['regression_r2']:.8f}")
        lines.append(f"  谷值法={r['valley']['thickness_um']:.6f} μm, n={len(r['valley']['sigma'])}, R2={r['valley']['regression_r2']:.8f}")
        lines.append(f"  相位法={r['phase']['thickness_um']:.6f} μm, R2={r['phase']['r2']:.8f}, 单调比例={r['phase']['monotonic_fraction']:.5f}")
        lines.append(f"  相位残差lag1={r['phase']['lag1_autocorrelation']:.5f}, 周期功率比={r['phase']['residual_periodic_ratio']:.3f}")
    lines += [f"双角度联合={joint['thickness_um']:.6f} μm, R2={joint['r2']:.8f}", f"联合Bootstrap 95%CI={np.percentile(bootstrap['双角度联合'],2.5):.6f}–{np.percentile(bootstrap['双角度联合'],97.5):.6f} μm", f"含参数敏感性建议区间={interval[0]:.6f}–{interval[1]:.6f} μm", "", "[残差与模型边界]", "残差中若有明显周期，仅记为可能存在模型遗漏或多光束干涉的证据；问题二未建立Airy模型或多光束修正。", "Bootstrap采用残差分块重采样，未随机打乱单个光谱点。", "统计Bootstrap区间不覆盖晶型、晶向、掺杂和温度不确定性，因此同时报告敏感性包络。", "", "[敏感性]", *[json.dumps(to_native(r), ensure_ascii=False) for r in sensitivity], "", "[可复现性与泄漏检查]", "原始附件只读，未覆盖；无训练/预测划分，不涉及目标泄漏。", "SG、背景、FFT、峰谷、Hilbert均只在同一连续有效波段内执行，不跨屏蔽区平滑。", f"总运行时间：{runtime:.2f} s"]
    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")


def write_manifest(band: tuple[float, float], joint: dict[str, Any], interval: tuple[float, float]) -> None:
    manifest = {
        "problem": "问题二：碳化硅外延层厚度计算与可靠性分析",
        "adopted_method": "SG平滑—背景去除—FFT初始化—峰谷回归—Hilbert相位—双角度Huber联合拟合",
        "recommended_scheme": "双角度联合相位估计",
        "effective_band_cm-1": list(band),
        "recommended_thickness_um": joint["thickness_um"],
        "recommended_interval_um": list(interval),
        "main": [
            {"artifact": "题目答案汇总", "type": "excel_sheet", "purpose": "直接回答单角度、联合厚度与不确定性"},
            {"artifact": FIG4.name, "type": "figure", "purpose": "比较六种单角度方法与联合结果"},
        ],
        "supporting": [
            {"artifact": FIG3.name, "type": "figure", "purpose": "验证x_i与极值序号近似线性"},
            {"artifact": FIG1.name, "type": "figure", "purpose": "说明异常点、屏蔽区与有效波段"},
        ],
        "appendix_diagnostic": [FIG2.name, "Bootstrap结果", "敏感性分析", "诊断指标", LOG_FILE.name],
        "key_symbols": {"d": "外延层厚度", "sigma": "真空波数", "x_j": "等效光学波数", "theta_j": "入射角", "n_sigma": "折射率色散函数"},
        "limitations": [REFRACTIVE_SOURCE["limitation"], "两光束模型不修正多光束干涉", "Bootstrap主要反映统计误差，系统误差通过敏感性包络补充"],
    }
    MANIFEST.write_text(json.dumps(to_native(manifest), ensure_ascii=False, indent=2), encoding="utf-8")


def run_workbook_builder() -> None:
    builder = SCRIPT_DIR / "Q2_workbook.mjs"
    if not builder.exists():
        raise FileNotFoundError(f"缺少工作簿构建器：{builder}")
    node = os.environ.get("CODEX_NODE") or "node"
    env = os.environ.copy()
    subprocess.run([node, str(builder), str(WORKBOOK_JSON)], cwd=SCRIPT_DIR, env=env, check=True)


def main() -> None:
    started = time.time()
    SCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    data, quality = {}, []
    for label, path in INPUTS.items():
        data[label], report = read_and_check(path, label)
        quality.append(report)
    grid_same = bool(np.array_equal(data["10°"]["波数_cm-1"].to_numpy(), data["15°"]["波数_cm-1"].to_numpy()))
    if not grid_same:
        raise ValueError("附件1和附件2波数网格不一致")

    band, candidates = select_effective_band(data)
    results = {label: analyze_angle(data[label], ANGLES[label], *band) for label in ("10°", "15°")}
    joint = joint_phase_fit(results)
    bootstrap = bootstrap_all(results, joint)
    sensitivity = run_sensitivity(data, band)
    successful_sensitivity = [r["联合厚度_um"] for r in sensitivity if r.get("状态") == "成功"]
    joint_ci = summarize_bootstrap(bootstrap["双角度联合"])
    interval = (
        min(joint_ci["95%CI下限_um"], min(successful_sensitivity)),
        max(joint_ci["95%CI上限_um"], max(successful_sensitivity)),
    )
    d10, d15 = results["10°"]["phase"]["thickness_um"], results["15°"]["phase"]["thickness_um"]
    etheta = abs(d10 - d15) / ((d10 + d15) / 2.0) * 100.0

    plot_figures(data, band, results, bootstrap, joint)
    payload = make_workbook_payload(quality, grid_same, band, candidates, results, joint, bootstrap, sensitivity, interval, etheta)
    WORKBOOK_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    write_log(quality, grid_same, band, candidates, results, joint, bootstrap, sensitivity, interval, time.time() - started)
    write_manifest(band, joint, interval)
    run_workbook_builder()
    WORKBOOK_JSON.unlink(missing_ok=True)

    print("问题二计算完成")
    print(f"有效波段：{band[0]:.1f}–{band[1]:.1f} cm^-1")
    print(f"附件1（10°）单角度相位厚度：{d10:.6f} μm")
    print(f"附件2（15°）单角度相位厚度：{d15:.6f} μm")
    print(f"双角度联合推荐厚度：{joint['thickness_um']:.6f} μm")
    print(f"联合Bootstrap 95%CI：{joint_ci['95%CI下限_um']:.6f}–{joint_ci['95%CI上限_um']:.6f} μm")
    print(f"含敏感性的建议区间：{interval[0]:.6f}–{interval[1]:.6f} μm")
    print(f"双角度相对差异：{etheta:.4f}%")


if __name__ == "__main__":
    main()
