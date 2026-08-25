#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问题一：单次反射与透射条件下的碳化硅外延层厚度模型。

本脚本仅进行理论关系验证、量纲检查和示意图绘制，不读取附件数据，
不执行峰谷检测、参数拟合、厚度估计或多光束 Airy 模型计算。
"""

from __future__ import annotations

import json
import math
import os
import platform
from datetime import datetime
from pathlib import Path

import numpy as np

# 技能规范要求优先使用 TkAgg。图形由 Figure 对象直接保存，因此运行时不弹窗，
# 也能在无交互终端中完成 PNG 输出。
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.figure import Figure
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle


OUTPUT_DIR = Path(__file__).resolve().parent
FIGURE_1 = OUTPUT_DIR / "图1_两光束干涉光路示意图.png"
FIGURE_2 = OUTPUT_DIR / "图2_理论干涉条纹关系示意图.png"
LOG_FILE = OUTPUT_DIR / "问题一诊断日志.txt"
MANIFEST_FILE = OUTPUT_DIR / "paper_manifest.json"


def configure_chinese_font() -> str:
    """选择可用中文字体，并保证坐标轴负号正常显示。"""
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
    ]
    installed = {item.name for item in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
    matplotlib.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["figure.dpi"] = 120
    matplotlib.rcParams["savefig.dpi"] = 220
    return selected


def refracted_angle(theta0: np.ndarray | float, n0: float, n: np.ndarray | float):
    """由斯涅尔定律计算折射角 theta1（弧度）。"""
    theta0 = np.asarray(theta0, dtype=float)
    n = np.asarray(n, dtype=float)
    argument = n0 * np.sin(theta0) / n
    if np.any(np.abs(argument) > 1.0):
        raise ValueError("理论参数不满足实数折射角条件。")
    return np.arcsin(argument)


def optical_factor(theta0: np.ndarray | float, n0: float, n: np.ndarray | float):
    """返回 n cos(theta1)=sqrt(n^2-n0^2 sin^2(theta0))。"""
    theta0 = np.asarray(theta0, dtype=float)
    n = np.asarray(n, dtype=float)
    radicand = n**2 - n0**2 * np.sin(theta0) ** 2
    if np.any(radicand <= 0.0):
        raise ValueError("根号内必须为正；请检查入射角和折射率示意参数。")
    return np.sqrt(radicand)


def optical_path_difference(
    d_relative: np.ndarray | float,
    theta0: np.ndarray | float,
    n0: float,
    n: np.ndarray | float,
):
    """两束光的有效光程差；d_relative 仅表示无量纲相对厚度。"""
    return 2.0 * np.asarray(d_relative, dtype=float) * optical_factor(theta0, n0, n)


def equivalent_optical_wavenumber(
    sigma_relative: np.ndarray | float,
    theta0: np.ndarray | float,
    n0: float,
    n: np.ndarray | float,
):
    """等效光学波数 x=sigma*sqrt(n^2-n0^2 sin^2(theta0))。"""
    return np.asarray(sigma_relative, dtype=float) * optical_factor(theta0, n0, n)


def phase_difference(
    sigma_relative: np.ndarray | float,
    d_relative: np.ndarray | float,
    theta0: np.ndarray | float,
    n0: float,
    n: np.ndarray | float,
    phi_r: float,
):
    """总相位差 delta=4*pi*d*x+phi_r。"""
    x = equivalent_optical_wavenumber(sigma_relative, theta0, n0, n)
    return 4.0 * np.pi * np.asarray(d_relative, dtype=float) * x + phi_r


def normalized_reflectance(x: np.ndarray, d_relative: float, phi_r: float = 0.0):
    """无量纲两光束反射率示意，不代表附件晶圆的实测反射率。"""
    background = 0.50
    envelope = 0.34
    return background + envelope * np.cos(4.0 * np.pi * d_relative * x + phi_r)


def verify_theoretical_relations() -> dict[str, object]:
    """用理论示意参数验证恒等关系，不反演任何实际厚度。"""
    # 以下全部为理论示意参数或无量纲参数，不代表碳化硅晶圆真实参数。
    n0_demo = 1.0
    n_demo = 2.20
    theta0_demo = math.radians(30.0)
    d_relative_demo = 0.80
    sigma_relative_demo = 1.70
    phi_r_demo = math.pi

    theta1_demo = float(refracted_angle(theta0_demo, n0_demo, n_demo))
    snell_residual = n0_demo * math.sin(theta0_demo) - n_demo * math.sin(theta1_demo)

    factor_direct = n_demo * math.cos(theta1_demo)
    factor_eliminated = float(optical_factor(theta0_demo, n0_demo, n_demo))
    factor_residual = factor_direct - factor_eliminated

    delta_l_from_theta1 = 2.0 * n_demo * d_relative_demo * math.cos(theta1_demo)
    delta_l_eliminated = float(
        optical_path_difference(d_relative_demo, theta0_demo, n0_demo, n_demo)
    )
    optical_path_residual = delta_l_from_theta1 - delta_l_eliminated

    delta_from_path = 2.0 * math.pi * sigma_relative_demo * delta_l_eliminated + phi_r_demo
    delta_from_x = float(
        phase_difference(
            sigma_relative_demo,
            d_relative_demo,
            theta0_demo,
            n0_demo,
            n_demo,
            phi_r_demo,
        )
    )
    phase_residual = delta_from_path - delta_from_x

    delta_x_one_period = 1.0 / (2.0 * d_relative_demo)
    same_extrema_phase_increment = 4.0 * math.pi * d_relative_demo * delta_x_one_period

    tolerance = 1e-12
    checks = {
        "斯涅尔定律残差": snell_residual,
        "折射角消元恒等式残差": factor_residual,
        "光程差两种表达残差": optical_path_residual,
        "相位差两种表达残差": phase_residual,
        "相邻同类极值相位增量": same_extrema_phase_increment,
        "相邻同类极值应为2π": abs(same_extrema_phase_increment - 2.0 * math.pi),
    }
    passed = all(abs(value) < tolerance for key, value in checks.items() if "相位增量" not in key)

    return {
        "参数性质": "理论示意参数/无量纲参数，不代表附件晶圆真实参数",
        "理论示意参数": {
            "空气折射率n0": n0_demo,
            "示意折射率n": n_demo,
            "示意入射角（度）": math.degrees(theta0_demo),
            "无量纲相对厚度": d_relative_demo,
            "无量纲波数": sigma_relative_demo,
            "示意反射附加相位": "π",
        },
        "数值恒等式检查": checks,
        "量纲检查": {
            "lambda*sigma": "[L]×[L^-1]=1，通过",
            "光程差DeltaL": "d×无量纲折射率因子=[L]，通过",
            "相位delta": "sigma×DeltaL=[L^-1]×[L]=1，弧度为无量纲，通过",
            "等效光学波数x": "sigma×无量纲因子=[L^-1]，通过",
            "厚度反演d": "1/(2Delta x)=[L]，通过",
            "厘米到微米": "1 cm=10^4 μm，通过",
        },
        "总体通过": passed,
    }


def draw_two_beam_path(font_name: str) -> None:
    """绘制空气—外延层—衬底三层结构和两束反射光。"""
    fig = Figure(figsize=(12.0, 7.2), constrained_layout=True)
    ax = fig.subplots()
    ax.set_xlim(-3.5, 4.6)
    ax.set_ylim(-2.35, 2.15)
    ax.set_aspect("equal")
    ax.axis("off")

    # 三层介质背景。
    ax.add_patch(Rectangle((-3.5, 0.0), 8.1, 2.15, color="#EAF4FF", zorder=0))
    ax.add_patch(Rectangle((-3.5, -1.45), 8.1, 1.45, color="#FFF4D6", zorder=0))
    ax.add_patch(Rectangle((-3.5, -2.35), 8.1, 0.90, color="#D9D9D9", zorder=0))
    ax.plot([-3.5, 4.6], [0.0, 0.0], color="#1F2937", linewidth=2.0)
    ax.plot([-3.5, 4.6], [-1.45, -1.45], color="#1F2937", linewidth=2.0)

    # 法线。
    ax.plot([0.0, 0.0], [-1.78, 1.62], linestyle="--", color="#6B7280", linewidth=1.2)
    ax.text(0.08, 1.45, "法线", fontsize=11, color="#4B5563")

    def arrow(start, end, color, width=2.5, style="-|>", z=3):
        patch = FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=16,
            linewidth=width,
            color=color,
            zorder=z,
        )
        ax.add_patch(patch)

    # 入射光和表面直接反射光。
    arrow((-2.45, 1.72), (0.0, 0.0), "#1D4ED8")
    arrow((0.0, 0.0), (2.45, 1.72), "#DC2626")

    # 进入外延层、在衬底界面一次反射、再透射回空气的光。
    arrow((0.0, 0.0), (0.78, -1.45), "#0F766E")
    arrow((0.78, -1.45), (1.56, 0.0), "#0F766E")
    arrow((1.56, 0.0), (4.01, 1.72), "#7C3AED")

    # 厚度双箭头。
    arrow((-3.00, -0.02), (-3.00, -1.43), "#111827", width=1.5, style="<->", z=4)
    ax.text(-3.24, -0.76, "外延层厚度 $d$", rotation=90, ha="center", va="center", fontsize=12)

    # 入射角和折射角示意圆弧。
    ax.add_patch(Arc((0, 0), 1.22, 1.22, theta1=90, theta2=145, color="#1D4ED8", linewidth=1.6))
    ax.text(-0.64, 0.56, r"入射角 $\theta_0$", fontsize=11, color="#1D4ED8", ha="center")
    ax.add_patch(Arc((0, 0), 0.93, 0.93, theta1=270, theta2=299, color="#0F766E", linewidth=1.6))
    ax.text(0.50, -0.44, r"折射角 $\theta_1$", fontsize=11, color="#0F766E")

    # 中文标签。
    ax.text(-3.18, 1.82, "空气", fontsize=14, weight="bold", color="#1E3A8A")
    ax.text(-3.18, -0.28, "外延层", fontsize=14, weight="bold", color="#92400E")
    ax.text(-3.18, -1.78, "衬底", fontsize=14, weight="bold", color="#374151")
    ax.text(-2.30, 1.32, "入射光", fontsize=12, color="#1D4ED8", rotation=-34)
    ax.text(1.03, 1.12, "外延层表面直接反射光", fontsize=12, color="#DC2626", rotation=34)
    ax.annotate(
        "衬底界面一次反射",
        xy=(0.78, -1.45),
        xytext=(1.62, -1.95),
        fontsize=12,
        color="#0F766E",
        arrowprops={"arrowstyle": "->", "color": "#0F766E", "linewidth": 1.2},
    )
    ax.text(2.24, 0.98, "一次反射后出射的光", fontsize=12, color="#7C3AED", rotation=34)
    ax.text(-0.05, 0.08, "表面反射点", fontsize=10, ha="right", va="bottom")
    ax.text(1.60, -0.09, "再次透射回空气", fontsize=10, ha="left", va="top")

    ax.set_title("单次反射与透射条件下的两光束干涉光路", fontsize=18, weight="bold", pad=18)
    ax.text(
        0.55,
        -2.17,
        "仅保留两束反射光；后续多次反射与透射不属于问题一模型。",
        fontsize=11,
        color="#7F1D1D",
        ha="center",
    )
    fig.savefig(FIGURE_1, bbox_inches="tight", facecolor="white")


def draw_fringe_density(font_name: str) -> None:
    """用无量纲变量说明相对厚度越大，条纹越密。"""
    x = np.linspace(0.0, 7.0, 1600)
    d_thin_relative = 0.60
    d_thick_relative = 1.40
    y_thin = normalized_reflectance(x, d_thin_relative)
    y_thick = normalized_reflectance(x, d_thick_relative)

    fig = Figure(figsize=(12.0, 7.0), constrained_layout=True)
    ax = fig.subplots()
    ax.plot(x, y_thin, color="#2563EB", linewidth=2.4, label="相对较薄（条纹较稀疏）")
    ax.plot(x, y_thick, color="#EA580C", linewidth=2.0, label="相对较厚（条纹较密集）")
    ax.set_xlim(0.0, 7.0)
    ax.set_ylim(0.05, 1.05)
    ax.set_xlabel("归一化等效光学波数 $x$", fontsize=13)
    ax.set_ylabel("归一化反射率", fontsize=13)
    ax.set_title("外延层相对厚度与干涉条纹疏密关系示意图", fontsize=18, weight="bold", pad=16)
    ax.grid(True, color="#CBD5E1", linewidth=0.8, alpha=0.65)
    ax.legend(loc="upper right", fontsize=11, frameon=True)

    # 标出较薄曲线上的两个相邻峰值，对应完整相位周期。
    peak_left = 1.0 / d_thin_relative
    peak_right = peak_left + 1.0 / (2.0 * d_thin_relative)
    y_arrow = 0.91
    ax.vlines([peak_left, peak_right], 0.82, y_arrow, colors="#1E40AF", linestyles="--", linewidth=1.0)
    ax.annotate(
        "",
        xy=(peak_right, y_arrow),
        xytext=(peak_left, y_arrow),
        arrowprops={"arrowstyle": "<->", "color": "#1E40AF", "linewidth": 1.6},
    )
    ax.text(
        (peak_left + peak_right) / 2.0,
        y_arrow + 0.035,
        r"相邻同类极值：$\Delta\delta=2\pi$",
        color="#1E40AF",
        fontsize=11,
        ha="center",
    )
    ax.text(
        0.02,
        0.04,
        "本图采用无量纲参数，仅用于说明模型机理，不代表附件晶圆的实际厚度。",
        transform=ax.transAxes,
        fontsize=11,
        color="#7F1D1D",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FEF2F2", "edgecolor": "#FCA5A5"},
    )
    ax.text(
        0.02,
        0.965,
        r"理论关系：$\delta=4\pi d x+\phi_r$，故相对厚度越大，单位 $x$ 区间内的周期数越多。",
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        color="#334155",
    )
    fig.savefig(FIGURE_2, bbox_inches="tight", facecolor="white")


def model_text() -> str:
    """返回问题一模型的紧凑理论说明。"""
    return """问题一两光束模型（不含任何附件计算）

1. 斯涅尔定律
   n0 sin(theta0) = n(sigma) sin(theta1)
   n(sigma) cos(theta1) = sqrt[n(sigma)^2 - n0^2 sin^2(theta0)]

2. 有效光程差
   DeltaL = 2 n(sigma) d cos(theta1)
          = 2 d sqrt[n(sigma)^2 - n0^2 sin^2(theta0)]
   说明：2d/cos(theta1)只是第二束光在膜内的几何路程。比较两束平行出射光在同一波前
   上的相位时，必须扣除由横向错位带来的外部传播相位；结合斯涅尔定律后得到
   2 n d cos(theta1)，而不是 2 n d/cos(theta1)。

3. 总相位差
   delta(sigma) = 4 pi d sigma sqrt[n(sigma)^2 - n0^2 sin^2(theta0)] + phi_r
   phi_r源自两个界面反射相位的相对差，在理想无吸收介质中通常为0或pi。

4. 两光束反射率
   R(sigma) = A(sigma) + B(sigma) cos(delta(sigma)) + epsilon(sigma)
   A为缓慢变化背景，B为振幅包络，epsilon为噪声与未建模误差。

5. 等效光学波数与厚度反演关系
   x = sigma sqrt[n(sigma)^2 - n0^2 sin^2(theta0)]
   delta = 4 pi d x + phi_r
   相邻同类极值：d = 1/[2(x_(m+1)-x_m)]
   相隔K个周期：d = K/[2(x_(m+K)-x_m)]
   回归形式：x_i = alpha + beta i，d = 1/(2 beta)
   截距alpha吸收未知初始级次和常数反射相位。

6. 常折射率近似
   d = 1/[2 DeltaSigma sqrt(n^2 - n0^2 sin^2(theta0))]
   若n0=1、DeltaSigma用cm^-1，则d(μm)=10^4/[2 DeltaSigma sqrt(n^2-sin^2(theta0))]。
   该式只适用于窄波段初步解释，优先采用n(sigma)色散模型。

7. 色散接口与可辨识性
   Cauchy: n(lambda)=a+b/lambda^2+c/lambda^4
   Sellmeier: n(lambda)^2=1+sum_j[B_j lambda^2/(lambda^2-C_j)]
   掺杂影响明显时可在介电常数中加入Drude修正。若n(sigma)完全未知，单角度条纹通常
   只能识别光学厚度，无法唯一分离折射率与几何厚度。

8. 边界
   本模型仅含两束光。多光束干涉条件、Airy公式及影响属于问题三，本问不展开。
"""


def write_diagnostics(font_name: str, verification: dict[str, object]) -> None:
    limitations = [
        "折射率色散模型或掺杂修正不准确会传递到光学厚度因子。",
        "入射角和波数轴标定误差会改变等效光学波数。",
        "背景漂移、噪声及极值定位误差会影响后续条纹位置识别。",
        "厚度不均匀、表面粗糙度、吸收和散射会削弱或展宽条纹。",
        "若反射附加相位随波数变化，相邻极值作差不能完全消去该项。",
        "若后续多次反射不可忽略，两光束模型将产生系统偏差；该问题留待问题三。",
    ]
    checks = verification["数值恒等式检查"]
    lines = [
        "问题一诊断日志",
        f"生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Python版本：{platform.python_version()}",
        f"中文字体：{font_name}",
        "运行范围：仅理论模型、量纲检查与示意图；未读取附件1—附件4。",
        "禁止项检查：未做峰谷检测、参数拟合、厚度估计、可靠性评价或多光束Airy计算。",
        "",
        "【理论示意参数】",
        json.dumps(verification["理论示意参数"], ensure_ascii=False, indent=2),
        "说明：以上均为理论示意参数或无量纲参数，不代表附件晶圆真实参数。",
        "",
        "【恒等式检查】",
    ]
    for name, value in checks.items():
        if name == "相邻同类极值相位增量":
            lines.append(f"{name}：{value:.12g} rad（理论值2π）")
        else:
            lines.append(f"{name}：{value:.3e}")
    lines.extend(["总体检查：" + ("通过" if verification["总体通过"] else "未通过"), "", "【量纲检查】"])
    lines.extend(f"{name}：{result}" for name, result in verification["量纲检查"].items())
    lines.extend(["", "【核心模型】", model_text(), "【已知限制】"])
    lines.extend(f"{index}. {item}" for index, item in enumerate(limitations, start=1))
    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "问题": "问题一：仅考虑一次反射和透射的外延层厚度测量数学模型",
        "范围声明": "仅建立两光束理论模型，不读取附件数据，不输出具体厚度，不展开问题二或问题三。",
        "采用方法": "斯涅尔定律 + 薄膜两光束干涉 + 等效光学波数线性化 + 相邻同类极值差分/回归接口",
        "main": {
            "正文公式": [
                "有效光程差 DeltaL=2d*sqrt[n(sigma)^2-n0^2*sin^2(theta0)]",
                "总相位差 delta=4*pi*d*x+phi_r",
                "两光束反射率 R=A+B*cos(delta)+epsilon",
                "相邻同类极值 d=1/[2*(x_(m+1)-x_m)]",
            ],
            "正文图片": [
                "图1_两光束干涉光路示意图.png",
                "图2_理论干涉条纹关系示意图.png",
            ],
            "正文表格": ["问题一模型说明.xlsx/核心公式"],
        },
        "supporting": [
            "问题一模型说明.xlsx/符号说明",
            "问题一模型说明.xlsx/模型假设",
            "问题一模型说明.xlsx/建模流程",
            "问题一模型说明.xlsx/适用范围与局限",
        ],
        "appendix_diagnostic": [
            "问题一诊断日志.txt",
            "问题一模型说明.xlsx/论文素材索引",
        ],
        "关键符号": ["d", "lambda", "sigma", "theta0", "theta1", "n0", "n(sigma)", "phi_r", "x", "R(sigma)"],
        "关键参数元数据": [
            {
                "name": "入射角theta0",
                "role": "决定折射角与等效光学波数",
                "why_required": "入射几何影响有效光程差",
                "provenance": "given",
                "selection_rule": "由实验设置给定；问题一仅保留符号",
                "decision_effect": "角度误差会传递至厚度反演",
                "unresolved_user_choice": False,
            },
            {
                "name": "折射率函数n(sigma)",
                "role": "描述色散并将波数转换为等效光学波数",
                "why_required": "几何厚度与折射率共同决定相位",
                "provenance": "assumed",
                "selection_rule": "实际应用需由可靠光学数据、Cauchy/Sellmeier或含Drude修正的模型确定",
                "decision_effect": "模型误差直接造成厚度系统误差",
                "unresolved_user_choice": True,
            },
            {
                "name": "反射附加相位phi_r",
                "role": "描述界面反射半波损失的相对相位",
                "why_required": "确定峰谷的绝对级次对应",
                "provenance": "derived",
                "selection_rule": "由界面折射率关系判断；常数时可被差分/截距吸收",
                "decision_effect": "若随波数变化会造成系统偏差",
                "unresolved_user_choice": False,
            },
        ],
        "建议方案": "正文采用含n(sigma)的等效光学波数模型；常折射率公式仅作窄波段机理说明。",
        "限制": [
            "单角度且折射率未知时，通常只能识别光学厚度。",
            "两光束近似要求后续多次反射可忽略。",
            "吸收、粗糙度、厚度不均匀和相位色散可能破坏理想条纹。",
        ],
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    font_name = configure_chinese_font()
    verification = verify_theoretical_relations()
    if not verification["总体通过"]:
        raise RuntimeError("理论关系验证未通过，请检查公式实现。")
    draw_two_beam_path(font_name)
    draw_fringe_density(font_name)
    write_diagnostics(font_name, verification)
    write_manifest()

    print(model_text())
    print("理论关系验证：通过")
    print("量纲一致性检查：通过")
    print("附件读取：否")
    print("实际厚度计算：否")
    print("已生成：")
    for path in [FIGURE_1, FIGURE_2, LOG_FILE, MANIFEST_FILE]:
        print(f"- {path.name}")


if __name__ == "__main__":
    main()
