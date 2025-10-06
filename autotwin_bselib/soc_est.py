# soc_with_auto_PE.py
# -*- coding: utf-8 -*-
"""
Battery SOC Estimation with optional quick PE refinement
双语注释 / Bilingual comments

满足需求 / What this script does:
- 函数式接口 run_soc_period(rack_id, start_time, end_time, show_plots)
- 固定使用项目默认 OCV 表（不可切换）
- 默认数据目录：脚本同目录下 opreational_dataset_for_SOC_e
- 任一 Gate 未通过 ⇒ 不中断，最终整段 SOC = -1（不展示其他）
- 其他错误（OCV/早检/重辨识/EKF） ⇒ 整段 SOC = 0
- 早检不达阈值 ⇒ 仅用该段前 ~1/3 数据做一次 quick-PE，并贯穿整段
- 需要画图时只在最后统一展示，不影响过程

CLI (简洁入口)：
python soc_with_auto_PE.py --rack 09 --start "2025-08-01 00:00:00" --end "2025-08-05 23:59:59" --show
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime as _dt
from typing import Optional
from pathlib import Path

# ===== 外部依赖模块 / Project modules =====
from autotwin_bselib.ocv import load_ocv_tables
from autotwin_bselib.ekf_core import run_ekf
from autotwin_bselib.ekf_core import OCVInterp
from autotwin_bselib.pe import (
    load_pe_params,
    quick_identify_params,
    save_pe_result,         # 未直接用到，但保留 import 以兼容后续扩展
)
from autotwin_bselib.pe import simulate_voltage
from autotwin_bselib.io_utils import (
    rename_rack_files_with_time,
    pick_triplet, load_triplet, resample_60s
)
from autotwin_bselib.gates import (
    gate1_first_ts_bar, gate2_cc_hard, gate2b_strict_ocv_rest
)

# ===== constants (保持与你原工程一致) =====
Q_nom           = 105.0
SOC_min_real    = 0.10
SOC_max_real    = 0.94
deltaT          = 60.0   # seconds
pack_series     = 16*15  # modules * cells
I_idle_thresh   = 0.001
rest_min_len    = 50
dV_flat_thr     = 0.05
tol_first_ms    = 5
seg_mae_thr     = 0.08
min_points_PE   = 3000
tol_OCV_pack    = 10.0
min_points_keep = 3000

# Early-check window and threshold
early_use_ratio = 1/3
early_max_pts   = 3600
early_min_pts   = 600
fit_threshold   = 80.0

# Fusion thresholds
S_low  = 0.10
S_high = 0.20
slope_floor = 1e-5


# --- Backward-compatible wrapper for pick_triplet ---
# 某些版本的 pick_triplet(dataFolder, rack_id) 不支持 ts_in 形参；
# 本包装优先尝试三参版本，若 TypeError 则回退到两参版本。
def _pick_triplet_bc(data_folder: str, rack_id: str, ts_in: str = ""):
    try:
        # Newer signature: pick_triplet(data_folder, rack_id, ts_in)
        return pick_triplet(data_folder, rack_id, ts_in)
    except TypeError:
        # Older signature: pick_triplet(data_folder, rack_id)
        return pick_triplet(data_folder, rack_id)



# ========== 可调用函数（核心实现，双语注释） / Callable function ==========

def run_soc_period(
    rack_id: str,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    show_plots: bool = False,
    dataset_dir: Optional[str] = None,  # <— 新增
    model_dir: Optional[str] = None,    # <— 新增
):

    """
    以函数方式运行 SOC 估计，仅返回指定时间段的用户口径 SOC（%）。
    Run SOC estimation as a function and return user-facing SOC (%) over a given period.

    规则 / Rules:
    - OCV 只使用项目内默认 SOC-OCV 表格（不可切换）。
      OCV is forced to use the project's default SOC-OCV tables (no alternative).
    - 输入时间支持人类可读格式，并自动对齐到数据 60s 采样网格。
      Human-readable time is accepted and snapped to the 60s dataset grid.
    - 若任一 Gate（1/2/2b）未通过（含重采样点数不足等），不打断流程，最终输出段内 SOC=-1。
      If any gate fails, do NOT abort; final SOC for the requested period becomes -1.
    - 其他失败（如 OCV 加载、早检/重辨识/EKF 失败）→ 段内 SOC=0。
      For other failures (OCV load, early-check/quick-PE/EKF), output SOC=0 over the period.
    - 若参数早检不达阈值，仅用该段前 1/3 数据做一次重辨识，并用这组参数贯穿该段。
      If early-fit below threshold, re-identify ONCE using the first ~1/3 window of THIS period and
      use the resulting params for the whole period.

    参数 / Args:
        rack_id: 机架ID（'09'或'9'均可） / rack ID (accepts '09' or '9')
        start_time, end_time: 人类可读时间（如 '2025-08-01 00:00:00' 或 '20250801000000'）
                              Human-readable times; several formats supported (see parser below).
        show_plots: 仅在结束时统一展示图，不影响计算。
                    Show plots only at the very end (if True), never during the pipeline.

    返回 / Returns:
        dict:
          - status: "ok" | "gate_fail" | "fail"
          - rack_id: "09"
          - time_axis: List[str]  # '%Y-%m-%d %H:%M:%S'
          - soc_estimated: List[float]  # user-facing SOC (%), or -1 / 0 according to failure rules
          - reason: Optional[str]
    """
    # ========= 内部工具 / Helpers =========
    def _build_axis_by_range(sdt: _dt.datetime, edt: _dt.datetime):
        """按 60s 生成时间轴（闭区间）；Generate a 60s time axis over [sdt, edt]."""
        axis = []
        cur = sdt
        while cur <= edt:
            axis.append(cur)
            cur += _dt.timedelta(seconds=deltaT)
        return axis

    def _pack_out(status: str, rack_id_fmt: str, wall_times: list, soc_value: float, reason: str):
        """打包输出；Pack the final output with a constant SOC value over wall_times."""
        return {
            "status": status,
            "rack_id": rack_id_fmt,
            "time_axis": [t.strftime("%Y-%m-%d %H:%M:%S") for t in wall_times],
            "soc_estimated": [float(soc_value)] * len(wall_times),
            "reason": reason
        }

    def _parse_human_time(s: Optional[str], fallback: Optional[_dt.datetime]) -> _dt.datetime:
        """解析人类可读时间；Parse human-readable time with multiple formats."""
        if s is None:
            if fallback is None:
                raise ValueError("No time provided")
            return fallback
        s = s.strip()
        fmts = [
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d %b %Y %H:%M:%S",
            "%d %b %Y %H:%M",
            "%d %b %Y",
        ]
        for fmt in fmts:
            try:
                return _dt.datetime.strptime(s, fmt)
            except Exception:
                continue
        # 兼容 yyyymmddHHMMSS；Also accept compact format
        try:
            return _dt.datetime.strptime(s, "%Y%m%d%H%M%S")
        except Exception:
            pass
        raise ValueError(f"Invalid time format: {s}")

    def _snap_to_grid(dt0: _dt.datetime, base: _dt.datetime) -> _dt.datetime:
        """对齐到以 base 为起点、步长 deltaT 的最近采样点（四舍五入）。
           Snap dt0 to the nearest grid point defined by base + k*deltaT."""
        sec = (dt0 - base).total_seconds()
        k = int(round(sec / deltaT))
        return base + _dt.timedelta(seconds=k * deltaT)

    # ========= 抑制过程展示，最后统一展示 / Suppress live plotting, show at end =========
    rack_id_fmt = f"{int(rack_id):02d}"
    _orig_show = plt.show
    try:
        def _silent_show(*a, **k): return None
        plt.show = _silent_show
        plt.ioff()

       # ========= 选择数据目录 / Choose dataset folder =========
        if dataset_dir is not None:
            dataFolder = os.path.abspath(os.path.expanduser(dataset_dir))
        else:
            dataFolder = os.path.join(os.path.dirname(__file__), "dataset")

        # 若数据目录不存在：按请求时间返回 0（非 Gate 错误）
        if not os.path.isdir(dataFolder):
            try:
                sdt_req = _parse_human_time(start_time, _dt.datetime(1970,1,1))
                edt_req = _parse_human_time(end_time, sdt_req)
                if edt_req < sdt_req: sdt_req, edt_req = edt_req, sdt_req
                req_axis = _build_axis_by_range(sdt_req, edt_req)
                return _pack_out("fail", rack_id_fmt, req_axis, 0.0, f"Data folder not found: {dataFolder}")
            except Exception as e:
                return {"status": "fail", "rack_id": rack_id_fmt, "time_axis": [], "soc_estimated": [], "reason": f"Bad time or data folder: {repr(e)}"}

        # ========= OCV 只用默认（不可切换）/ OCV fixed to project default =========
        try:
            OCV = load_ocv_tables(model_dir)
            ocv_interp = OCVInterp(OCV["SOC_c"], OCV["OCV_c"], OCV["SOC_d"], OCV["OCV_d"])
        except Exception as e:
            # OCV 失败：按请求返回 0（非 Gate 错误）
            try:
                sdt_req = _parse_human_time(start_time, _dt.datetime(1970,1,1))
                edt_req = _parse_human_time(end_time, sdt_req)
                if edt_req < sdt_req: sdt_req, edt_req = edt_req, sdt_req
                req_axis = _build_axis_by_range(sdt_req, edt_req)
                return _pack_out("fail", rack_id_fmt, req_axis, 0.0, f"OCV load failure: {repr(e)}")
            except Exception as e2:
                return {"status": "fail", "rack_id": rack_id_fmt, "time_axis": [], "soc_estimated": [], "reason": f"OCV load failure + bad time: {repr(e)}; {repr(e2)}"}

        # ========= 门禁（Gate）与重采样 / Gates and resampling =========
        try:
            rename_rack_files_with_time(dataFolder, dry_run=False, data_header_lines=3, tolerance_ms=tol_first_ms)
            curFile, volFile, socFile, tagTS = _pick_triplet_bc(dataFolder, rack_id_fmt, ts_in="")
            raw_t_ms3, raw_I, raw_V, raw_SOCpct = load_triplet(curFile, volFile, socFile, header_lines=3)

            # Gate 1
            gate1_first_ts_bar(raw_t_ms3, rack_id_fmt, tol_first_ms=tol_first_ms, save=False)

            # 数据拉直 / Arrays
            t_ms    = raw_t_ms3[:, 0]
            I_meas  = raw_I.astype(float)
            V_meas  = raw_V.astype(float)
            SOC_pct = raw_SOCpct.astype(float)
            SOC_real = SOC_pct/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real

            # Gate 2
            gate2_cc_hard(I_meas, SOC_real, t_ms, Q_nom, SOC_min_real, SOC_max_real,
                          thr=seg_mae_thr, rack_id=rack_id_fmt, save=False)

            # Gate 2b
            t_ms, raw_I, raw_V, SOC_real_all = gate2b_strict_ocv_rest(
                t_ms, I_meas, V_meas, SOC_real, ocv_interp,
                pack_series=pack_series,
                I_idle_thresh=I_idle_thresh, dV_flat_thr=dV_flat_thr, rest_min_len=rest_min_len,
                tol_OCV_pack=tol_OCV_pack, min_points_keep=min_points_keep,
                rack_id=rack_id_fmt, save=False
            )

            # 重采样 / Resample to 60s
            t_grid, I_grid, V_grid, SOCp_grid = resample_60s(
                t_ms, raw_I, raw_V, SOCp=SOC_real_all * 100.0, deltaT=deltaT
            )
            Nrs = len(t_grid)
            if Nrs < min_points_PE:
                # Gate 等价失败：最终段内输出 -1
                try:
                    try:
                        t0_wall = _dt.datetime.strptime(tagTS, "%Y%m%d%H%M%S")
                    except Exception:
                        t0_wall = _dt.datetime(1970,1,1,0,0,0)
                    all_wall_times = [t0_wall + _dt.timedelta(seconds=float(k)*deltaT) for k in range(len(t_grid))]
                    sdt_req = _parse_human_time(start_time, all_wall_times[0] if all_wall_times else t0_wall)
                    edt_req = _parse_human_time(end_time,   all_wall_times[-1] if all_wall_times else t0_wall)
                    if edt_req < sdt_req: sdt_req, edt_req = edt_req, sdt_req
                    req_axis = _build_axis_by_range(sdt_req, edt_req)
                    return _pack_out("gate_fail", rack_id_fmt, req_axis, -1.0, f"Too few points after resampling: {Nrs} < {min_points_PE}")
                except Exception as e2:
                    return {"status": "gate_fail", "rack_id": rack_id_fmt, "time_axis": [], "soc_estimated": [], "reason": f"Too few points + time resolve error: {repr(e2)}"}

        except Exception as e:
            # 任一 Gate 抛错：最终段内输出 -1
            try:
                try:
                    t0_wall = _dt.datetime.strptime(tagTS, "%Y%m%d%H%M%S") if 'tagTS' in locals() else _dt.datetime(1970,1,1,0,0,0)
                except Exception:
                    t0_wall = _dt.datetime(1970,1,1,0,0,0)
                all_wall_times = [t0_wall + _dt.timedelta(seconds=float(k)*deltaT) for k in range(len(t_grid))] if 't_grid' in locals() else []
                sdt_req = _parse_human_time(start_time, all_wall_times[0] if all_wall_times else t0_wall)
                edt_req = _parse_human_time(end_time,   all_wall_times[-1] if all_wall_times else t0_wall)
                if edt_req < sdt_req: sdt_req, edt_req = edt_req, sdt_req
                req_axis = _build_axis_by_range(sdt_req, edt_req)
                return _pack_out("gate_fail", rack_id_fmt, req_axis, -1.0, f"Gate failure: {repr(e)}")
            except Exception as e2:
                return {"status": "gate_fail", "rack_id": rack_id_fmt, "time_axis": [], "soc_estimated": [], "reason": f"Gate failure + time resolve error: {repr(e2)}"}

        # ========= 将人类时间对齐到数据网格 / Align human times to dataset grid =========
        try:
            # 数据集墙钟基准；Base wall-clock
            try:
                t0_wall = _dt.datetime.strptime(tagTS, "%Y%m%d%H%M%S")
            except Exception:
                t0_wall = _dt.datetime(1970,1,1,0,0,0)

            all_wall_times = [t0_wall + _dt.timedelta(seconds=float(k)*deltaT) for k in range(len(t_grid))]

            # 解析请求并对齐；Parse & snap
            sdt_req = _parse_human_time(start_time, all_wall_times[0])
            edt_req = _parse_human_time(end_time,   all_wall_times[-1])
            if edt_req < sdt_req:
                sdt_req, edt_req = edt_req, sdt_req

            sdt = _snap_to_grid(sdt_req, all_wall_times[0])
            edt = _snap_to_grid(edt_req, all_wall_times[0])

            # 请求时间轴（即使只有部分交集）/ Requested axis
            req_wall_axis = _build_axis_by_range(sdt, edt)

            # 交集 mask / Intersection
            mask = np.array([(wt >= sdt) and (wt <= edt) for wt in all_wall_times], dtype=bool)
            if not np.any(mask):
                # 无交集：非 Gate 错误，按 0 输出
                return _pack_out("fail", rack_id_fmt, req_wall_axis, 0.0, "No overlap with available data")

            # 片段数据 / Segment
            I_seg    = (-I_grid[mask]).astype(float)     # quick-PE expects u=-I
            V_seg    = ( V_grid[mask]).astype(float)
            SOCp_seg = (SOCp_grid[mask]).astype(float)
            wall_seg = [wt for (wt, m) in zip(all_wall_times, mask) if m]

            # 实际 SOC（0~1）与初值；Actual SOC (0~1) & initial
            SOC_real_seg = SOCp_seg/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real
            SOC0_live    = float(SOC_real_seg[0])

        except Exception as e:
            # 对齐失败：非 Gate 错误，按 0 输出
            return _pack_out("fail", rack_id_fmt, req_wall_axis if 'req_wall_axis' in locals() else [], 0.0, f"Time align failure: {repr(e)}")

        # ========= 参数来源 + 早检阈值 / Params source + early-fit threshold =========
        search_dirs = [
        dataFolder,
        os.path.dirname(__file__),
        os.path.join(os.path.dirname(__file__), "model"),
    ]
        if model_dir:
            model_dir_abs = os.path.abspath(os.path.expanduser(model_dir))
            # 放到最前，使传入目录优先生效
            search_dirs.insert(0, model_dir_abs)

        pe_res = load_pe_params(rack_id_fmt, search_dirs)

        if pe_res is None:
            paramVec = np.array([0.30, 0.20, 0.20, 1500.0, 1500.0, Q_nom, 0.0, 0.0, 0.0], float)
        else:
            paramVec = pe_res.param_vec.astype(float)

        # 早期窗口 ~ 前 1/3（带上下限）/ Early window ~ 1/3 (with min/max clamps)
        Nest = int(np.floor(early_use_ratio * len(V_seg)))
        Nest = min(Nest, early_max_pts)
        Nest = max(Nest, early_min_pts)
        Nest = min(Nest, len(V_seg)-1)
        Nest = max(Nest, 10)

        # 早检拟合度 / Early-fit
        try:
            y_sim0 = simulate_voltage(
                param_vec=paramVec, u=I_seg[:Nest], dt=deltaT, soc0=SOC0_live,
                ocv_interp=ocv_interp, pack_series=pack_series,
                soc_min=SOC_min_real, soc_max=SOC_max_real
            )
            L = min(len(V_seg[:Nest]), len(y_sim0))
            num = np.linalg.norm(V_seg[:L] - y_sim0[:L])
            den = max(1e-9, np.linalg.norm(V_seg[:L] - np.mean(V_seg[:L])))
            fitpct = max(0.0, min(100.0, 100.0*(1 - num/den)))
        except Exception as e:
            # 预检失败：非 Gate 错误，按 0 输出
            return _pack_out("fail", rack_id_fmt, wall_seg, 0.0, f"Early-check failure: {repr(e)}")

        # 阈值不达标 ⇒ 仅用该段前 ~1/3 重辨识一次 / Below threshold ⇒ single quick-PE on first ~1/3
        if fitpct < fit_threshold:
            try:
                paramVec = quick_identify_params(
                    u=I_seg[:Nest], y=V_seg[:Nest], dt=deltaT, soc0=SOC0_live,
                    ocv_interp=ocv_interp, pack_series=pack_series,
                    init_params=paramVec, soc_min=SOC_min_real, soc_max=SOC_max_real, max_nfev=200
                )
            except Exception as e:
                # 重辨识失败：非 Gate 错误，按 0 输出
                return _pack_out("fail", rack_id_fmt, wall_seg, 0.0, f"Quick-PE failure: {repr(e)}")

        # ========= EKF 全段运行 / EKF over the whole segment =========
        try:
            out = run_ekf(
                I_grid[mask], V_grid[mask], SOCp_grid[mask], paramVec, deltaT, ocv_interp,
                SOC_min_real, SOC_max_real, I_idle_thresh,
                S_low, S_high, slope_floor, pack_series
            )
        except Exception as e:
            # EKF 失败：非 Gate 错误，按 0 输出
            return _pack_out("fail", rack_id_fmt, wall_seg, 0.0, f"EKF failure: {repr(e)}")

        # 用户口径 SOC（%）/ User-facing SOC (%)
        soc_fused  = out["soc_fused"]
        user_soc_est = (soc_fused - SOC_min_real) / (SOC_max_real - SOC_min_real) * 100.0
        user_soc_est = np.clip(user_soc_est, 0.0, 100.0)

        # ========= 最后统一绘图（如需要）/ Final plotting if requested =========
        if show_plots:
            try:
                # SOC
                fig1 = plt.figure(figsize=(8, 4))
                ax1  = fig1.add_subplot(111)
                ax1.plot(wall_seg, user_soc_est, "-", lw=1.2, label="est→user SOC (%)")
                ax1.set_ylabel("SOC (%)"); ax1.grid(True); ax1.legend()
                ax1.set_title(f"Rack {rack_id_fmt} | SOC estimated (segment)")
                ax1.set_xlabel("Time"); fig1.autofmt_xdate()

                # Voltage
                fig2 = plt.figure(figsize=(7, 3))
                ax2  = fig2.add_subplot(111)
                ax2.plot(wall_seg, V_grid[mask], "-", lw=1.1, label="Measured")
                ax2.plot(wall_seg, out["v_ekf"],  "-", lw=1.1, label="Pred (EKF)")
                ax2.set_ylabel("Voltage (V)"); ax2.grid(True); ax2.legend()
                ax2.set_xlabel("Time"); fig2.autofmt_xdate()

                # 统一展示 / Show once at end
                plt.show = _orig_show
                plt.show()
            except Exception:
                pass

        # ========= 成功返回 / Success =========
        return {
            "status": "ok",
            "rack_id": rack_id_fmt,
            "time_axis": [wt.strftime("%Y-%m-%d %H:%M:%S") for wt in wall_seg],
            "soc_estimated": [float(x) for x in user_soc_est],
            "reason": None
        }

    finally:
        # 恢复 show / Restore show
        plt.show = _orig_show


# ========== 简洁主入口（按你的输入习惯）/ Minimal CLI entrypoint ==========

def _cli():
    """
    简洁命令行入口：只要 Rack + 时间段 + 是否画图
    Minimal CLI: rack + time range + show plots
    """
    import json
    ap = argparse.ArgumentParser(description="SOC estimation over a time period (function-style)")
    ap.add_argument("--rack", required=True, help="Rack ID, e.g., 09 or 9")
    ap.add_argument("--start", default=None, help="Start time (human-readable, e.g., 2025-08-01 00:00:00)")
    ap.add_argument("--end",   default=None, help="End time   (human-readable, e.g., 2025-08-05 23:59:59)")
    ap.add_argument("--show",  action="store_true", help="Show plots at the end")
    ap.add_argument("--data-dir", default=None, help="Directory containing the dataset")
    ap.add_argument("--model-dir", default=None, help="Directory to search PE parameter files")

    
    args = ap.parse_args()

    res = run_soc_period(
        rack_id=args.rack,
        start_time=args.start,
        end_time=args.end,
        show_plots=args.show,
        dataset_dir=args.data_dir,   # <— 新增
        model_dir=args.model_dir,    # <— 新增
    )

    # 输出 JSON，便于后续 API/脚本处理 / Print JSON for downstream use
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
