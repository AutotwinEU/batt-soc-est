# soc_est.py
# -*- coding: utf-8 -*-
"""
SOC Estimation (epoch-ms time, string rack_id, Neo4j source)

- 数据来源：Neo4j 中该 rack_id 的 current/voltage/soc 点位（按时间戳交集对齐）→ Gates → 60s 重采样
- 与请求窗口做交集；有就只算有的部分，完全无交集则 status='no_interval'

- OCV：model/OCV_charge.csv & model/OCV_discharge.csv
- 参数：model/PE_result_Rack_<rack_id>.JASON（若 quick-PE 触发则覆盖写回）

- 任一计算错误（Gate/OCV/PE/EKF…）⇒ 输出仍生成，但 SOC 全为 -1（长度=裁剪后的请求子段）
- 早检不达阈值：段首 ~1/3 快速 PE（一次），贯穿全段；可选保存注册表 JSON（ask_save_pe + save_pe_path）
"""

import os, argparse, json
import neo4j
import numpy as np
import datetime as _dt
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import matplotlib.pyplot as plt

# ===== 相对导入 =====
from .ekf_core import run_ekf, OCVInterp
from .pe import quick_identify_params, simulate_voltage
from .io_utils import resample_60s
from .gates import gate1_first_ts_bar, gate2_cc_hard, gate2b_strict_ocv_rest

# ===== 常量 =====
Q_nom           = 105.0
SOC_min_real    = 0.10
SOC_max_real    = 0.94
deltaT          = 60.0
deltaT_ms       = int(deltaT * 1000)
pack_series     = 16 * 15
I_idle_thresh   = 0.001
rest_min_len    = 50
dV_flat_thr     = 0.05
tol_first_ms    = 5
seg_mae_thr     = 0.08
min_points_PE   = 3000
tol_OCV_pack    = 10.0
min_points_keep = 3000

early_use_ratio = 1/3
early_max_pts   = 3600
early_min_pts   = 600
fit_threshold   = 80.0

S_low  = 0.10
S_high = 0.20
slope_floor = 1e-5


# ---------- 工具 ----------
def build_ms_axis_by_range(start_ms: int, end_ms: int) -> List[int]:
    if end_ms < start_ms:
        start_ms, end_ms = end_ms, start_ms
    n = max(0, (end_ms - start_ms) // deltaT_ms)
    return [int(start_ms + k * deltaT_ms) for k in range(n + 1)]

def ms_to_datetime(ms: int) -> _dt.datetime:
    return _dt.datetime.utcfromtimestamp(ms / 1000.0)

def pack_out(status: str, rack_code: str, time_axis_ms: List[int], soc_value: float, reason: str) -> Dict[str, Any]:
    return {
        "status": status,
        "rack_id": rack_code,
        "time_axis": [int(x) for x in time_axis_ms],
        "soc_estimated": [float(soc_value)] * len(time_axis_ms),
        "reason": reason
    }


# ---------- Neo4j 读取 ----------
neo4j_uri = "neo4j://localhost:7687" if os.getenv("NEO4J_URI") is None else os.getenv("NEO4J_URI")
neo4j_username = "neo4j" if os.getenv("NEO4J_USERNAME") is None else os.getenv("NEO4J_USERNAME")
neo4j_password = "12345678" if os.getenv("NEO4J_PASSWORD") is None else os.getenv("NEO4J_PASSWORD")
neo4j_database = "neo4j" if os.getenv("NEO4J_DATABASE") is None else os.getenv("NEO4J_DATABASE")

def get_rack_ids():
    neo4j_auth = (neo4j_username, neo4j_password)
    with neo4j.GraphDatabase.driver(neo4j_uri, auth=neo4j_auth).session(database=neo4j_database) as sess:
        rows = sess.execute_read(read_rack_ids)
    rack_ids = [row["rack_id"] for row in rows]
    return rack_ids

def read_rack_ids(tx: neo4j.ManagedTransaction):
    return tx.run(
        """
        MATCH (ts:TimeSeries)
        RETURN ts.rackId As rack_id
        ORDER BY rack_id
        """
    ).data()

def read_voltage_data(tx: neo4j.ManagedTransaction, rack_id: str, start_ms: int, end_ms: int):
    return tx.run(
        f"""
        MATCH (vr:VoltageReading)-[:POINT_OF]->(ts:TimeSeries)
        WHERE ts.rackId = '{rack_id}' AND {start_ms} <= vr.timestamp <= {end_ms}
        RETURN vr.timestamp AS timestamp, vr.value AS value
        ORDER BY timestamp
        """
    ).data()

def read_current_data(tx: neo4j.ManagedTransaction, rack_id: str, start_ms: int, end_ms: int):
    return tx.run(
        f"""
        MATCH (cr:CurrentReading)-[:POINT_OF]->(ts:TimeSeries)
        WHERE ts.rackId = '{rack_id}' AND {start_ms} <= cr.timestamp <= {end_ms}
        RETURN cr.timestamp AS timestamp, cr.value AS value
        ORDER BY timestamp
        """
    ).data()

def read_soc_data(tx: neo4j.ManagedTransaction, rack_id: str, start_ms: int, end_ms: int):
    return tx.run(
        f"""
        MATCH (sr:SoCReading)-[:POINT_OF]->(ts:TimeSeries)
        WHERE ts.rackId = '{rack_id}' AND {start_ms} <= sr.timestamp <= {end_ms}
        RETURN sr.timestamp AS timestamp, sr.value AS value
        ORDER BY timestamp
        """
    ).data()


def _align_triple_from_neo(cur_rows, vol_rows, soc_rows):
    """
    将三路点按共同时间戳对齐，返回：
    t_ms (float, 毫秒), I_meas(A), V_meas(V), SOC_pct(%)
    """
    if not (cur_rows and vol_rows and soc_rows):
        raise RuntimeError("Neo4j returned empty series for current/voltage/soc.")

    ts_i = np.array([r["timestamp"] for r in cur_rows], dtype=np.int64)
    ts_v = np.array([r["timestamp"] for r in vol_rows], dtype=np.int64)
    ts_s = np.array([r["timestamp"] for r in soc_rows], dtype=np.int64)

    # 取三者公共时间戳
    common = sorted(set(ts_i.tolist()) & set(ts_v.tolist()) & set(ts_s.tolist()))
    if len(common) < 10:
        raise RuntimeError("Too few common timestamps after aligning I/V/SOC.")

    common = np.array(common, dtype=np.int64)

    # 建立从时间戳到值的映射，提速查找
    map_i = {int(r["timestamp"]): float(r["value"]) for r in cur_rows}
    map_v = {int(r["timestamp"]): float(r["value"]) for r in vol_rows}
    map_s = {int(r["timestamp"]): float(r["value"]) for r in soc_rows}

    I_meas  = np.array([map_i[int(t)] for t in common], dtype=float)
    V_meas  = np.array([map_v[int(t)] for t in common], dtype=float)
    SOC_pct = np.array([map_s[int(t)] for t in common], dtype=float)

    # 构造 raw_t_ms3（与原 Gate1 接口一致：三列同一时间）
    t_ms3 = np.stack([common.astype(float), common.astype(float), common.astype(float)], axis=1)
    return t_ms3, I_meas, V_meas, SOC_pct


# --- 参数 & OCV：从 model 目录读取 ---
def _params_json_path(model_dir: str, rack_code: str) -> Path:
    base = f"PE_result_Rack_{rack_code}"
    for ext in (".JASON", ".json", ".JSON"):
        p = Path(model_dir) / f"{base}{ext}"
        if p.exists():
            return p
    return Path(model_dir) / f"{base}.JASON"

def load_params_from_model(model_dir: str, rack_code: str) -> np.ndarray:
    p = _params_json_path(model_dir, rack_code)
    try:
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            vec = data.get("param_vec")
            if isinstance(vec, list) and len(vec) >= 9:
                return np.array(vec[:9], float)
    except Exception:
        pass
    # 基本参数
    return np.array([0.30, 0.20, 0.20, 1500.0, 45700.0, Q_nom, -0.15, 1.10, -0.43], float)

def save_params_to_model(model_dir: str, rack_code: str, param_vec: np.ndarray, fitpct: float) -> str:
    p = _params_json_path(model_dir, rack_code)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "param_vec": [float(x) for x in np.asarray(param_vec, float).reshape(-1).tolist()],
        "Q_hat": float(param_vec[5]) if len(param_vec) > 5 else float(Q_nom),
        "fitpct": float(fitpct),
    }
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)

def load_ocv_from_model(model_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import pandas as pd
    ch = Path(model_dir) / "OCV_charge.csv"
    dc = Path(model_dir) / "OCV_discharge.csv"
    if not ch.exists() or not dc.exists():
        raise FileNotFoundError(f"OCV csv not found under {model_dir}")
    dfc = pd.read_csv(ch)
    dfd = pd.read_csv(dc)
    if dfc.shape[1] < 2 or dfd.shape[1] < 2:
        raise ValueError("OCV csv must have at least two columns (SOC, OCV)")
    soc_c, ocv_c = dfc.iloc[:, 0].to_numpy(float), dfc.iloc[:, 1].to_numpy(float)
    soc_d, ocv_d = dfd.iloc[:, 0].to_numpy(float), dfd.iloc[:, 1].to_numpy(float)
    if np.nanmax(soc_c) > 1.0: soc_c = soc_c / 100.0
    if np.nanmax(soc_d) > 1.0: soc_d = soc_d / 100.0
    return soc_c, ocv_c, soc_d, ocv_d


# ========== 核心函数 ==========
def run_soc_period(
    rack_id: str,
    start_ms: int = 0,
    end_ms: int = 0,
    model_dir: str = "models",
    show_plots: bool = False,
    save_plots_dir: Optional[str] = None,
    ask_save_pe: bool = False,
    save_pe_path: Optional[str] = None,
) -> Dict[str, Any]:

    rack_code = str(rack_id).strip()

    _orig_show = plt.show
    try:
        def _silent_show(*a, **k): return None
        plt.show = _silent_show
        plt.ioff()

        # 1) 窗口检查
        if (start_ms is None) or (end_ms is None):
            return {"status": "fail", "rack_id": rack_code, "time_axis": [], "soc_estimated": [], "reason": "start_ms/end_ms required."}
        req_s_ms = int(start_ms); req_e_ms = int(end_ms)
        if req_e_ms < req_s_ms:
            req_s_ms, req_e_ms = req_e_ms, req_s_ms

        # 2) OCV
        model_dir_abs = os.path.abspath(os.path.expanduser(model_dir)) if model_dir else os.path.join(os.path.dirname(__file__), "model")
        try:
            SOC_c, OCV_c, SOC_d, OCV_d = load_ocv_from_model(model_dir_abs)
            ocv_interp = OCVInterp(SOC_c, OCV_c, SOC_d, OCV_d)
        except Exception as e:
            req_axis_ms = build_ms_axis_by_range(req_s_ms, req_e_ms)
            return pack_out("fail", rack_code, req_axis_ms, -1.0, f"OCV load failure: {repr(e)}")

        # 3) Neo4j 读取并对齐
        with neo4j.GraphDatabase.driver(neo4j_uri, auth=(neo4j_username, neo4j_password)).session(database=neo4j_database) as sess:
            cur_rows = sess.execute_read(lambda tx: read_current_data(tx, rack_id, req_s_ms, req_e_ms))
            vol_rows = sess.execute_read(lambda tx: read_voltage_data(tx, rack_id, req_s_ms, req_e_ms))
            soc_rows = sess.execute_read(lambda tx: read_soc_data(tx, rack_id, req_s_ms, req_e_ms))

        try:
            raw_t_ms3, raw_I, raw_V, raw_SOCpct = _align_triple_from_neo(cur_rows, vol_rows, soc_rows)
        except Exception as e:
            req_axis_ms = build_ms_axis_by_range(req_s_ms, req_e_ms)
            return pack_out("gate_fail", rack_code, req_axis_ms, -1.0, f"Align failure: {repr(e)}")

        # 4) Gates + 重采样（与原逻辑一致）
        try:
            # Gate 1
            gate1_first_ts_bar(raw_t_ms3, rack_code, tol_first_ms=tol_first_ms, save=False)

            t_ms    = raw_t_ms3[:, 0]
            I_meas  = raw_I.astype(float)
            V_meas  = raw_V.astype(float)
            SOC_pct = raw_SOCpct.astype(float)
            SOC_real = SOC_pct/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real

            # Gate 2
            gate2_cc_hard(I_meas, SOC_real, t_ms, Q_nom, SOC_min_real, SOC_max_real,
                          thr=seg_mae_thr, rack_id=rack_code, save=False)

            # Gate 2b
            t_ms, raw_I, raw_V, SOC_real_all = gate2b_strict_ocv_rest(
                t_ms, I_meas, V_meas, SOC_real, ocv_interp,
                pack_series=pack_series,
                I_idle_thresh=I_idle_thresh, dV_flat_thr=dV_flat_thr, rest_min_len=rest_min_len,
                tol_OCV_pack=tol_OCV_pack, min_points_keep=min_points_keep,
                rack_id=rack_code, save=False
            )

            # 重采样到 60s
            t_grid, I_grid, V_grid, SOCp_grid = resample_60s(
                t_ms, raw_I, raw_V, SOCp=SOC_real_all * 100.0, deltaT=deltaT
            )

            # 绝对毫秒时间轴：以对齐后首个时间戳为基准
            base_ms = int(t_ms[0])
            all_ms_axis = [int(base_ms + k * deltaT_ms) for k in range(len(t_grid))]

            if len(all_ms_axis) < min_points_PE:
                req_axis_ms = build_ms_axis_by_range(req_s_ms, req_e_ms)
                return pack_out("gate_fail", rack_code, req_axis_ms, -1.0, f"Too few points after resampling: {len(all_ms_axis)} < {min_points_PE}")

            # 与请求窗口做交集
            data_start_ms = int(all_ms_axis[0])
            data_end_ms   = int(all_ms_axis[-1])
            clip_s_ms = max(req_s_ms, data_start_ms)
            clip_e_ms = min(req_e_ms, data_end_ms)
            if clip_e_ms < clip_s_ms:
                return {"status": "no_interval", "rack_id": rack_code, "time_axis": [], "soc_estimated": [], "reason": f"Requested [{req_s_ms},{req_e_ms}] not in [{data_start_ms},{data_end_ms}]."}

            req_axis_ms = build_ms_axis_by_range(clip_s_ms, clip_e_ms)
            mask = np.array([(ms >= clip_s_ms) and (ms <= clip_e_ms) for ms in all_ms_axis], dtype=bool)
            if not np.any(mask):
                return {"status": "no_interval", "rack_id": rack_code, "time_axis": [], "soc_estimated": [], "reason": f"No usable points within [{clip_s_ms},{clip_e_ms}]."}

        except Exception as e:
            req_axis_ms = build_ms_axis_by_range(req_s_ms, req_e_ms)
            return pack_out("gate_fail", rack_code, req_axis_ms, -1.0, f"Gate/IO failure: {repr(e)}")

        # 5) 片段数据
        try:
            I_seg    = ( np.asarray(I_grid[mask], float)).astype(float)
            V_seg    = ( np.asarray(V_grid[mask], float)).astype(float)
            SOCp_seg = ( np.asarray(SOCp_grid[mask], float)).astype(float)
            axis_ms  = [all_ms_axis[i] for i, m in enumerate(mask) if m]

            SOC_real_seg = SOCp_seg/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real
            SOC0_live    = float(SOC_real_seg[0])
        except Exception as e:
            return pack_out("fail", rack_code, req_axis_ms, -1.0, f"Segment build failure: {repr(e)}")

        # 6) 参数 + 早检 + quick-PE（一次），并覆盖保存到 model
        pe_refit = False
        pe_saved_to = None
        try:
            paramVec = load_params_from_model(model_dir_abs, rack_code)
        except Exception:
            paramVec = np.array([0.30, 0.20, 0.20, 1500.0, 45700.0, Q_nom, -0.15, 1.10, -0.43], float)

        Nest = int(np.floor(early_use_ratio * len(V_seg)))
        Nest = min(Nest, early_max_pts)
        Nest = max(Nest, early_min_pts)
        Nest = min(Nest, len(V_seg)-1)
        Nest = max(Nest, 10)

        try:
            y_sim0 = simulate_voltage(
                param_vec=paramVec, u=-I_seg[:Nest], dt=deltaT, soc0=SOC0_live,
                ocv_interp=ocv_interp, pack_series=pack_series,
                soc_min=SOC_min_real, soc_max=SOC_max_real
            )
            L = min(len(V_seg[:Nest]), len(y_sim0))
            num = np.linalg.norm(V_seg[:L] - y_sim0[:L])
            den = max(1e-9, np.linalg.norm(V_seg[:L] - np.mean(V_seg[:L])))
            fitpct = max(0.0, min(100.0, 100.0*(1 - num/den)))
        except Exception as e:
            return pack_out("fail", rack_code, req_axis_ms, -1.0, f"Early-check failure: {repr(e)}")

        if fitpct < fit_threshold:
            try:
                paramVec = quick_identify_params(
                    u=-I_seg[:Nest], y=V_seg[:Nest], dt=deltaT, soc0=SOC0_live,
                    ocv_interp=ocv_interp, pack_series=pack_series,
                    init_params=paramVec, soc_min=SOC_min_real, soc_max=SOC_max_real, max_nfev=200
                )
                pe_refit = True
                try:
                    pe_saved_to = save_params_to_model(model_dir_abs, rack_code, paramVec, fitpct)
                    print(f"[PE] Saved new baseline for {rack_code} to {pe_saved_to}")
                except Exception as se:
                    print(f"[PE] Save failed: {repr(se)}")

                if ask_save_pe and save_pe_path:
                    try:
                        path = Path(save_pe_path)
                        reg = {}
                        if path.exists():
                            reg = json.loads(path.read_text(encoding="utf-8"))
                        reg[rack_code] = [float(x) for x in np.asarray(paramVec, float).reshape(-1).tolist()]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
                        print(f"[PE] Also wrote registry to {path}")
                    except Exception as se:
                        print(f"[PE] Registry save skipped/failed: {repr(se)}")

            except Exception as e:
                return pack_out("fail", rack_code, req_axis_ms, -1.0, f"Quick-PE failure: {repr(e)}")

        # 7) EKF
        try:
            out = run_ekf(
                np.asarray(I_seg, float),
                np.asarray(V_seg, float),
                np.asarray(SOCp_seg, float),
                np.asarray(paramVec, float),
                deltaT, ocv_interp,
                SOC_min_real, SOC_max_real, I_idle_thresh,
                S_low, S_high, slope_floor, pack_series
            )
        except Exception as e:
            return pack_out("fail", rack_code, req_axis_ms, -1.0, f"EKF failure: {repr(e)}")

        soc_fused  = out["soc_fused"]
        v_ekf_seg  = np.asarray(out["v_ekf"], float)   # EKF  predicted voltage (V)
        user_soc_est = (soc_fused - SOC_min_real) / (SOC_max_real - SOC_min_real) * 100.0
        user_soc_est = np.clip(user_soc_est, 0.0, 100.0)

        bms_soc_pct = np.asarray(SOCp_seg, float)
        bms_soc_pct = np.clip(bms_soc_pct, 0.0, 100.0)

        diff = user_soc_est - bms_soc_pct
        mae  = float(np.mean(np.abs(diff)))
        rmse = float(np.sqrt(np.mean(diff**2)))
        den  = float(np.sum((bms_soc_pct - np.mean(bms_soc_pct))**2))
        r2   = float(1.0 - (np.sum(diff**2) / (den if den > 1e-12 else 1e-12)))

        pe_params_used = [float(x) for x in np.asarray(paramVec, float).reshape(-1).tolist()]

        # 8) 绘图
        plot_paths = {}
        if show_plots or save_plots_dir:
            try:
                wall_times = [ms_to_datetime(ms) for ms in axis_ms]

                fig1 = plt.figure(figsize=(8, 4))
                ax1  = fig1.add_subplot(111)
                ax1.plot(wall_times, user_soc_est, "-",  lw=1.2, label="EKF est (user %)")
                ax1.plot(wall_times, bms_soc_pct,  "--", lw=1.0, label="BMS SOC (user %)")
                ax1.set_ylabel("SOC (%)"); ax1.grid(True); ax1.legend()
                ax1.set_title(f"Rack {rack_code} | SOC estimated vs BMS")
                ax1.set_xlabel("Time"); fig1.autofmt_xdate()

                fig2 = plt.figure(figsize=(7, 3))
                ax2  = fig2.add_subplot(111)
                ax2.plot(wall_times, V_seg,    "-", lw=1.1, label="Measured V")
                ax2.plot(wall_times, v_ekf_seg, "-", lw=1.1, label="Pred V (EKF)")
                ax2.set_ylabel("Voltage (V)"); ax2.grid(True); ax2.legend()
                ax2.set_xlabel("Time"); fig2.autofmt_xdate()

                if save_plots_dir:
                    outdir = Path(save_plots_dir)
                    outdir.mkdir(parents=True, exist_ok=True)
                    stamp = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                    p1 = outdir / f"soc_compare_{rack_code}_{stamp}.png"
                    p2 = outdir / f"voltage_compare_{rack_code}_{stamp}.png"
                    fig1.savefig(p1, dpi=150, bbox_inches="tight")
                    fig2.savefig(p2, dpi=150, bbox_inches="tight")
                    plot_paths["soc"] = str(p1)
                    plot_paths["voltage"] = str(p2)
                    print(f"[PLOT] saved:\n  {p1}\n  {p2}")

                if show_plots:
                    _orig_show()
                else:
                    plt.close(fig1); plt.close(fig2)
            except Exception as _e:
                print(f"[PLOT] skipped due to error: {_e}")

        return {
            "status": "ok",
            "rack_id": rack_code,
            "time_axis": [int(x) for x in axis_ms],
            "soc_estimated": [float(x) for x in user_soc_est],
            "bms_soc": [float(x) for x in bms_soc_pct],
            "v_pred_ekf": [float(x) for x in v_ekf_seg],   # 新增：EKF 预测电压序列
            "metrics": {"mae_pct": mae, "rmse_pct": rmse, "r2": r2},
            "pe_params_used": pe_params_used,
            "pe_refit": bool(pe_refit),
            "pe_fitpct_early": float(fitpct),
            "pe_saved_to": pe_saved_to,
            "plots": plot_paths or None,
            "reason": None
        }


    finally:
        plt.show = _orig_show


# ========== CLI ==========
def _cli():
    ap = argparse.ArgumentParser(description="SOC estimation (epoch-ms time, string rack_id via Neo4j)")
    ap.add_argument("--rack-id", required=True)
    ap.add_argument("--start-ms", type=int, required=True)
    ap.add_argument("--end-ms",   type=int, required=True)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--show-plots",  action="store_true")
    ap.add_argument("--save-plots-dir", default=None)
    ap.add_argument("--ask-save-pe", action="store_true")
    ap.add_argument("--save-pe-path", default=None)
    args = ap.parse_args()

    res = run_soc_period(
        rack_id=args.rack_id,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        model_dir=args.model_dir,
        show_plots=args.show,
        save_plots_dir=args.save_plots_dir,
        ask_save_pe=args.ask_save_pe,
        save_pe_path=args.save_pe_path,
    )
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
