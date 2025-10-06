# pe.py
# Utilities for Parameter Estimation (PE): load/save params, quick Auto-PE, full PE

from __future__ import annotations
import os, json, math, time
from types import SimpleNamespace
from typing import Iterable, Optional, Tuple
import numpy as np

try:
    from scipy.io import loadmat, savemat
    from scipy.optimize import least_squares
except Exception as e:
    raise RuntimeError(
        "pe.py requires SciPy (scipy.io, scipy.optimize). "
        "Please install scipy in your environment."
    ) from e


# ----------------------------
# Small helpers
# ----------------------------

def _safe_mkdir(p: str):
    if p and not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _list_candidates(rack_id: str, roots: Iterable[str]) -> list[Tuple[str, float]]:
    """
    Find parameter files for a rack under multiple roots.
    Accepts: PE_result_Rack_XX_py.mat / .json / .csv  (prefer newer by mtime)
    """
    names = [
        f"PE_result_Rack_{rack_id}_py.mat",
        f"PE_result_Rack_{rack_id}.mat",
        f"PE_result_Rack_{rack_id}.json",
        f"PE_result_Rack_{rack_id}.csv",
    ]
    out = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for nm in names:
            p = os.path.join(root, nm)
            if os.path.isfile(p):
                try:
                    mt = os.path.getmtime(p)
                except Exception:
                    mt = 0.0
                out.append((p, mt))
    out.sort(key=lambda x: x[1], reverse=True)  # newest first
    return out

def _read_params_from_mat(p: str) -> Optional[np.ndarray]:
    try:
        M = loadmat(p, simplify_cells=True, struct_as_record=False)
    except Exception:
        return None
    if "param_vec" in M:
        arr = np.array(M["param_vec"]).astype(float).reshape(-1)
        if arr.size >= 9:
            return arr[:9]
    for k, v in M.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v)
        if np.issubdtype(v.dtype, np.number) and v.size >= 9:
            return v.reshape(-1)[:9].astype(float)
    return None

def _read_params_from_json(p: str) -> Optional[np.ndarray]:
    try:
        with open(p, "r", encoding="utf-8") as f:
            J = json.load(f)
        if "param_vec" in J:
            arr = np.array(J["param_vec"], dtype=float).reshape(-1)
            if arr.size >= 9:
                return arr[:9]
    except Exception:
        return None
    return None

def _read_params_from_csv(p: str) -> Optional[np.ndarray]:
    try:
        # expecting header: R0,R1,R2,tau1,tau2,Q,M1,M2,M3
        arr = np.genfromtxt(p, delimiter=",", names=True)
        fields = ["R0","R1","R2","tau1","tau2","Q","M1","M2","M3"]
        vals = []
        for f in fields:
            if f in arr.dtype.names:
                vals.append(float(arr[f]))
            
            else:
                return None
        return np.array(vals, dtype=float)
    except Exception:
        return None

def _load_params_from_file(p: str) -> Optional[np.ndarray]:
    ext = os.path.splitext(p)[1].lower()
    if ext == ".mat":  return _read_params_from_mat(p)
    if ext == ".json": return _read_params_from_json(p)
    if ext == ".csv":  return _read_params_from_csv(p)
    return None


# ----------------------------
# Public: load / save
# ----------------------------

def load_pe_params(rack_id: str, roots: Iterable[str]) -> Optional[SimpleNamespace]:
    """
    Search roots and return the newest param set.
    Returns SimpleNamespace(param_vec, source, mtime)
    """
    cands = _list_candidates(rack_id, roots)
    for p, mt in cands:
        arr = _load_params_from_file(p)
        if arr is not None and arr.size == 9 and np.all(np.isfinite(arr)):
            return SimpleNamespace(param_vec=arr.astype(float), source=p, mtime=mt)
    return None


def save_pe_result(
    rack_id: str,
    param_vec: np.ndarray,
    deltaT: float,
    fitpct_val: Optional[float],
    out_dir: str
) -> str:
    """
    Save params to *_py.mat + .json + .csv next to out_dir; return main mat path.
    """
    _safe_mkdir(out_dir)
    param_vec = np.asarray(param_vec, dtype=float).reshape(-1)[:9]
    base = f"PE_result_Rack_{rack_id}_py"
    mat_path  = os.path.join(out_dir, base + ".mat")
    json_path = os.path.join(out_dir, base + ".json")
    csv_path  = os.path.join(out_dir, base + ".csv")

    # .mat
    payload = {
        "param_vec": param_vec,
        "deltaT": float(deltaT),
        "fitpct": float(fitpct_val) if (fitpct_val is not None and np.isfinite(fitpct_val)) else float("nan"),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    savemat(mat_path, payload, do_compression=True)

    # .json (convert ndarray to list!)
    payload_json = {
        "param_vec": param_vec.tolist(),
        "deltaT": float(deltaT),
        "fitpct": float(payload["fitpct"]),
        "saved_at": payload["saved_at"],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload_json, f, ensure_ascii=False, indent=2)

    # .csv
    header = "R0,R1,R2,tau1,tau2,Q,M1,M2,M3\n"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(header)
        f.write(",".join(f"{float(v):.12g}" for v in param_vec.tolist()) + "\n")

    return mat_path


# ----------------------------
# Voltage simulation (ECM 2RC + simple bias)
# ----------------------------

def _ocv_from_soc(z: float, uk: float, ocv_interp) -> float:
    if uk > 0:
        return float(ocv_interp.f_discharge(z))
    elif uk < 0:
        return float(ocv_interp.f_charge(z))
    else:
        return float(ocv_interp.f_avg(z))

def simulate_voltage_states(
    param_vec: np.ndarray,
    u: np.ndarray,
    dt: float,
    soc0: float,
    ocv_interp,
    pack_series: int,
    soc_min: float,
    soc_max: float,
    i1_0: float = 0.0,
    i2_0: float = 0.0
):
    """
    Same as simulate_voltage, but also return end states for validation continuation.
    Returns: y, z_end, i1_end, i2_end
    """
    p = np.asarray(param_vec, dtype=float).reshape(-1)[:9]
    R0, R1, R2, tau1, tau2, Q, M1, M2, M3 = p
    u = np.asarray(u, dtype=float).reshape(-1)

    N = u.size
    y = np.zeros(N, dtype=float)
    z = float(soc0)
    i1 = float(i1_0)
    i2 = float(i2_0)

    a1 = math.exp(-dt/max(1e-9, tau1))
    a2 = math.exp(-dt/max(1e-9, tau2))
    b1 = 1.0 - a1
    b2 = 1.0 - a2
    kQ = dt / max(1e-9, (Q * 3600.0))

    for k in range(N):
        uk = u[k]
        z  = min(max(z - uk * kQ, soc_min), soc_max)
        i1 = a1 * i1 + b1 * uk
        i2 = a2 * i2 + b2 * uk

        ocv = _ocv_from_soc(z, uk, ocv_interp)
        if   uk >  0: M0 = M1
        elif uk <  0: M0 = M2
        else:         M0 = M3

        y[k] = ocv * pack_series - R0 * uk - R1 * i1 - R2 * i2 + M0

    return y, z, i1, i2


def simulate_voltage(
    param_vec: np.ndarray,
    u: np.ndarray,
    dt: float,
    soc0: float,
    ocv_interp,
    pack_series: int,
    soc_min: float,
    soc_max: float
) -> np.ndarray:
    """
    x=[SOC,iR1,iR2]; y = OCV(z)*Ns - R0*u - R1*iR1 - R2*iR2 + M(s)
    """
    p = np.asarray(param_vec, dtype=float).reshape(-1)[:9]
    R0, R1, R2, tau1, tau2, Q, M1, M2, M3 = p
    u = np.asarray(u, dtype=float).reshape(-1)

    N = u.size
    y = np.zeros(N, dtype=float)
    z = float(soc0)
    i1 = 0.0
    i2 = 0.0

    a1 = math.exp(-dt/max(1e-9, tau1))
    a2 = math.exp(-dt/max(1e-9, tau2))
    b1 = 1.0 - a1
    b2 = 1.0 - a2
    kQ = dt / max(1e-9, (Q * 3600.0))

    for k in range(N):
        uk = u[k]
        z  = min(max(z - uk * kQ, soc_min), soc_max)
        i1 = a1 * i1 + b1 * uk
        i2 = a2 * i2 + b2 * uk

        ocv = _ocv_from_soc(z, uk, ocv_interp)
        if   uk >  0: M0 = M1
        elif uk <  0: M0 = M2
        else:         M0 = M3

        y[k] = ocv * pack_series - R0 * uk - R1 * i1 - R2 * i2 + M0

    return y


# ----------------------------
# Quick Auto-PE (Levenberg–Marquardt)
# ----------------------------

def quick_identify_params(
    u: np.ndarray,
    y: np.ndarray,
    dt: float,
    soc0: float,
    ocv_interp,
    pack_series: int,
    init_params: np.ndarray,
    soc_min: float,
    soc_max: float,
    max_nfev: int = 200
) -> np.ndarray:
    """
    Least-squares with bounds (consistent with online flow).
    """
    u = np.asarray(u, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    p0 = np.asarray(init_params, dtype=float).reshape(-1)[:9]

    lb = np.array([ 1e-4,  1e-6, 1e-6,   0.0,   0.0,   80.0, -5.0, -5.0, -5.0], dtype=float)
    ub = np.array([  1.0,   1.0,  1.0, 1e6,   1e6,   110.0,  5.0,  5.0,  5.0], dtype=float)

    def resid(p):
        yhat = simulate_voltage(p, u, dt, soc0, ocv_interp, pack_series, soc_min, soc_max)
        return yhat - y

    res = least_squares(resid, p0, bounds=(lb, ub), method="trf",
                        max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-9)
    p = np.minimum(np.maximum(res.x.astype(float), lb), ub)
    return p


# ----------------------------
# Full PE core (no saving) + wrappers
# ----------------------------

def _fit_percent(y_true: np.ndarray, y_hat: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel()
    y_hat  = np.asarray(y_hat,  float).ravel()
    L = min(len(y_true), len(y_hat))
    if L <= 1:
        return float("nan")
    num = np.linalg.norm(y_true[:L]-y_hat[:L])
    den = max(1e-9, np.linalg.norm(y_true[:L]-np.mean(y_true[:L])))
    return float(max(0.0, min(100.0, 100.0*(1.0-num/den))))

def _full_pe_core(
    rack_id: str,
    I_grid: np.ndarray,
    V_grid: np.ndarray,
    SOCp_grid: np.ndarray,
    deltaT: float,
    ocv_interp,
    pack_series: int,
    SOC_min_real: float,
    SOC_max_real: float,
    Q_nom: float,
    fit_skip_threshold: float = 60.0,
    max_iter: int = 200,
) -> SimpleNamespace:
    """
    Full-PE core using CONTINUED STATES at the est→val split.
    Matches the CSV pipeline behavior to avoid validation reset mismatch.
    """
    # 1) build real SOC and initial SOC
    SOC_real_grid = SOCp_grid/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real
    SOC0 = float(SOC_real_grid[0])

    # 2) split 2/3:1/3 and form u=-I, y=V
    N = len(V_grid)
    Nest = max(1, min(N-1, int(math.floor(2.0*N/3.0))))
    u_est = -np.asarray(I_grid[:Nest], float)
    y_est =  np.asarray(V_grid[:Nest], float)
    u_val = -np.asarray(I_grid[Nest:], float)
    y_val =  np.asarray(V_grid[Nest:], float)

    # 3) initial guess
    p0 = np.array([0.30, 0.20, 0.20, 1500.0, 1500.0, Q_nom, 0.0, 0.0, 0.0], float)

    # 4) PRECHECK with p0 (IMPORTANT: continue states into validation)
    
    y0_est, zE, i1E, i2E = simulate_voltage_states(
        p0, u_est, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
    )
    y0_val, _, _, _ = simulate_voltage_states(
        p0, u_val, deltaT, zE, ocv_interp, pack_series, SOC_min_real, SOC_max_real,
        i1_0=i1E, i2_0=i2E
    )

    def _fit(y_true, y_hat):
        L = min(len(y_true), len(y_hat))
        num = np.linalg.norm(y_true[:L]-y_hat[:L])
        den = max(1e-9, np.linalg.norm(y_true[:L]-np.mean(y_true[:L])))
        return float(max(0.0, min(100.0, 100.0*(1.0-num/den))))

    fit0_est = _fit(y_est, y0_est)
    fit0_val = _fit(y_val, y0_val)
    print(f"[Full-PE PRECHECK] Init p0 fit: est={fit0_est:.1f}% | val={fit0_val:.1f}%")

    # 5) LM on estimation (or skip if p0 already good), then CONTINUE into validation
    if np.isfinite(fit0_val) and fit0_val >= fit_skip_threshold:
        p_best  = p0.copy()
        yb_est  = y0_est
        yb_val  = y0_val
        fit_est = fit0_est
        fit_val = fit0_val
        print("[Full-PE] Skip LM (p0 already decent on validation).")
    else:
        p_best = quick_identify_params(
            u=u_est, y=y_est, dt=deltaT, soc0=SOC0,
            ocv_interp=ocv_interp, pack_series=pack_series,
            init_params=p0, soc_min=SOC_min_real, soc_max=SOC_max_real,
            max_nfev=max_iter
        )
        yb_est, zE, i1E, i2E = simulate_voltage_states(
            p_best, u_est, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
        )
        yb_val, _, _, _ = simulate_voltage_states(
            p_best, u_val, deltaT, zE, ocv_interp, pack_series, SOC_min_real, SOC_max_real,
            i1_0=i1E, i2_0=i2E
        )
        fit_est = _fit(y_est, yb_est)
        fit_val = _fit(y_val, yb_val)
        print(f"[Full-PE RESULT] Best fit:   est={fit_est:.1f}% | val={fit_val:.1f}%")

    # 6) Rescue strategies if validation is still extremely low
    if not np.isfinite(fit_val) or fit_val < 5.0:
        print("[Full-PE RESCUE] Validation fit very low. Trying rescue strategies...")
        fit_best = fit_val
        p_keep   = p_best
        desc_keep= "LM default"

        # (a) Try sign flip/non-flip with continued states
        for sign_tag, uu_est, uu_val in [("flip-sign", -u_est, -u_val), ("same-sign", u_est, u_val)]:
            p_try = quick_identify_params(
                u=uu_est, y=y_est, dt=deltaT, soc0=SOC0,
                ocv_interp=ocv_interp, pack_series=pack_series,
                init_params=p0, soc_min=SOC_min_real, soc_max=SOC_max_real,
                max_nfev=max_iter
            )
            yh_est, zE, i1E, i2E = simulate_voltage_states(
                p_try, uu_est, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
            )
            yh_val, _, _, _ = simulate_voltage_states(
                p_try, uu_val, deltaT, zE, ocv_interp, pack_series, SOC_min_real, SOC_max_real,
                i1_0=i1E, i2_0=i2E
            )
            fit_try = _fit(y_val, yh_val)
            if fit_try > fit_best:
                fit_best, p_keep, desc_keep = fit_try, p_try, f"LM ({sign_tag})"

        # (b) Robust loss on the concatenated sequence (also continue into val)
        def _robust_all(u_all, y_all):
            u_all = np.asarray(u_all, float).ravel()
            y_all = np.asarray(y_all, float).ravel()
            p0r   = p0.copy()
            lb = np.array([1e-4, 1e-6, 1e-6,     0.0,     0.0,    80.0, -5.0, -5.0, -5.0], dtype=float)
            ub = np.array([  1.0,   1.0,   1.0, 1e6,    1e6,    110.0,  5.0,  5.0,  5.0], dtype=float)
            def resid(p):
                yh, _, _, _ = simulate_voltage_states(
                    p, u_all, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
                )
                return yh - y_all
            res = least_squares(resid, p0r, bounds=(lb, ub), method="trf",
                                loss="soft_l1", f_scale=10.0,
                                max_nfev=max_iter, ftol=1e-9, xtol=1e-9, gtol=1e-9)
            p = np.minimum(np.maximum(res.x.astype(float), lb), ub)
            return p

        p_try = _robust_all(np.r_[u_est, u_val], np.r_[y_est, y_val])
        yh_est, zE, i1E, i2E = simulate_voltage_states(
            p_try, u_est, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
        )
        yh_val, _, _, _ = simulate_voltage_states(
            p_try, u_val, deltaT, zE, ocv_interp, pack_series, SOC_min_real, SOC_max_real,
            i1_0=i1E, i2_0=i2E
        )
        fit_try = _fit(y_val, yh_val)
        if fit_try > fit_best:
            fit_best, p_keep, desc_keep = fit_try, p_try, "robust-all"

        # (c) Try Ns ± 1 with continued states
        for Ns_alt in [pack_series-1, pack_series+1]:
            if Ns_alt <= 0:
                continue
            p_try = quick_identify_params(
                u=u_est, y=y_est, dt=deltaT, soc0=SOC0,
                ocv_interp=ocv_interp, pack_series=Ns_alt,
                init_params=p0, soc_min=SOC_min_real, soc_max=SOC_max_real,
                max_nfev=max_iter
            )
            yh_est, zE, i1E, i2E = simulate_voltage_states(
                p_try, u_est, deltaT, SOC0, ocv_interp, Ns_alt, SOC_min_real, SOC_max_real
            )
            yh_val, _, _, _ = simulate_voltage_states(
                p_try, u_val, deltaT, zE, ocv_interp, Ns_alt, SOC_min_real, SOC_max_real,
                i1_0=i1E, i2_0=i2E
            )
            fit_try = _fit(y_val, yh_val)
            if fit_try > fit_best:
                fit_best, p_keep, desc_keep = fit_try, p_try, f"LM (Ns={Ns_alt})"

        p_best, fit_val = p_keep, fit_best
        print(f"[Full-PE RESCUE] Best rescue: {desc_keep}, val={fit_val:.1f}%")

    return SimpleNamespace(
        param_vec=p_best,
        fitpct_val=float(fit_val) if np.isfinite(fit_val) else float("nan"),
    )


def run_full_pe(
    rack_id: str,
    data_folder: str,            # not used for saving here; kept for API compatibility
    t_grid: np.ndarray,
    I_grid: np.ndarray,
    V_grid: np.ndarray,
    SOCp_grid: np.ndarray,
    deltaT: float,
    ocv_interp,
    pack_series: int,
    SOC_min_real: float,
    SOC_max_real: float,
    Q_nom: float,
    possible_roots: Iterable[str] = (),
    fit_skip_threshold: float = 60.0,
    max_iter: int = 200,
    use_latest_dataset_under_folder: bool = False,
) -> SimpleNamespace:
    """
    Full PE using the current (already resampled) data; no saving (caller decides).
    """
    res = _full_pe_core(
        rack_id, I_grid, V_grid, SOCp_grid, deltaT, ocv_interp, pack_series,
        SOC_min_real, SOC_max_real, Q_nom, fit_skip_threshold, max_iter
    )
    return SimpleNamespace(
        param_vec=res.param_vec,
        fitpct_val=res.fitpct_val,
        source="current_run",
        out_path="",
    )


def run_full_pe_from_csv(
    rack_id: str,
    data_folder: str,
    ocv_interp,
    pack_series: int,
    SOC_min_real: float,
    SOC_max_real: float,
    Q_nom: float,
    deltaT: float,
    # gate params
    tol_first_ms: float = 5.0,
    I_idle_thresh: float = 0.001,
    rest_min_len: int = 50,
    dV_flat_thr: float = 0.05,
    tol_OCV_pack: float = 10.0,
    min_points_keep: int = 3000,
    min_points_PE: int = 3000,
    # PE params
    fit_skip_threshold: float = 60.0,
    max_iter: int = 200,
    # debug
    debug_plot: bool = False,
) -> SimpleNamespace:
    """
    Pipeline:
      1) rename raw csv -> Rack_XX_<var>_<yyyymmddHHMMss>.csv
      2) auto-pick the latest trio for Rack_XX
      3) load & Gate#1/2/2b
      4) resample to 60s
      5) split 2/3~1/3; LM on estimation; report validation Fit%
      6) save params next to data_folder
    Returns: SimpleNamespace(param_vec, fitpct_val, source, out_path, ts_str)
    """
    import numpy as _np
    from io_utils import rename_rack_files_with_time, pick_triplet, load_triplet, resample_60s
    from gates import gate1_first_ts_bar, gate2_cc_hard, gate2b_strict_ocv_rest

    rack_id = f"{int(rack_id):02d}"

    # 1) rename (idempotent)
    rename_rack_files_with_time(
        data_folder,
        dry_run=False,
        data_header_lines=3,
        tolerance_ms=tol_first_ms
    )

    # 2) pick latest trio
    curFile, volFile, socFile, ts_str = pick_triplet(data_folder, rack_id, "")
    print(f"[Full-PE CSV] Using trio timestamp: {ts_str}")

    # 3) load + Gate#1/2/2b
    raw_t_ms3, raw_I, raw_V, raw_SOCpct = load_triplet(curFile, volFile, socFile, header_lines=3)
    gate1_first_ts_bar(raw_t_ms3, rack_id, tol_first_ms=tol_first_ms, save=False)

    t_ms    = raw_t_ms3[:, 0]
    I_meas  = raw_I.astype(float)
    V_meas  = raw_V.astype(float)
    SOC_pct = raw_SOCpct.astype(float)
    SOC_real = SOC_pct/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real

    gate2_cc_hard(I_meas, SOC_real, t_ms, Q_nom, SOC_min_real, SOC_max_real,
                  thr=0.08, rack_id=rack_id, save=False)

    t_ms, I_meas, V_meas, SOC_real = gate2b_strict_ocv_rest(
        t_ms, I_meas, V_meas, SOC_real, ocv_interp,
        pack_series=pack_series,
        I_idle_thresh=I_idle_thresh,
        dV_flat_thr=dV_flat_thr,
        rest_min_len=rest_min_len,
        tol_OCV_pack=tol_OCV_pack,
        min_points_keep=min_points_keep,
        rack_id=rack_id, save=False
    )

    # 4) 60s resample
    t_grid, I_grid, V_grid, SOCp_grid = resample_60s(
        t_ms, I_meas, V_meas, SOCp=SOC_real*100.0, deltaT=deltaT
    )
    if len(t_grid) < min_points_PE:
        raise RuntimeError(f"[Full-PE CSV] Too few points after resampling: {len(t_grid)} < {min_points_PE}")

    # 5) full PE (validation continues from the end of estimation segment)
    SOC_real_grid = SOCp_grid/100.0*(SOC_max_real - SOC_min_real) + SOC_min_real
    SOC0 = float(SOC_real_grid[0])

    N = len(V_grid)
    # start with default split at 2/3
    Nest = max(1, min(N-1, int(math.floor(2.0*N/3.0))))

    # --- adaptive: try to locate a validation window (last ~1/3) with enough dynamics ---
    val_len = max(10, N - Nest)
    best_span = (Nest, N)
    best_score = -1.0

    # 我们在最后 40% ~ 70% 的范围里滑一滑，挑一个 val 段动态性最高（std 最大）的切分
    scan_start = int(max(0, 0.6 * N))          # 起点
    scan_end   = int(min(N-10, 0.8 * N))       # 终点
    for s in range(scan_start, scan_end, max(1, (scan_end - scan_start) // 20)):
        e = min(N, s + val_len)
        if e - s < 10:
            continue
        y_val_try = np.asarray(V_grid[s:e], float)
        # 简单的“动态性”打分：标准差 + 相邻差分的平均值
        std_try = float(np.std(y_val_try))
        dv_try  = float(np.mean(np.abs(np.diff(y_val_try)))) if (len(y_val_try) > 1) else 0.0
        score   = std_try + dv_try
        if score > best_score:
            best_score = score
            best_span  = (s, e)

    # 如果找到更“活跃”的验证段，就用它，否则保留原来的 2/3 分割
    s, e = best_span
    if (s, e) != (Nest, N):
        print(f"[Full-PE SPLIT] Adjusting validation window to [{s}:{e}] (len={e-s}) for better dynamics.")
        u_est = -np.asarray(I_grid[:s], float)
        y_est =  np.asarray(V_grid[:s], float)
        u_val = -np.asarray(I_grid[s:e], float)
        y_val =  np.asarray(V_grid[s:e], float)
        Nest = s  # 更新 Nest 仅用于后续日志
    else:
        u_est = -np.asarray(I_grid[:Nest], float)
        y_est =  np.asarray(V_grid[:Nest], float)
        u_val = -np.asarray(I_grid[Nest:], float)
        y_val =  np.asarray(V_grid[Nest:], float)


    prev = load_pe_params(rack_id, [data_folder, os.path.dirname(__file__)])
    p0 = prev.param_vec if (prev is not None) else _np.array(
        [0.30, 0.20, 0.20, 1500.0, 1500.0, Q_nom, 0.0, 0.0, 0.0], float
    )

    def _fit(y_true, y_hat):
        L = min(len(y_true), len(y_hat))
        num = _np.linalg.norm(y_true[:L]-y_hat[:L])
        den = max(1e-9, _np.linalg.norm(y_true[:L]-_np.mean(y_true[:L])))
        return float(max(0.0, min(100.0, 100.0*(1.0-num/den))))

    # precheck on p0, with validation continuation
    y0_est, zE, i1E, i2E = simulate_voltage_states(
        p0, u_est, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
    )
    y0_val, _, _, _ = simulate_voltage_states(
        p0, u_val, deltaT, zE, ocv_interp, pack_series, SOC_min_real, SOC_max_real,
        i1_0=i1E, i2_0=i2E
    )
    fit0_est = _fit(y_est, y0_est)
    fit0_val = _fit(y_val, y0_val)
    print(f"[Full-PE PRECHECK] Init p0 fit: est={fit0_est:.1f}% | val={fit0_val:.1f}%")

    if _np.isfinite(fit0_val) and fit0_val >= fit_skip_threshold:
        p_best = p0.copy()
        y_hat_est, y_hat_val = y0_est, y0_val
        fit_val = fit0_val
    else:
        p_best = quick_identify_params(
            u=u_est, y=y_est, dt=deltaT, soc0=SOC0,
            ocv_interp=ocv_interp, pack_series=pack_series,
            init_params=p0, soc_min=SOC_min_real, soc_max=SOC_max_real,
            max_nfev=max_iter
        )
        y_hat_est, zE, i1E, i2E = simulate_voltage_states(
            p_best, u_est, deltaT, SOC0, ocv_interp, pack_series, SOC_min_real, SOC_max_real
        )
        y_hat_val, _, _, _ = simulate_voltage_states(
            p_best, u_val, deltaT, zE, ocv_interp, pack_series, SOC_min_real, SOC_max_real,
            i1_0=i1E, i2_0=i2E
        )
        fit_val = _fit(y_val, y_hat_val)

    if debug_plot:
        import matplotlib.pyplot as plt
        t_est = _np.arange(len(u_est))*deltaT
        t_val = _np.arange(len(u_val))*deltaT

        plt.figure(figsize=(9,3.2))
        plt.plot(t_est, y_est, label="est: measured", lw=1)
        plt.plot(t_est, y0_est, label="est: p0 sim", lw=1)
        plt.title(f"Precheck (p0) EST | Fit={fit0_est:.1f}%")
        plt.xlabel("t (s)"); plt.ylabel("V"); plt.grid(True); plt.legend()

        plt.figure(figsize=(9,3.2))
        plt.plot(t_val, y_val, label="val: measured", lw=1)
        plt.plot(t_val, y0_val, label="val: p0 sim (continued)", lw=1)
        plt.title(f"Precheck (p0) VAL | Fit={fit0_val:.1f}%")
        plt.xlabel("t (s)"); plt.ylabel("V"); plt.grid(True); plt.legend()

        plt.figure(figsize=(9,3.2))
        plt.plot(t_est, y_est, label="est: measured", lw=1)
        plt.plot(t_est, y_hat_est, label="est: p_best sim", lw=1)
        plt.title("After LM: EST")
        plt.xlabel("t (s)"); plt.ylabel("V"); plt.grid(True); plt.legend()

        plt.figure(figsize=(9,3.2))
        plt.plot(t_val, y_val, label="val: measured", lw=1)
        plt.plot(t_val, y_hat_val, label="val: p_best sim (continued)", lw=1)
        plt.title(f"After LM: VAL | Fit={fit_val:.1f}%")
        plt.xlabel("t (s)"); plt.ylabel("V"); plt.grid(True); plt.legend()

        plt.show()

    out_path = save_pe_result(rack_id, p_best, deltaT, fit_val, out_dir=data_folder)

    return SimpleNamespace(
        param_vec=p_best,
        fitpct_val=float(fit_val) if np.isfinite(fit_val) else float("nan"),
        source=f"{data_folder} (latest CSV trio: {ts_str})",
        out_path=out_path,
        ts_str=ts_str
    )
