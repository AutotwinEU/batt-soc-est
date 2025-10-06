# gates.py
import numpy as np
import matplotlib.pyplot as plt

def gate1_first_ts_bar(raw_t_ms, rack_id, tol_first_ms=5, save=False, outdir="figs"):
    first_ms = np.array([raw_t_ms[0,0], raw_t_ms[0,1], raw_t_ms[0,2]], float)
    dtBar = first_ms - np.min(first_ms)
    plt.figure(figsize=(5,3))
    plt.bar(["current","voltage","soc"], dtBar)
    plt.axhline(tol_first_ms, ls="--", label="Tolerance")
    plt.grid(True, axis="y")
    plt.title(f"Gate#1 First-ts Δt (ms), mismatch={dtBar.max()-dtBar.min():.3f} ms")
    plt.legend()
    if save:
        import os
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/gate1_ts_rack{rack_id}.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if np.any(np.abs(first_ms - np.median(first_ms)) > tol_first_ms):
        raise RuntimeError(f"Gate#1 FAILED: first timestamps mismatch > {tol_first_ms} ms.")
    print(f"[Gate#1 PASS] First timestamps aligned within {tol_first_ms} ms.")

def gate2_cc_hard(I_meas, SOC_real, t_ms, Q_nom, SOC_min_real, SOC_max_real,
                  thr=0.08, rack_id="??", save=False, outdir="figs"):
    t_sec = (t_ms - t_ms[0])/1000.0
    dt_vec = np.r_[np.diff(t_sec), np.median(np.diff(t_sec))]
    u_dis  = -I_meas

    soc_cc = np.zeros_like(SOC_real)
    soc_cc[0] = SOC_real[0]
    for k in range(1,len(SOC_real)):
        soc_cc[k] = soc_cc[k-1] - u_dis[k-1]*dt_vec[k-1]/(Q_nom*3600.0)
    soc_cc = np.clip(soc_cc, SOC_min_real, SOC_max_real)

    err_point = np.abs(soc_cc - SOC_real)
    first_bad = np.flatnonzero(err_point > thr)
    first_bad = int(first_bad[0]) if first_bad.size>0 else None

    plt.figure(figsize=(7,3))
    plt.plot(t_sec, SOC_real, "k--", lw=1.2, label="SOC_raw (real)")
    plt.plot(t_sec, soc_cc,   "r-",  lw=1.2, label="SOC_cc from SOC0")
    if first_bad is not None:
        plt.axvline(t_sec[first_bad], color="m", ls="--")
        ttl = f"Gate#2 HARD: first violation @ idx={first_bad} (t={t_sec[first_bad]:.0f}s), thr={thr:.4f}"
    else:
        ttl = f"Gate#2 HARD: PASS | MAE={np.mean(err_point):.4f} (thr={thr:.4f})"
    plt.title(ttl)
    plt.xlabel("Time (s)"); plt.ylabel("SOC (real)"); plt.grid(True); plt.legend()
    if save:
        import os
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/gate2_cc_rack{rack_id}.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    if first_bad is not None:
        raise RuntimeError(f"Gate #2 FAILED (hard gate): |SOC_cc - SOC_raw| exceeded {thr:.4f} at idx={first_bad} (t={t_sec[first_bad]:.0f}s).")
    print(f"[Gate#2 PASS] Single-pass CC hard gate: All points within {thr:.4f} (real SOC units).")

def gate2b_strict_ocv_rest(t_ms, raw_I, raw_V, SOC_real_all,
                           ocv_interp, pack_series=16*15,
                           I_idle_thresh=0.001, dV_flat_thr=0.05, rest_min_len=50,
                           tol_OCV_pack=5.0, min_points_keep=3000,
                           rack_id="??", save=False, outdir="figs"):
    t_sec = (t_ms - t_ms[0])/1000.0
    dV = np.r_[0.0, np.abs(np.diff(raw_V))]
    is_rest = (np.abs(raw_I)<=I_idle_thresh) & (dV<=dV_flat_thr)
    # 找到静置片段
    starts = np.flatnonzero(np.diff(np.r_[0,is_rest.astype(int)])==1)
    ends   = np.flatnonzero(np.diff(np.r_[is_rest.astype(int),0])==-1)
    ok = (ends - starts + 1) >= rest_min_len
    rs, re = starts[ok], ends[ok]
    if rs.size == 0:
        print("[Gate#2b] No valid rest segment. Skip strict OCV check.")
        return t_ms, raw_I, raw_V, SOC_real_all  # 不截断

    z_axis = np.linspace(0.10, 0.94, 400)
    V_ocv_axis = ocv_interp.f_avg(z_axis)*pack_series

    fig = plt.figure(figsize=(6,4))
    plt.plot(z_axis, V_ocv_axis, "k-", lw=1.2)
    plt.plot(z_axis, V_ocv_axis+tol_OCV_pack, "--")
    plt.plot(z_axis, V_ocv_axis-tol_OCV_pack, "--")
    plt.xlabel("SOC (real)"); plt.ylabel("Pack Voltage (V)")
    plt.title(f"Rest vs OCV band (±{tol_OCV_pack:.0f} V)")
    plt.grid(True)

    settle_min_sec = 15*60
    stab_win = 5
    dv_stab_thr = 0.2
    violate = np.zeros(rs.size, dtype=bool)

    def median_window(x, idx, w=5):
        half = w//2
        i0 = max(0, idx-half); i1 = min(len(x)-1, idx+half)
        return np.median(x[i0:i1+1])

    for i in range(rs.size):
        idx = np.arange(rs[i], re[i]+1)
        if t_sec[idx[-1]] - t_sec[idx[0]] < settle_min_sec:
            continue
        seg = idx[(t_sec[idx] - t_sec[idx[0]]) >= settle_min_sec]
        Vc = raw_V[seg]
        dVc = np.r_[0.0, np.abs(np.diff(Vc))]
        dvmax = np.maximum.accumulate(dVc)  # 简化稳定性判据
        if np.any(dvmax<=dv_stab_thr):
            loc = seg[np.where(dvmax<=dv_stab_thr)[0][0]]
        else:
            loc = seg[-1]

        Vrep = median_window(raw_V, loc, 5)
        zrep = median_window(SOC_real_all, loc, 5)
        Vocv = ocv_interp.f_avg(zrep)*pack_series
        plt.scatter([zrep],[Vrep], s=24)
        if abs(Vrep - Vocv) > tol_OCV_pack:
            violate[i] = True

    if np.any(violate):
        ibad = np.flatnonzero(violate)[0]
        cut_at = rs[ibad]
        if cut_at-1 >= min_points_keep:
            print(f"[Gate#2b] OCV violation at rest seg #{ibad+1} (idx={cut_at}). Keep prefix ({cut_at-1} pts) and re-check.")
            keep = slice(0, cut_at-1)
            t_ms2 = t_ms[keep]
            raw_I2 = raw_I[keep]
            raw_V2 = raw_V[keep]
            SOC2  = SOC_real_all[keep]
            if save:
                import os
                os.makedirs(outdir, exist_ok=True)
                plt.savefig(f"{outdir}/gate2b_rest_rack{rack_id}.png", dpi=150, bbox_inches="tight")
                plt.close(fig)
            else:
                plt.show()
            return t_ms2, raw_I2, raw_V2, SOC2
        else:
            raise RuntimeError(f"[Gate#2b FAIL] Not enough points before first OCV violation (remain {cut_at-1} < {min_points_keep}).")
    else:
        print(f"[Gate#2b OK] All rest segments satisfy |V-OCV| ≤ {tol_OCV_pack:.1f} V")
        if save:
            import os
            os.makedirs(outdir, exist_ok=True)
            plt.savefig(f"{outdir}/gate2b_rest_rack{rack_id}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()
        return t_ms, raw_I, raw_V, SOC_real_all

def early_check_plot(t_chk, y_meas, y_sim, Nest, fitpct, thr, SOC0, elapsed_s, rack_id="??", save=False, outdir="figs"):
    plt.figure(figsize=(7,3))
    plt.plot(t_chk, y_meas, "k-", label="Measured")
    plt.plot(t_chk, y_sim,  "b-", label="Model")
    plt.grid(True)
    plt.legend()
    plt.title(f"Early check | N={Nest} | Fit={fitpct:.2f}% (thr={thr:.0f}%) | SoC0={SOC0:.3f} | {elapsed_s:.2f}s")
    plt.xlabel("Time (s)"); plt.ylabel("Voltage (V)")
    if save:
        import os
        os.makedirs(outdir, exist_ok=True)
        plt.savefig(f"{outdir}/gate3_early_rack{rack_id}.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
