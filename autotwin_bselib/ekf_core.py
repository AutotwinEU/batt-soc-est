# ekf_core.py —— 用这个完整版本覆盖 class OCVInterp

# ekf_core.py
# -*- coding: utf-8 -*-
"""
EKF core for 2-RC battery ECM + OCV tables (charge/discharge),
with slope-adaptive fusion to Coulomb Counting.

Exports:
    - class OCVInterp
    - function run_ekf(I, V, SOCp, param_vec, deltaT, ocv_interp, ...)
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
from scipy.interpolate import PchipInterpolator



class OCVInterp:
    """
    OCV 插值器（cell 电压层面）：
      - f_charge(z),  f_discharge(z),  f_avg(z)
      - slope_c(z),   slope_d(z),      slope_avg(z)   # dOCV/dSOC
    说明：
      * 返回标量/数组自适应：标量 -> float，数组 -> ndarray
      * 斜率是“每单位真实 SOC (0~1) 变化的电压变化”，单位约 V/1SOC。
    """
    def __init__(self, SOC_c, OCV_c, SOC_d, OCV_d):
        self.SOC_c = np.asarray(SOC_c, dtype=float)
        self.OCV_c = np.asarray(OCV_c, dtype=float)
        self.SOC_d = np.asarray(SOC_d, dtype=float)
        self.OCV_d = np.asarray(OCV_d, dtype=float)

        # 主插值器（支持外推）
        self.f_c = PchipInterpolator(self.SOC_c, self.OCV_c, extrapolate=True)
        self.f_d = PchipInterpolator(self.SOC_d, self.OCV_d, extrapolate=True)
        # 一阶导数插值器
        self.df_c = self.f_c.derivative()
        self.df_d = self.f_d.derivative()

    # ---- OCV 值 ----
    def f_charge(self, z):
        val = self.f_c(z)
        return float(val) if np.ndim(val) == 0 else val

    def f_discharge(self, z):
        val = self.f_d(z)
        return float(val) if np.ndim(val) == 0 else val

    def f_avg(self, z):
        vc = self.f_c(z)
        vd = self.f_d(z)
        val = 0.5*(vc + vd)
        return float(val) if np.ndim(val) == 0 else val

    # ---- dOCV/dSOC 斜率 ----
    def slope_c(self, z):
        """充电曲线斜率 dOCV/dSOC."""
        val = self.df_c(z)
        return float(val) if np.ndim(val) == 0 else val

    def slope_d(self, z):
        """放电曲线斜率 dOCV/dSOC."""
        val = self.df_d(z)
        return float(val) if np.ndim(val) == 0 else val

    def slope_avg(self, z):
        """(充/放)平均曲线斜率 dOCV/dSOC."""
        vc = self.df_c(z)
        vd = self.df_d(z)
        val = 0.5*(vc + vd)
        return float(val) if np.ndim(val) == 0 else val


# --------------- Battery model (2-RC) -----------------

def _state_step(x: np.ndarray, u_dis: float, p: np.ndarray, dt: float) -> np.ndarray:
    """
    Forward-Euler 离散化的状态方程：
      x = [SOC, iR1, iR2]
      u_dis = 放电为正(A)，即 load current（正值消耗电量）
      p = [R0 R1 R2 tau1 tau2 Q M1 M2 M3]
    """
    # unpack
    tau1, tau2, Q = p[3], p[4], p[5]

    # states
    soc, i1, i2 = float(x[0]), float(x[1]), float(x[2])

    # dynamics
    dsoc = -(1.0/(Q*3600.0)) * u_dis
    di1  = (-1.0/tau1) * i1 + (1.0/tau1) * u_dis
    di2  = (-1.0/tau2) * i2 + (1.0/tau2) * u_dis

    xnext = np.empty(3, dtype=float)
    xnext[0] = soc + dt * dsoc
    xnext[1] = i1  + dt * di1
    xnext[2] = i2  + dt * di2
    # clamp SOC to [0,1] (real bounds由外层转换为 user%)
    xnext[0] = float(np.clip(xnext[0], 0.0, 1.0))
    return xnext


def _meas_voltage(x: np.ndarray, u_dis: float, p: np.ndarray,
                  ocv: OCVInterp, pack_series: int) -> float:
    """
    测量方程：终端电压
      y = OCV(pack) - R0*u - R1*iR1 - R2*iR2 + M0
    放电为正：u_dis > 0 -> 使用放电OCV与 M1；充电 -> 充电OCV与 M2；静置 -> 平均OCV与 M3
    """
    R0, R1, R2 = p[0], p[1], p[2]
    M1, M2, M3 = p[6], p[7], p[8]
    soc, i1, i2 = float(x[0]), float(x[1]), float(x[2])

    if u_dis > 0:        # discharge
        M0 = M1
        ocv_cell = ocv.f_d(soc)
    elif u_dis < 0:      # charge
        M0 = M2
        ocv_cell = ocv.f_c(soc)
    else:                # idle
        M0 = M3
        ocv_cell = ocv.f_avg(soc)

    y = float(ocv_cell) * float(pack_series) - R0*u_dis - R1*i1 - R2*i2 + M0
    return y


# --------------- EKF helpers -----------------

def _num_jacobian_x(f: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    数值雅可比 (对状态 x 的偏导)，简单前向差分
    返回形状 (1, nx)，仅用于测量函数的 H
    """
    nx = x.size
    base = float(f(x))
    J = np.zeros((1, nx), dtype=float)
    for i in range(nx):
        xp = x.copy()
        xp[i] += eps
        J[0, i] = (float(f(xp)) - base) / eps
    return J


def _num_jacobian_state(F: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    数值雅可比 (对状态 x 的偏导)，用于状态转移矩阵 F_k = d f / d x
    返回形状 (nx, nx)
    """
    nx = x.size
    x_base = F(x)
    J = np.zeros((nx, nx), dtype=float)
    for i in range(nx):
        xp = x.copy()
        xp[i] += eps
        J[:, i] = (F(xp) - x_base) / eps
    return J


# --------------- Public EKF runner -----------------

def run_ekf(
    I: np.ndarray,
    V: np.ndarray,
    SOCp: np.ndarray,
    param_vec: np.ndarray,
    deltaT: float,
    ocv_interp: OCVInterp,
    SOC_min_real: float,
    SOC_max_real: float,
    I_idle_thresh: float,
    S_low: float,
    S_high: float,
    slope_floor: float,
    pack_series: int,
    Q_proc: Tuple[float, float, float] = (1e-5, 1e-6, 1e-6),
    R_meas: float = 0.8,
    P0_diag: Tuple[float, float, float] = (1e-5, 0.0, 0.0),
) -> Dict[str, np.ndarray]:
    """
    执行 EKF + 斜率自适应融合。
    参数：
        I, V, SOCp: 同长度序列（秒级均匀采样）
        param_vec: [R0 R1 R2 tau1 tau2 Q M1 M2 M3]
        deltaT: 采样时间(s)
        ocv_interp: OCVInterp
        SOC_min_real, SOC_max_real: 实际SOC映射到user%的范围（用于CC与最终映射）
        I_idle_thresh: 判定“静置”的电流阈值
        S_low, S_high, slope_floor: 斜率自适应融合的参数
        pack_series: 串联节数（决定 pack 电压 = cell_OCV * pack_series）
        Q_proc: 过程噪声对角线
        R_meas: 测量噪声（标量）
        P0_diag: 初始协方差对角线

    返回：
        dict 包含：
            soc_fused: 融合后的 real SOC (0~1)
            v_ekf: EKF 预测/更正后的电压（测量模型输出）
            alpha: 每个时刻融合权重 alpha
            soc_cc: 累积库仑计 real SOC
            soc_ekf: EKF 估计的 real SOC
    """
    I = np.asarray(I, dtype=float).ravel()
    V = np.asarray(V, dtype=float).ravel()
    SOCp = np.asarray(SOCp, dtype=float).ravel()
    n = min(len(I), len(V), len(SOCp))
    I, V, SOCp = I[:n], V[:n], SOCp[:n]

    # 由 user(%) 反映到 real(0~1)
    SOC_real = SOCp/100.0 * (SOC_max_real - SOC_min_real) + SOC_min_real
    soc0 = float(SOC_real[0])

    p = np.asarray(param_vec, dtype=float).ravel()
    Q_Ah = float(p[5])

    # --- Coulomb Counting (real SOC) ---
    soc_cc = np.empty(n, dtype=float)
    soc_cc[0] = soc0
    for k in range(1, n):
        # 放电为正的 u_dis：数据 I 常以“充电为正/放电为负”，这里统一成放电为正
        u_dis_prev = -float(I[k-1])
        soc_cc[k] = soc_cc[k-1] - u_dis_prev * deltaT / (Q_Ah * 3600.0)
        soc_cc[k] = float(np.clip(soc_cc[k], SOC_min_real, SOC_max_real))

    # --- EKF 初始化 ---
    x = np.array([soc0, 0.0, 0.0], dtype=float)
    P = np.diag(P0_diag)
    Qk = np.diag(Q_proc)
    Rk = np.array([[R_meas]], dtype=float)

    soc_ekf = np.empty(n, dtype=float)
    v_ekf = np.empty(n, dtype=float)
    alpha = np.empty(n, dtype=float)

    # 方便计算的函数闭包（便于雅可比数值求导）
    def f_state(xx: np.ndarray, udis: float) -> np.ndarray:
        return _state_step(xx, udis, p, deltaT)

    def h_meas(xx: np.ndarray, udis: float) -> float:
        return _meas_voltage(xx, udis, p, ocv_interp, pack_series)

    # 斜率 -> alpha
    def _alpha_from_slope(s):
        # 线性从 [S_low, S_high] 拉伸到 [0,1]
        aa = (abs(s) - S_low) / max(S_high - S_low, 1e-9)
        return float(np.clip(aa, 0.0, 1.0))

    for k in range(n):
        u_dis = -float(I[k])  # 放电为正
        yk = float(V[k])

        # ---------- Predict ----------
        F_func = lambda xx: f_state(xx, u_dis)
        F = _num_jacobian_state(F_func, x)
        x_pred = f_state(x, u_dis)
        P_pred = F @ P @ F.T + Qk

        # ---------- Update ----------
        H_func = lambda xx: h_meas(xx, u_dis)
        H = _num_jacobian_x(H_func, x_pred)
        y_pred = h_meas(x_pred, u_dis)
        Syy = H @ P_pred @ H.T + Rk  # innovation covariance
        K = P_pred @ H.T @ np.linalg.inv(Syy)
        innov = np.array([[yk - y_pred]], dtype=float)  # (1,1)
        x_upd = x_pred + (K @ innov).ravel()
        P_upd = (np.eye(3) - K @ H) @ P_pred

        # clamp SOC
        x_upd[0] = float(np.clip(x_upd[0], 0.0, 1.0))

        # ---------- 融合（斜率自适应） ----------
        # 根据电流状态选择斜率
        if abs(I[k]) <= I_idle_thresh:
            slope_now = ocv_interp.slope_avg(x_upd[0])
        elif I[k] < -I_idle_thresh:  # I<0 -> u_dis>0 -> 放电
            slope_now = ocv_interp.slope_d(x_upd[0])
        else:  # I>0 -> 充电
            slope_now = ocv_interp.slope_c(x_upd[0])

        slope_use = max(abs(float(slope_now)), slope_floor)
        a = _alpha_from_slope(slope_use)
        z_fused = a * x_upd[0] + (1.0 - a) * soc_cc[k]
        z_fused = float(np.clip(z_fused, SOC_min_real, SOC_max_real))
        x_upd[0] = z_fused

        # 保存输出
        soc_ekf[k] = x_upd[0]
        v_ekf[k] = h_meas(x_upd, u_dis)
        alpha[k] = a

        # 准备下次循环
        x, P = x_upd, P_upd

    return {
        "soc_fused": soc_ekf,      # 已融合后的 real SOC (0~1)
        "v_ekf": v_ekf,            # EKF 对电压的估计
        "alpha": alpha,            # 融合权重
        "soc_cc": soc_cc,          # 库仑计（real SOC）
        "soc_ekf": soc_ekf.copy(), # 为了兼容性，等同 soc_fused
    }
