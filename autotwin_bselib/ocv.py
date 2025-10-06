# soc_py/ocv.py
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import loadmat

def _clean_vec(x):
    x = np.asarray(x, dtype=float).ravel()
    # 去 NaN、去重、排序
    ok = np.isfinite(x)
    x = x[ok]
    x = np.unique(x)
    return x

def _build_from_arrays(SOC_c, OCV_c, SOC_d, OCV_d):
    SOC_c = _clean_vec(SOC_c)
    OCV_c = np.asarray(OCV_c, dtype=float).ravel()[: len(SOC_c)]
    SOC_d = _clean_vec(SOC_d)
    OCV_d = np.asarray(OCV_d, dtype=float).ravel()[: len(SOC_d)]

    if len(SOC_c) < 3 or len(SOC_d) < 3:
        raise ValueError("OCV表点数太少（<3）。")

    if (SOC_c.min() < 0) or (SOC_c.max() > 1) or (SOC_d.min() < 0) or (SOC_d.max() > 1):
        raise ValueError("SoC 必须在 [0,1]。")

    return {
        "SOC_c": SOC_c,
        "OCV_c": OCV_c,
        "SOC_d": SOC_d,
        "OCV_d": OCV_d,
    }

def _try_from_mat(path: Path):
    m = loadmat(str(path), simplify_cells=True, struct_as_record=False)
    # 原 MATLAB 结构：chargeOCV_table.SoC / .OCV, dischargeOCV_table.SoC / .OCV
    try:
        ch = m["chargeOCV_table"]
        dc = m["dischargeOCV_table"]
        SOC_c = ch["SoC"]
        OCV_c = ch["OCV"]
        SOC_d = dc["SoC"]
        OCV_d = dc["OCV"]
        return _build_from_arrays(SOC_c, OCV_c, SOC_d, OCV_d)
    except Exception:
        # 也可能是平铺四个向量：
        for keys in [
            ("SOC_c","OCV_c","SOC_d","OCV_d"),
            ("SoC_charge","OCV_charge","SoC_discharge","OCV_discharge"),
        ]:
            if all(k in m for k in keys):
                return _build_from_arrays(m[keys[0]], m[keys[1]], m[keys[2]], m[keys[3]])
        raise KeyError("MAT 文件里没找到可识别的 (SoC, OCV) 表。")

def _read_two_csv_in_dir(dirpath: Path):
    f_charge = dirpath / "OCV_charge.csv"
    f_dis    = dirpath / "OCV_discharge.csv"
    if not (f_charge.exists() and f_dis.exists()):
        raise FileNotFoundError(f"目录 {dirpath} 下未找到 OCV_charge.csv / OCV_discharge.csv")
    C = pd.read_csv(f_charge)
    D = pd.read_csv(f_dis)
    if not {"SoC","OCV"}.issubset(C.columns) or not {"SoC","OCV"}.issubset(D.columns):
        raise KeyError("CSV 列名必须为 SoC, OCV")
    return _build_from_arrays(C["SoC"], C["OCV"], D["SoC"], D["OCV"])

def _read_single_csv(file: Path):
    df = pd.read_csv(file)
    cols = set(df.columns.str.lower())
    # 单表四列
    if {"soc_charge","ocv_charge","soc_discharge","ocv_discharge"}.issubset(cols):
        return _build_from_arrays(
            df[[c for c in df.columns if c.lower()=="soc_charge"][0]],
            df[[c for c in df.columns if c.lower()=="ocv_charge"][0]],
            df[[c for c in df.columns if c.lower()=="soc_discharge"][0]],
            df[[c for c in df.columns if c.lower()=="ocv_discharge"][0]],
        )
    # 单表两列（文件名含 charge/dis 决定是哪一条），不推荐但兼容
    if {"soc","ocv"}.issubset(cols):
        name = file.name.lower()
        if "charge" in name:
            raise ValueError("检测到充电表，但缺少放电表；请改用目录方式放两张 CSV。")
        if "dis" in name or "discharge" in name:
            raise ValueError("检测到放电表，但缺少充电表；请改用目录方式放两张 CSV。")
    raise KeyError(
        "单个 CSV 需要四列：SoC_charge,OCV_charge,SoC_discharge,OCV_discharge；"
        "或者把两张表 OCV_charge.csv / OCV_discharge.csv 放在同一目录。"
    )

def load_ocv_tables(path_like: str | Path):
    """
    万能入口：
    - .mat ：按老结构读
    - .csv ：单表四列
    - 目录 ：包含 OCV_charge.csv 与 OCV_discharge.csv
    返回 dict：{SOC_c, OCV_c, SOC_d, OCV_d}
    """
    p = Path(path_like)
    if p.is_dir():
        return _read_two_csv_in_dir(p)
    suf = p.suffix.lower()
    if suf == ".mat":
        return _try_from_mat(p)
    if suf == ".csv":
        return _read_single_csv(p)
    raise ValueError(f"不支持的 OCV 路径类型：{p}")
