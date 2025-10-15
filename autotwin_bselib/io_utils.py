# io_utils.py
# -*- coding: utf-8 -*-
import os, re, glob, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Tuple, Optional, Iterable

# 统一的变量名映射（容错）
VARMAP = {
    "current": "current",
    "voltage": "voltage",
    "soc": "soc",
    "max_cell_voltage": "max_cell_voltage",
    "min_cell_voltage": "min_cell_voltage",
    "rack_current": "current",
    "rack_voltage": "voltage",
    "rack_soc": "soc",
    "rack_max_cell_voltage": "max_cell_voltage",
    "rack_min_cell_voltage": "min_cell_voltage",
}

UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
TS14_RE = re.compile(r"(\d{14})")

# ---------- 读取首时间戳（毫秒） ----------
def _read_first_timestamp_ms(csv_path: str, header_lines: int = 3) -> int:
    df = pd.read_csv(csv_path, skiprows=header_lines)
    col = "Time" if "Time" in df.columns else df.columns[0]
    t0 = df[col].iloc[0]
    # 直接数字（ms）
    try:
        t0 = float(t0)
        return int(t0)
    except Exception:
        pass
    # 解析常见时间字符串
    fmts = ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")
    for fmt in fmts:
        try:
            dt = datetime.strptime(str(t0), fmt)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    # 解析为 POSIX 秒
    try:
        dt = datetime.fromtimestamp(float(t0))
        return int(dt.timestamp() * 1000)
    except Exception:
        raise ValueError(f"无法解析首行时间: {t0}")

# ---------- 把 GUID 命名的三件套标准化为：{GUID}_rack_<var>_<yyyymmddHHMMSS>.csv ----------
def rename_rack_files_with_time(folder: str, dry_run: bool = False, data_header_lines: int = 3, tolerance_ms: int = 5):
    """
    标准化命名：
      - 目标：{GUID}_rack_current_<yyyymmddHHMMSS>.csv / voltage / soc
      - 不再进行 GUID→数字 Rack_## 的映射
      - 已经是目标格式的文件不动
      - 旧的 Rack_##_* 格式也不动（保持兼容）
    分组三件套并检查首时间戳一致性（±tolerance_ms）。
    """
    folder = str(folder)
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        print(f'No CSV in "{folder}".')
        return

    # 已标准化目标格式：<GUID>_rack_<var>_<TS>.csv
    target_pat = re.compile(r"^[0-9a-f\-]{36}_rack_(current|voltage|soc)_\d{14}\.csv$", re.I)
    # 可重命名的原始格式：<GUID>_rack_<var>.csv（无 TS）
    raw_pat = re.compile(r"([0-9a-f\-]{36}).*?_rack_([a-z_]+)\.csv$", re.I)

    # 分组：guid -> {var: path}
    groups = {}
    for p in files:
        name = os.path.basename(p)
        if target_pat.match(name):
            # 已是目标格式，跳过
            continue
        m = raw_pat.search(name)
        if not m:
            # 留给旧 Rack_##_* 命名或其他文件，跳过
            continue
        guid = m.group(1).lower()
        var = VARMAP.get(m.group(2).lower(), m.group(2).lower())
        groups.setdefault(guid, {})[var] = p

    required = {"current", "voltage", "soc"}
    renamed, skipped = 0, 0
    for guid, mapping in groups.items():
        if not required.issubset(mapping.keys()):
            skipped += len(mapping)
            continue
        try:
            t0I = _read_first_timestamp_ms(mapping["current"], data_header_lines)
            t0V = _read_first_timestamp_ms(mapping["voltage"], data_header_lines)
            t0S = _read_first_timestamp_ms(mapping["soc"], data_header_lines)
        except Exception:
            skipped += len(mapping)
            continue

        t0s = [t0I, t0V, t0S]
        if max(t0s) - min(t0s) > tolerance_ms:
            skipped += len(mapping)
            continue

        t0_ms = int(np.median(t0s))
        ts_str = datetime.utcfromtimestamp(t0_ms / 1000).strftime("%Y%m%d%H%M%S")

        for vname, old_path in mapping.items():
            new_name = f"{guid}_rack_{vname}_{ts_str}.csv"
            new_path = os.path.join(folder, new_name)
            i, base = 1, new_path
            while os.path.exists(new_path):
                stem, ext = os.path.splitext(base)
                new_path = f"{stem}_{i}{ext}"
                i += 1
            if dry_run:
                print(f"DRY: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
            else:
                try:
                    shutil.move(old_path, new_path)
                    print(f"OK: {os.path.basename(old_path)} -> {os.path.basename(new_path)}")
                    renamed += 1
                except Exception:
                    skipped += 1

    print(f"[Rename] Done. Renamed: {renamed} | Skipped: {skipped}")

# ---------- 提取 14 位时间戳 ----------
def extract_ts(name: str) -> Optional[str]:
    m = TS14_RE.search(name)
    return m.group(1) if m else None

# ---------- 查找三件套（兼容 ts_in / ts_req；优先 GUID 命名，其次 Rack_## 命名） ----------
def pick_triplet(folder: str, rack_id: str, ts_in: str = "", ts_req: str = "") -> Tuple[str, str, str, str]:
    """
    优先假设 rack_id 是 GUID（UUID），查找：
      {GUID}_rack_current_*.csv / _voltage_*.csv / _soc_*.csv
    若找不到，兼容旧格式：
      Rack_{rack_id}_current_*.csv / Rack_{rack_id}_voltage_*.csv / Rack_{rack_id}_soc_*.csv

    ts_in/ts_req（可二选一）指定 14 位时间戳；不指定则取三者时间戳交集的“最新一组”。
    返回: (curFile, volFile, socFile, ts14)
    """
    folder = Path(folder)
    rid = str(rack_id).strip()

    def collect(prefix: str, cur_pat: str, vol_pat: str, soc_pat: str) -> Optional[Tuple[Iterable[Path], Iterable[Path], Iterable[Path]]]:
        cands_c = list(folder.glob(cur_pat))
        cands_v = list(folder.glob(vol_pat))
        cands_s = list(folder.glob(soc_pat))
        return (cands_c, cands_v, cands_s) if (cands_c and cands_v and cands_s) else None

    # 模式 1：GUID 命名
    have = collect(
        rid,
        f"{rid}_rack_current_*.csv",
        f"{rid}_rack_voltage_*.csv",
        f"{rid}_rack_soc_*.csv",
    )

    # 模式 2：旧 Rack_## 命名
    if not have:
        have = collect(
            f"Rack_{rid}",
            f"Rack_{rid}_current_*.csv",
            f"Rack_{rid}_voltage_*.csv",
            f"Rack_{rid}_soc_*.csv",
        )
        if not have:
            raise FileNotFoundError(f"Files not found for '{rack_id}' (current/voltage/soc).")

    cands_c, cands_v, cands_s = have

    def ts_set(files: Iterable[Path]) -> set:
        return set(filter(None, (extract_ts(f.name) for f in files)))

    ts_want = ts_in or ts_req
    if ts_want:
        ts = ts_want
    else:
        common = ts_set(cands_c) & ts_set(cands_v) & ts_set(cands_s)
        if not common:
            raise RuntimeError("No common timestamp across trio.")
        ts = max(common)  # 最新一组

    # 选定文件
    def pick_one(files: Iterable[Path], key: str) -> Path:
        for f in files:
            if f.name.endswith(f"_{ts}.csv") and key in f.name:
                return f
        # 容错：匹配同 ts 的第一个
        for f in files:
            if f.name.endswith(f"_{ts}.csv"):
                return f
        raise FileNotFoundError(f"Missing {key} file for timestamp {ts}")

    cur = pick_one(cands_c, "current")
    vol = pick_one(cands_v, "voltage")
    soc = pick_one(cands_s, "soc")

    return str(cur), str(vol), str(soc), ts

# ---------- 读取三件套 ----------
def load_triplet(fI: str, fV: str, fS: str, header_lines: int = 3):
    TI = pd.read_csv(fI, skiprows=header_lines)
    TV = pd.read_csv(fV, skiprows=header_lines)
    TS = pd.read_csv(fS, skiprows=header_lines)

    # 统一列名（容错：若没有标准名，则取前两列/第一列）
    def norm(df: pd.DataFrame, want: str):
        cols = [c for c in df.columns]
        # 时间列
        tcol = "Time" if "Time" in cols else cols[0]
        # 值列
        if want == "current":
            vcol = "Current" if "Current" in cols else (cols[1] if len(cols) > 1 else cols[0])
        elif want == "voltage":
            vcol = "Voltage" if "Voltage" in cols else (cols[1] if len(cols) > 1 else cols[0])
        else:
            vcol = "soc" if "soc" in cols else (cols[1] if len(cols) > 1 else cols[0])
        out = pd.DataFrame({"Time": df[tcol], "Value": df[vcol]})
        return out

    TI = norm(TI, "current")
    TV = norm(TV, "voltage")
    TS = norm(TS, "soc")

    tI = TI["Time"].to_numpy(dtype=float)
    tV = TV["Time"].to_numpy(dtype=float)
    tS = TS["Time"].to_numpy(dtype=float)

    if not (np.array_equal(tI, tV) and np.array_equal(tI, tS)):
        raise ValueError("Time vectors must match (current/voltage/soc).")

    t_ms3 = np.stack([tI, tV, tS], axis=1)
    I = TI["Value"].to_numpy(dtype=float)
    V = TV["Value"].to_numpy(dtype=float)
    S = TS["Value"].to_numpy(dtype=float)
    return t_ms3, I, V, S

# ---------- 60s 重采样 ----------
def resample_60s(t_ms: np.ndarray, I: np.ndarray, V: np.ndarray, SOCp: np.ndarray, deltaT: float = 60.0):
    t0, t1 = float(t_ms[0]), float(t_ms[-1])
    grid_ms = np.arange(t0, t1 + 1e-9, deltaT * 1000.0)
    Ig = np.interp(grid_ms, t_ms, I)
    Vg = np.interp(grid_ms, t_ms, V)
    Sg = np.interp(grid_ms, t_ms, SOCp)
    t_grid = (grid_ms - grid_ms[0]) / 1000.0  # 秒
    return t_grid, Ig, Vg, Sg
