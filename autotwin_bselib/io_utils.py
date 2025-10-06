# io_utils.py
import os, re, glob, shutil
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

GUID2RACK = {
    "0b822130-62a1-4a7e-8516-9d5802e2713e": 1,
    "a10d3a06-681f-4646-b222-80c4ce9992e6": 2,
    "c8b2f111-f075-4e89-bf6e-01db4bb94bc5": 3,
    "4d4305fb-23f3-4967-b823-7563f13717e2": 4,
    "b7ce837b-d7b5-4bfd-8316-f4665613dfb9": 5,
    "2fb4ec59-aa24-4bfe-a70e-6fd40bb3ec26": 6,
    "a70b1f2e-cfc0-4655-bb16-b5d02b85e012": 7,
    "fbbe9145-4f19-4718-9ffc-70cd7f768c9c": 8,
    "dd6427f6-7684-4544-aa4b-c632160918e7": 9,
    "c81ab4a1-1835-4f06-8a2b-d423cd0ea66f": 10,
    "873a9abf-647e-421e-a1b1-299f8eec094e": 11,
    "4073fd3b-89b7-4d5a-b15e-c7e18843ffa1": 12,
    "aeabe2b7-e23c-472a-8ef7-2c03f4a32b61": 13,
    "855577f5-ec31-472b-bb1f-cbd876af2dec": 14
}

VARMAP = {
    "current":"current",
    "voltage":"voltage",
    "soc":"soc",
    "max_cell_voltage":"max_cell_voltage",
    "min_cell_voltage":"min_cell_voltage",
    "rack_current":"current",
    "rack_voltage":"voltage",
    "rack_soc":"soc",
    "rack_max_cell_voltage":"max_cell_voltage",
    "rack_min_cell_voltage":"min_cell_voltage",
}

def _read_first_timestamp_ms(csv_path, header_lines=3):
    # 兼容你数据的前三行头
    df = pd.read_csv(csv_path, skiprows=header_lines)
    # 第一列/Time 列
    col = "Time" if "Time" in df.columns else df.columns[0]
    t0 = df[col].iloc[0]
    # 直接数字（ms）
    try:
        t0 = float(t0)
        return int(t0)
    except Exception:
        pass
    # 尝试解析时间字符串
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(str(t0), fmt)
            return int(dt.timestamp()*1000)
        except Exception:
            continue
    # 尝试作为 POSIX 秒
    try:
        dt = datetime.fromtimestamp(float(t0))
        return int(dt.timestamp()*1000)
    except Exception:
        raise ValueError(f"无法解析首行时间: {t0}")

def rename_rack_files_with_time(folder, dry_run=False, data_header_lines=3, tolerance_ms=5):
    """把 <GUID>_rack_<var>.csv 统改名为 Rack_XX_<var>_<yyyymmddHHMMss>.csv"""
    folder = str(folder)
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        print(f'No CSV in "{folder}".'); 
        return

    # 找三元组分组
    groups = {}  # guid -> {var: path}
    pat = re.compile(r"([0-9a-f\-]{36}).*?_rack_([a-z_]+)\.csv$", re.I)
    for p in files:
        name = os.path.basename(p)
        if re.match(r"^Rack_\d{2}_.+\.csv$", name):  # 已经标准化的跳过
            continue
        m = pat.search(name)
        if not m: 
            continue
        guid, var = m.group(1).lower(), m.group(2).lower()
        var = VARMAP.get(var, var)
        groups.setdefault(guid, {})[var] = p

    required = {"current","voltage","soc"}
    renamed, skipped = 0, 0
    for guid, mapping in groups.items():
        if not required.issubset(mapping.keys()):
            skipped += len(mapping); 
            continue
        if guid not in GUID2RACK:
            skipped += len(mapping); 
            continue

        try:
            t0I = _read_first_timestamp_ms(mapping["current"], data_header_lines)
            t0V = _read_first_timestamp_ms(mapping["voltage"], data_header_lines)
            t0S = _read_first_timestamp_ms(mapping["soc"],     data_header_lines)
        except Exception:
            skipped += len(mapping); 
            continue

        t0s = [t0I,t0V,t0S]
        if max(t0s)-min(t0s) > tolerance_ms:
            skipped += len(mapping); 
            continue
        t0_ms = int(np.median(t0s))
        dt = datetime.utcfromtimestamp(t0_ms/1000)
        ts_str = dt.strftime("%Y%m%d%H%M%S")

        rack_num = GUID2RACK[guid]
        for vname, old_path in mapping.items():
            new_name = f"Rack_{rack_num:02d}_{vname}_{ts_str}.csv"
            new_path = os.path.join(folder, new_name)
            # 避免覆盖
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

def extract_ts(name):
    m = re.search(r"_(\d{14})\.csv$", name)
    return m.group(1) if m else None

def pick_triplet(folder, rack_id, ts_req=""):
    folder = Path(folder)
    cands_c = list(folder.glob(f"Rack_{rack_id}_current_*.csv"))
    cands_v = list(folder.glob(f"Rack_{rack_id}_voltage_*.csv"))
    cands_s = list(folder.glob(f"Rack_{rack_id}_soc_*.csv"))
    if not (cands_c and cands_v and cands_s):
        raise FileNotFoundError(f"Files not found for Rack_{rack_id} (current/voltage/soc).")

    def ts_set(files):
        return set(filter(None, (extract_ts(f.name) for f in files)))

    if ts_req:
        ts = ts_req
    else:
        common = ts_set(cands_c) & ts_set(cands_v) & ts_set(cands_s)
        if not common:
            raise RuntimeError("No common timestamp across trio.")
        # 取最新
        ts = max(common)

    cur = folder / f"Rack_{rack_id}_current_{ts}.csv"
    vol = folder / f"Rack_{rack_id}_voltage_{ts}.csv"
    soc = folder / f"Rack_{rack_id}_soc_{ts}.csv"
    for p in (cur,vol,soc):
        if not p.exists():
            raise FileNotFoundError(f"Missing: {p}")
    return str(cur), str(vol), str(soc), ts

def load_triplet(fI, fV, fS, header_lines=3):
    TI = pd.read_csv(fI, skiprows=header_lines)
    TV = pd.read_csv(fV, skiprows=header_lines)
    TS = pd.read_csv(fS, skiprows=header_lines)
    TI.columns = ["Time","Current"]
    TV.columns = ["Time","Voltage"]
    TS.columns = ["Time","soc"]

    tI = TI["Time"].to_numpy(float)
    tV = TV["Time"].to_numpy(float)
    tS = TS["Time"].to_numpy(float)
    if not (np.array_equal(tI, tV) and np.array_equal(tI, tS)):
        raise ValueError("Time vectors must match (current/voltage/soc).")

    t_ms3 = np.stack([tI,tV,tS], axis=1)
    I = TI["Current"].to_numpy(float)
    V = TV["Voltage"].to_numpy(float)
    S = TS["soc"].to_numpy(float)
    return t_ms3, I, V, S

def resample_60s(t_ms, I, V, SOCp, deltaT=60):
    t0, t1 = t_ms[0], t_ms[-1]
    grid_ms = np.arange(t0, t1+1e-9, deltaT*1000.0)
    # 线性插值
    Ig = np.interp(grid_ms, t_ms, I)
    Vg = np.interp(grid_ms, t_ms, V)
    Sg = np.interp(grid_ms, t_ms, SOCp)
    t_grid = (grid_ms - grid_ms[0]) / 1000.0
    return t_grid, Ig, Vg, Sg
