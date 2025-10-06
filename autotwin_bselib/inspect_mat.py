# inspect_mat.py
import argparse, numpy as np
from scipy.io import loadmat

def as_shape(v):
    if isinstance(v, np.ndarray):
        return f"ndarray shape={v.shape} dtype={v.dtype}"
    return type(v).__name__

def walk(obj, prefix="", depth=0, max_depth=2):
    if depth>max_depth: return
    if isinstance(obj, dict):
        for k,v in obj.items():
            if str(k).startswith("__"): continue
            print(f"{prefix}{k} -> {as_shape(v)}")
            walk(v, prefix=f"{prefix}{k}.", depth=depth+1, max_depth=max_depth)
    elif hasattr(obj, "__dict__"):
        for k,v in obj.__dict__.items():
            if str(k).startswith("__"): continue
            print(f"{prefix}{k} -> {as_shape(v)}")
            walk(v, prefix=f"{prefix}{k}.", depth=depth+1, max_depth=max_depth)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", required=True, help="Path to SOC_OCV.mat")
    args = ap.parse_args()

    m = loadmat(args.mat, simplify_cells=True, struct_as_record=False)
    print("Top-level keys:")
    keys = [k for k in m.keys() if not k.startswith("__")]
    for k in keys:
        print(" -", k, "->", as_shape(m[k]))
    print("\nPeek nested (depth<=1):")
    for k in keys:
        print(f"\n[{k}]")
        walk(m[k], prefix=f"{k}.", depth=0, max_depth=1)
