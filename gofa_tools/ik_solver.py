"""
Wrapper of the supplied CRB 15000 IK executable (Linux or Windows, picked from the OS).

Poses are sent as 4x4 tool0 matrices through the batch command `solve`, so no
quaternion convention is involved. Angles are returned in radians.
"""
import json
import os
import stat
import subprocess
from pathlib import Path

import numpy as np

IK_ROOT = Path(__file__).resolve().parent.parent / "abbGofa_ik"   # folder from abbGofa_ik.zip
IK_BINARY = IK_ROOT / ("crb15000-ik-windows-x86_64/crb15000-ik.exe" if os.name == "nt"
                       else "crb15000-ik-linux-x86_64/crb15000-ik")


def _ensure_executable(path=IK_BINARY):
    if not path.exists():
        raise FileNotFoundError(f"IK solver not found at {path}.\n"
                                f"Unzip abbGofa_ik.zip in the repository root.")
    if os.name != "nt" and not os.access(path, os.X_OK):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _solve_raw(Ts):
    """Solver JSON for a list of 4x4 tool0 targets: one {"Ok": ...} or {"Err": ...} per pose."""
    _ensure_executable()
    req = {"frame": "tool0", "poses": [np.asarray(T, float).ravel().tolist() for T in Ts]}
    res = subprocess.run([str(IK_BINARY), "solve"], input=json.dumps(req),
                         capture_output=True, text=True)
    # The solver exits with 1 as soon as ONE pose fails, but still prints the JSON
    # results of all poses. Parse them first and only stop if there is no JSON.
    try:
        results = json.loads(res.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"IK solver failed (exit {res.returncode}): {res.stderr or res.stdout}")
    if not isinstance(results, list) or len(results) != len(Ts):
        raise RuntimeError(f"IK solver failed (exit {res.returncode}): {results}")
    return results


def solve_batch(Ts):
    """All IK solutions for each pose: a list (one per pose) of lists of q arrays.
    A pose the solver cannot resolve gives an empty list and a warning."""
    out = []
    for r in _solve_raw(Ts):
        if "Err" in r:
            print(f"[warn] solver error: {r['Err']}")
            out.append([])
        else:
            out.append([np.array(s["q"], float) for s in r["Ok"]["solutions"]])
    return out


def solve_pose(T):
    """All IK solutions (list of q arrays) for one 4x4 tool0 target. Raises on solver error."""
    r = _solve_raw([T])[0]
    if "Err" in r:
        raise RuntimeError(r["Err"])
    return [np.array(s["q"], float) for s in r["Ok"]["solutions"]]
