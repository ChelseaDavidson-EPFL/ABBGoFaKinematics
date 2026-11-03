"""Place this script alongside crb15000-ik.exe (Windows) or crb15000-ik (Linux)."""
import json
import os
from pathlib import Path
import subprocess

binary_name = "crb15000-ik.exe" if os.name == "nt" else "crb15000-ik"
binary = Path(__file__).resolve().parent / binary_name

request = {
    # Position in metres; quaternion order w, x, y, z.
    "pose": [0.571, 0.0, 0.899, 1.0, 0.0, 1.0, 0.0],
    "frame": "tool0",
    "respect_limits": False,
}

result = subprocess.run(
    [str(binary), "solve-pose"],
    input=json.dumps(request),
    capture_output=True,
    text=True,
)
if not result.stdout.strip():
    raise RuntimeError(f"Solver failed to produce JSON (exit {result.returncode}): {result.stderr}")
report = json.loads(result.stdout)
if result.returncode != 0:
    raise RuntimeError(report.get("error", result.stderr))

print(f"Found {len(report['solutions'])} IK solutions")
for index, solution in enumerate(report["solutions"], start=1):
    angles = ", ".join(f"{q:.10f}" for q in solution["q"])
    print(f"IK {index}: [{angles}]")
    print(f"  Within joint limits: {solution['within_limits']}")
