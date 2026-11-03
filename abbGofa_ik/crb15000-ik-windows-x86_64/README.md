# Windows executable

Extract the ZIP and open PowerShell in the extracted directory:

```powershell
.\crb15000-ik.exe solve-pose --input pose.json
```

This is a native 64-bit Windows executable, intended for Windows 10/11 x64.
It runs without Rust, Python, WSL, or separate numerical-library DLLs.
The solver's Rust source and URDF are not needed by the recipient.

To call it from Python (Python 3 must be installed for this example):

```powershell
python example_script.py
```

Keep `example_script.py` alongside `crb15000-ik.exe`. It finds the executable
relative to the script, regardless of the working directory. Edit the `pose`
array in the script to change the target.

Input is one JSON object with `pose: [px, py, pz, qw, qx, qy, qz]`, position
in metres relative to the robot base, and quaternion in w/x/y/z order.
Nonzero quaternions are normalized. Optional `frame` defaults to `"tool0"`;
`"link6"` and `"dh6"` are also supported. Optional `respect_limits` defaults
to `false`, returning solutions both inside and outside joint limits.

Output JSON contains `solutions` and `diagnostics`. Each solution includes
`q` (six joint angles in radians), `within_limits` (boolean), FK position and
rotation errors, and analytical branch metadata. All returned configurations
are real and verified by forward kinematics. Joint 3 is represented inside
`[-3.9269908169872414, 1.4835298641951802]` whenever an equivalent angle fits;
otherwise that configuration is flagged invalid. Angles are never clamped.
Joint 6 uses `[-4.71238898038469, 4.71238898038469]`; the other joints use
`[-π, π]`. Limit checks allow `1e-12` radians of numerical tolerance.

Exit status 0 means the solve completed; `solutions` can still be empty.
Input errors and unresolved numerical roots return `{"error":"message"}`
and exit status 1. One representative per configuration modulo `2π` is
returned. Continuous singular families and base-axis singular branches are
not enumerated; branch completeness is not certified for every pose.
The limit flag checks joint positions, not collisions or motion paths.

Validation: the Windows executable and all 26 Windows test cases passed under
Wine 10 on Linux, including 7,628 reference IK branches across 1,000 poses and
the joint-3 boundary checks. It has not been tested on a physical Windows machine.
Its DLL imports are limited to Windows system libraries; no MinGW, GMP, MPFR,
or MPC DLLs are required.

## Rebuild from source on Linux

Install MinGW-w64 and add the Rust target:

```sh
rustup target add x86_64-pc-windows-gnu
bash scripts/build_windows.sh
```

The build script links the numerical and compiler support libraries statically.
The executable is written to
`rust/target/x86_64-pc-windows-gnu/release/crb15000-ik.exe`.
