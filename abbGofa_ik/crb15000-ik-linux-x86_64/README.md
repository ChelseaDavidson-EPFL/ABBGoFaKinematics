# CRB15000 IK executable

This package solves poses for the ABB CRB 15000-5/0.95 geometry.
The supplied executable needs Linux x86_64, glibc 2.38 or newer, and libgcc_s.
Rust, Python, GMP, and a separate URDF file are not needed at runtime.

From the extracted directory:

```sh
./crb15000-ik solve-pose --input pose.json
```

Or provide one JSON object through standard input, followed by EOF:

```sh
echo '{"pose":[0.571,0,0.899,0.7071067811865476,0,0.7071067811865476,0]}' \
  | ./crb15000-ik solve-pose
```

The input `pose` is `[px, py, pz, qw, qx, qy, qz]`. Position is in metres
relative to the robot base. The quaternion is normalized automatically and must
be nonzero. Optional `frame` is `"tool0"` (default), `"link6"`, or `"dh6"`.
The sample describes the zero-joint tool pose.

Output is a JSON object with `solutions` and numerical `diagnostics`.
Each solution contains:

- `q`: six joint angles `[q1, q2, q3, q4, q5, q6]` in radians.
- `within_limits`: whether all six angles satisfy the URDF position limits.
- `position_error`: forward-kinematics position error in metres.
- `rotation_error`: forward-kinematics orientation error in radians.
- `t6`, `radial_branch`, `j4_branch`: analytical solver metadata.

Only real, forward-kinematics-verified configurations are returned. By default,
all returned configurations are included, including those flagged invalid.
Add `"respect_limits": true` to the input to return only valid configurations.
`example-output.json` shows the full result for `pose.json`.

Joint 3 is shifted by a whole turn when needed to fit its URDF range
`[-3.9269908169872414, 1.4835298641951802]`. For example, `2.783185307179586`
is represented as approximately `-3.5`. If no equivalent angle fits, the
configuration is flagged invalid; angles are never clamped. Joint 6 uses
`[-4.71238898038469, 4.71238898038469]`; the other joints use `[-π, π]`.
Limit checks allow `1e-12` radians of numerical tolerance.

The solver returns one representative per configuration modulo `2π`, rather
than listing additional equivalent joint-6 turns. Its generic-pose formulation
does not enumerate continuous singular families or base-axis singular branches.
Numerical checks do not prove branch completeness for every possible pose.
The validity flag checks joint positions only, not collisions or motion paths.

Exit status is 0 for a completed solve, even if `solutions` is empty. Invalid
input and unresolved roots return `{"error":"message"}` on stdout, an error
message on stderr, and exit status 1. Applications should check both exit status
and the returned solutions array.

For another OS/CPU or an older Linux runtime, rebuild from the Rust source on
the intended platform using `cargo build --release --locked`. See the source
project's `rust/README.md` for the library API and batch matrix interface.
