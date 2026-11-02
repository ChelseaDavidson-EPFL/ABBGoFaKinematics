"""
Screw model of the ABB GoFa (CRB 15000-5/0.95), space (world/base) frame.

Conventions
-----------
* Twist ordering: S_i = [w_i ; v_i]  (angular part first, linear part second).
* v_i = -w_i x r_i, with r_i any point on the joint axis, expressed in the
  base frame at the zero configuration.
* M = ^W T_tool0(0) is the tool0 pose at q = 0 (matches the URDF's
  flange->tool0 joint, rpy = (0, pi/2, 0)).
* Joint limits are taken from the URDF.
"""
import numpy as np

JOINT_NAMES = [f"joint_{i}" for i in range(1, 7)]

# Joint axes (unit vectors) in the base frame at q = 0
OMEGA = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
], dtype=float)

# A point on each joint axis (base frame, q = 0) [m]
R_AXIS = np.array([
    [0.000, 0, 0.265],
    [0.000, 0, 0.265],
    [0.000, 0, 0.709],
    [0.000, 0, 0.819],
    [0.470, 0, 0.819],
    [0.571, 0, 0.899],
], dtype=float)

# Linear part of each screw axis: v_i = -w_i x r_i
V = -np.cross(OMEGA, R_AXIS)

# Screw axes, one per COLUMN: shape (6, 6)  ->  S[:, i] = [w_i; v_i]
S_LIST = np.vstack([OMEGA.T, V.T])

# Home configuration of tool0 in the base frame
M = np.array([
    [ 0, 0, 1, 0.571],
    [ 0, 1, 0, 0.000],
    [-1, 0, 0, 0.899],
    [ 0, 0, 0, 1.000],
], dtype=float)

# Joint limits [rad] (lower, upper), from the URDF
Q_LIMITS = np.array([
    [-np.pi,           np.pi],
    [-np.pi,           np.pi],
    [-3.9269908169872414, 1.4835298641951802],
    [-np.pi,           np.pi],
    [-np.pi,           np.pi],
    [-4.71238898038469,   4.71238898038469],
])

# Same table as in the assignment, used as a self-check
_V_EXPECTED = np.array([
    [ 0.000, 0.000, 0.000],
    [-0.265, 0.000, 0.000],
    [-0.709, 0.000, 0.000],
    [ 0.000, 0.819, 0.000],
    [-0.819, 0.000, 0.470],
    [ 0.000, 0.899, 0.000],
])


def _check_table():
    assert np.allclose(V, _V_EXPECTED), "Screw v_i do not match the table"


_check_table()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("Screw axes S_i = [w; v] (one per row):")
    print(np.hstack([OMEGA, V]))
    print("\nM =\n", M)
    print("\nJoint limits [rad]:\n", Q_LIMITS)