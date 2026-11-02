"""
Standard Denavit-Hartenberg model of the ABB GoFa (CRB 15000-5/0.95).

    ^{i-1}T_i = Rz(theta_i + offset_i) Tz(d_i) Tx(a_i) Rx(alpha_i)

Frames at q = 0 (world axes):
    z0 = z (joint 1)   z1 = y (joint 2)   z2 = y (joint 3)
    z3 = x (joint 4)   z4 = y (joint 5)   z5 = x (joint 6)
A fixed tool transform Rz(pi) takes frame 6 to the URDF tool0 frame.
"""
import numpy as np

#                 alpha_i       a_i     d_i    theta_offset_i
DH_TABLE = np.array([
    [-np.pi / 2,  0.000,  0.265,  0.0],
    [ 0.0,        0.444,  0.000, -np.pi / 2],
    [-np.pi / 2,  0.110,  0.000,  0.0],
    [ np.pi / 2,  0.000,  0.470,  0.0],
    [-np.pi / 2,  0.080,  0.000,  0.0],
    [ 0.0,        0.000,  0.101,  0.0],
])

T_BASE = np.eye(4)                                   # base frame == frame 0
T_TOOL = np.eye(4)                                   # frame 6 -> tool0
T_TOOL[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)  # Rz(pi)


def dh_transform(theta, d, a, alpha):
    """Single-link standard-DH transform ^{i-1}T_i."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("DH table [alpha, a, d, theta_offset]:\n", DH_TABLE)
    print("\nT_tool =\n", T_TOOL)