"""
Denavit-Hartenberg model of the ABB GoFa (CRB 15000-5/0.95).

    ^{J_i}T_{J_i+1} = Rz(theta_i + offset_i) Tz(d_i) Tx(a_i) Rx(alpha_i)

Frame J1 sits at the world origin (^W T_J1 = I), so the base height 0.265 m
is d in the J1 -> J2 row and the table has exactly one row per joint.

Frames at q = 0 (world axes):
    z_J1 = z (joint 1)   z_J2 = y (joint 2)   z_J3 = y (joint 3)
    z_J4 = x (joint 4)   z_J5 = y (joint 5)   z_J6 = x (joint 6)
The last row (d = 0.101, offset = pi) ends directly at the URDF tool0 frame
(z out of the flange along world x, x pointing down).
"""
import numpy as np
import sympy as sym

#    alpha_i      a_i     d_i    theta_offset_i
DH_TABLE = np.array([
    [-np.pi / 2,  0.000,  0.265,  0.0],         # J1 -> J2   (q1)
    [ 0.0,        0.444,  0.000, -np.pi / 2],   # J2 -> J3   (q2 - 90)
    [-np.pi / 2,  0.110,  0.000,  0.0],         # J3 -> J4   (q3)
    [ np.pi / 2,  0.000,  0.470,  0.0],         # J4 -> J5   (q4)
    [-np.pi / 2,  0.080,  0.000,  0.0],         # J5 -> J6   (q5)
    [ 0.0,        0.000,  0.101,  np.pi],       # J6 -> tool (q6 + 180)
])

DH_TABLE_sym = np.array([
    [-sym.pi / 2,  0.000,  0.265,  0.0],
    [ 0.0,        0.444,  0.000, -sym.pi / 2],
    [-sym.pi / 2,  0.110,  0.000,  0.0],
    [ sym.pi / 2,  0.000,  0.470,  0.0],
    [-sym.pi / 2,  0.080,  0.000,  0.0],
    [ 0.0,        0.000,  0.101,  sym.pi],
])

T_BASE = np.eye(4)                                   # ^W T_J1 = I
T_TOOL = np.eye(4)                                   # tool0 reached by last row (offset pi)

def dh_transform_q (q):
    """Get transform depend on the q"""
    return dh_transform(*DH_TABLE[q][::-1])

def dh_transform(theta, d, a, alpha):
    """Single-link DH transform ^{J_i}T_{J_i+1}."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])

def dh_frames(q):
    """[^W T_J1, ^W T_J2, ..., ^W T_J6, ^W T_tool]: frames after each DH row."""
    T, out = T_BASE.copy(), [T_BASE.copy()]
    for i in range(6):
        alpha, a, d, off = DH_TABLE[i]
        T = T @ dh_transform(q[i] + off, d, a, alpha)
        out.append(T.copy())
    return out

def dh_transform_symbolic(q: int):
    """Symbolic ^{J_q}T_{J_q+1} as a function of joint symbol q{q} (q is 1-based)."""
    alpha, a, d, offset = DH_TABLE_sym[q - 1]
    theta = sym.Symbol(f"q{q}") + offset

    # ^{J_i}T_{J_i+1} = Rz(theta) Tz(d) Tx(a) Rx(alpha)
    # With homogeneus matrices
    Rz = sym.diag(sym.rot_ccw_axis3(theta), 1)
    Tz = sym.eye(4); Tz[2, 3] = d
    Tx = sym.eye(4); Tx[0, 3] = a
    Rx = sym.diag(sym.rot_ccw_axis1(alpha), 1)

    return sym.simplify(Rz * Tz * Tx * Rx)

def dh_frames_symbolic():
    T, out = sym.eye(4), [sym.eye(4)]
    for i in range(6):
        T = T * dh_transform_symbolic(i + 1)
        out.append(T.copy())
    return out

if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    print("DH table [alpha, a, d, theta_offset]:\n", DH_TABLE)

    # self-check: symbolic chain matches numeric chain at a random q
    qs = np.random.uniform(-np.pi, np.pi, 6)
    T_sym, T_num = sym.eye(4), np.eye(4)
    for i in range(6):
        alpha, a, d, off = DH_TABLE[i]
        T_sym = T_sym * dh_transform_symbolic(i + 1)
        T_num = T_num @ dh_transform(qs[i] + off, d, a, alpha)
    T_eval = np.array(T_sym.subs({f"q{i+1}": qs[i] for i in range(6)}), dtype=float)
    assert np.allclose(T_eval, T_num), "symbolic FK != numeric FK"
    print("\nsymbolic FK matches numeric FK")