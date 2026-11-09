"""
Standard Denavit-Hartenberg model of the ABB GoFa (CRB 15000-5/0.95).

    ^{i-1}T_i = Rz(theta_i + offset_i) Tz(d_i) Tx(a_i) Rx(alpha_i)

Frames at q = 0 (world axes):
    z0 = z (joint 1)   z1 = y (joint 2)   z2 = y (joint 3)
    z3 = x (joint 4)   z4 = y (joint 5)   z5 = x (joint 6)
A fixed tool transform Rz(pi) takes frame 6 to the URDF tool0 frame.
"""
import numpy as np
import sympy as sym

#    alpha_i      a_i     d_i    theta_offset_i
DH_TABLE = np.array([
    [-np.pi / 2,  0.000,  0.265,  0.0],
    [ 0.0,        0.444,  0.000, -np.pi / 2],
    [-np.pi / 2,  0.110,  0.000,  0.0],
    [ np.pi / 2,  0.000,  0.470,  0.0],
    [-np.pi / 2,  0.080,  0.000,  0.0],
    [ 0.0,        0.000,  0.101,  0.0],
])

DH_TABLE_sym = np.array([
    [-sym.pi / 2,  0.000,  0.265,  0.0],
    [ 0.0,        0.444,  0.000, -sym.pi / 2],
    [-sym.pi / 2,  0.110,  0.000,  0.0],
    [ sym.pi / 2,  0.000,  0.470,  0.0],
    [-sym.pi / 2,  0.080,  0.000,  0.0],
    [ 0.0,        0.000,  0.101,  0.0],
])

T_BASE = np.eye(4)                                   # base frame == frame 0
T_TOOL = np.eye(4)                                   # frame 6 -> tool0
T_TOOL[:3, :3] = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=float)  # Rz(pi)

def dh_transform_q (q):
    """Get transform depend on the q"""
    return dh_transform(*DH_TABLE[q][::-1])

def dh_transform(theta, d, a, alpha):
    """Single-link standard-DH transform ^{i-1}T_i."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st * ca,  st * sa, a * ct],
                     [st,  ct * ca, -ct * sa, a * st],
                     [0.0,      sa,       ca,      d],
                     [0.0,     0.0,      0.0,    1.0]])

def dh_frames(q):
    """[0T0, 0T1, ..., 0T6]: standard-DH frames (after each DH row)."""
    T, out = T_BASE.copy(), [T_BASE.copy()]
    for i in range(6):
        alpha, a, d, off = DH_TABLE[i]
        T = T @ dh_transform(q[i] + off, d, a, alpha)
        out.append(T.copy())
    return out

def dh_transform_symbolic(q: int):
    """Symbolic ^{q-1}T_q as a function of joint symbol q{q} (q is 1-based)."""
    alpha, a, d, offset = DH_TABLE_sym[q - 1]
    theta = sym.Symbol(f"q{q}") + offset

    # ^{i-1}T_i = Rz(theta) Tz(d) Tx(a) Rx(alpha)
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
    print("\nT_tool =\n", T_TOOL)

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
