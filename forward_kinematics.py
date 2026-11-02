"""
Forward kinematics of the ABB GoFa: DH, Product of Exponentials (PoE) and a
URDF reference, plus a main() that cross-validates them.

Error metrics (same as the slide "Validate before you trust the model"):
    e_p = || p_test - p_ref ||_2                         [m]
    e_R = arccos( (tr(R_ref^T R_test) - 1) / 2 )         [rad]
with the arccos argument clamped to [-1, 1].
"""
import os
import sys
import xml.etree.ElementTree as ET
import numpy as np

from models.screw import S_LIST, M, Q_LIMITS
from models.dh import DH_TABLE, T_BASE, T_TOOL, dh_transform

# ----------------------------------------------------------------------------
# Lie-group helpers
# ----------------------------------------------------------------------------
def skew(w):
    return np.array([[0, -w[2], w[1]],
                     [w[2], 0, -w[0]],
                     [-w[1], w[0], 0]], dtype=float)


def exp_twist(S, theta):
    """Matrix exponential exp([S] theta) in SE(3) for S = [w; v].

    Handles revolute (|w| = 1) and prismatic (w = 0, |v| = 1) screws.
    """
    w, v = np.asarray(S[:3], float), np.asarray(S[3:], float)
    T = np.eye(4)
    if np.linalg.norm(w) < 1e-12:                       # pure translation
        T[:3, 3] = v * theta
        return T
    W = skew(w)
    R = np.eye(3) + np.sin(theta) * W + (1 - np.cos(theta)) * W @ W   # Rodrigues
    G = (np.eye(3) * theta + (1 - np.cos(theta)) * W
         + (theta - np.sin(theta)) * W @ W)
    T[:3, :3] = R
    T[:3, 3] = G @ v
    return T


# ----------------------------------------------------------------------------
# 1) Product of Exponentials (space form):  T = e^{[S1]q1} ... e^{[S6]q6} M
# ----------------------------------------------------------------------------
def fk_poe(q, S_list=S_LIST, M_home=M):
    T = np.eye(4)
    for i in range(S_list.shape[1]):
        T = T @ exp_twist(S_list[:, i], q[i])
    return T @ M_home


# ----------------------------------------------------------------------------
# 2) Denavit-Hartenberg (standard convention); parameters live in models/dh_model.py
# ----------------------------------------------------------------------------
def fk_dh(q, dh_table=DH_TABLE, T_base=T_BASE, T_tool=T_TOOL):
    T = T_base.copy()
    for i in range(len(dh_table)):
        alpha, a, d, off = dh_table[i]
        T = T @ dh_transform(q[i] + off, d, a, alpha)
    return T @ T_tool


# ----------------------------------------------------------------------------
# 3) URDF reference: chain of <origin> * Rot(axis, q) over the movable joints
# ----------------------------------------------------------------------------
def _rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_angle_R(axis, angle):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


class UrdfFK:
    """Minimal serial-chain FK straight from the URDF (base_link -> tool0)."""

    def __init__(self, urdf_path, root="base_link", tip="tool0"):
        joints = {}
        for j in ET.parse(urdf_path).getroot().findall("joint"):
            org = j.find("origin")
            xyz = [float(x) for x in (org.get("xyz", "0 0 0") if org is not None else "0 0 0").split()]
            rpy = [float(x) for x in (org.get("rpy", "0 0 0") if org is not None else "0 0 0").split()]
            ax = j.find("axis")
            joints[j.find("child").get("link")] = dict(
                name=j.get("name"), type=j.get("type"),
                parent=j.find("parent").get("link"),
                xyz=xyz, rpy=rpy,
                axis=[float(x) for x in ax.get("xyz").split()] if ax is not None else None)
        chain, link = [], tip
        while link != root:                          # walk tip -> root
            chain.append(joints[link])
            link = joints[link]["parent"]
        self.chain = chain[::-1]
        self.n_dof = sum(j["type"] in ("revolute", "continuous") for j in self.chain)

    def fk(self, q):
        T, k = np.eye(4), 0
        for j in self.chain:
            To = np.eye(4)
            To[:3, :3] = _rpy_to_R(*j["rpy"])
            To[:3, 3] = j["xyz"]
            T = T @ To
            if j["type"] in ("revolute", "continuous"):
                Tq = np.eye(4)
                Tq[:3, :3] = _axis_angle_R(j["axis"], q[k])
                T = T @ Tq
                k += 1
        return T


# ----------------------------------------------------------------------------
# Error metrics
# ----------------------------------------------------------------------------
def position_error(T_test, T_ref):
    """e_p = ||p_test - p_ref||_2  [m]"""
    return float(np.linalg.norm(T_test[:3, 3] - T_ref[:3, 3]))


def rotation_error(T_test, T_ref):
    """e_R = arccos((tr(R_ref^T R_test) - 1)/2)  [rad], argument clamped to [-1, 1]"""
    c = (np.trace(T_ref[:3, :3].T @ T_test[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


def pose_errors(T_test, T_ref):
    return position_error(T_test, T_ref), rotation_error(T_test, T_ref)


# ----------------------------------------------------------------------------
# Main: cross-validation
# ----------------------------------------------------------------------------
def main(urdf_path="models/abb_gofa/crb15000_5_95.urdf",
         n_random=1000, seed=0, tol_p=1e-9, tol_R=1e-6):
    # tol_R is looser: arccos near 1 amplifies round-off (~sqrt(2*eps) ~ 3e-8 rad)
    rng = np.random.default_rng(seed)
    urdf = UrdfFK(urdf_path) if os.path.exists(urdf_path) else None
    if urdf is None:
        print(f"[warn] URDF not found at {urdf_path}; comparing DH vs PoE only.")

    lo, hi = Q_LIMITS[:, 0], Q_LIMITS[:, 1]
    span = hi - lo
    configs = {"zero": np.zeros(6)}
    for k in range(n_random):                                   # inside limits
        configs[f"rand_{k}"] = rng.uniform(lo, hi)
    for k in range(100):                                        # just outside range
        configs[f"outside_{k}"] = rng.uniform(lo - 0.1 * span, hi + 0.1 * span)

    # pairs: (name, test_fn, ref_fn); the "ref" in the formula is the second
    pairs = [("DH  vs PoE ", fk_dh, fk_poe)]
    if urdf is not None:
        pairs += [("DH  vs URDF", fk_dh, urdf.fk),
                  ("PoE vs URDF", fk_poe, urdf.fk)]

    print(f"{len(configs)} configurations "
          f"(1 zero, {n_random} random in limits, 100 just outside limits)\n")
    print(f"{'pair':<12}{'max e_p [m]':>14}{'max e_R [rad]':>16}{'zero e_p':>12}{'zero e_R':>12}   status")
    ok_all = True
    for name, f_test, f_ref in pairs:
        eps, eRs = [], []
        for q in configs.values():
            ep, eR = pose_errors(f_test(q), f_ref(q))
            eps.append(ep); eRs.append(eR)
        ep0, eR0 = eps[0], eRs[0]
        ok = max(eps) < tol_p and max(eRs) < tol_R
        ok_all &= ok
        print(f"{name:<12}{max(eps):>14.3e}{max(eRs):>16.3e}{ep0:>12.2e}{eR0:>12.2e}   "
              f"{'PASS' if ok else 'FAIL'}")

    print("\nZero-configuration tool0 pose (PoE):\n", np.round(fk_poe(np.zeros(6)), 6))
    print("\nOverall:", "ALL MODELS AGREE" if ok_all else "MISMATCH FOUND")
    return 0 if ok_all else 1


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "models/abb_gofa/crb15000_5_95.urdf"
    sys.exit(main(path))