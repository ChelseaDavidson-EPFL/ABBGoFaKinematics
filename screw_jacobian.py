"""
Screw-based Geometric Jacobian for ABB GoFa (CRB 15000-5/0.95).
Computes the spatial Jacobian in the world frame and the subchain Jacobian ^3 J_5.
"""

import numpy as np
from models.screw import S_LIST, OMEGA, R_AXIS, M
from models.dh import DH_TABLE, T_BASE, dh_transform
from forward_kinematics import fk_poe, exp_twist, skew
from scipy.spatial.transform import Rotation as Rot


def compute_world_jacobian(q, p_ref=None):
    """
    Computes the 6x6 geometric Jacobian J_0(q) expressed in the world frame,
    following the course convention [v; w] (linear velocity stacked above angular).
    """
    # 1. Reference point: tool0 origin by default, or any world point p_ref
    p_t = fk_poe(q)[:3, 3] if p_ref is None else np.asarray(p_ref, float)
    
    J_v = np.zeros((3, 6))
    J_w = np.zeros((3, 6))
    
    # Cumulative transformation to track individual joint axis frames
    T_cum = np.eye(4)
    
    for i in range(6):
        # Transform initial axis orientation and point to current configuration
        z_i = T_cum[:3, :3] @ OMEGA[i]
        p_i = T_cum[:3, :3] @ R_AXIS[i] + T_cum[:3, 3]
        
        # Column components: v_i = z_i x (p_t - p_i), w_i = z_i (slide 27, Lec 5)
        J_v[:, i] = np.cross(z_i, p_t - p_i)
        J_w[:, i] = z_i
        
        # Update cumulative transformation for the next joint using your proven exp_twist
        T_cum = T_cum @ exp_twist(S_LIST[:, i], q[i])
        
    return np.vstack([J_v, J_w])


def dh_frames(q):
    """[0T0, 0T1, ..., 0T6]: standard-DH frames (after each DH row)."""
    T, out = T_BASE.copy(), [T_BASE.copy()]
    for i in range(6):
        alpha, a, d, off = DH_TABLE[i]
        T = T @ dh_transform(q[i] + off, d, a, alpha)
        out.append(T.copy())
    return out
 
 
def compute_subchain_jacobian_3J5(q, point="O5"):
    """
    ^3 J_5 (6x6, linear first): Jacobian of the point O5 = origin of DH frame 5
    (point="O4": origin of DH frame 4 = intersection of axes 4 and 5), with both
    blocks rotated into the DH frame 3:  ^3J_5 = blkdiag(R03^T, R03^T) ^0J_5.
    """
    F = dh_frames(q)
    p = F[5][:3, 3] if point == "O5" else F[4][:3, 3]
    R30 = F[3][:3, :3].T
    J_0 = compute_world_jacobian(q, p_ref=p)
    Z = np.zeros((3, 3))
    return np.block([[R30, Z], [Z, R30]]) @ J_0



def validate_jacobian(q, eps=1e-6):
    """
    Central finite differences of PoE FK, in the WORLD frame (same frame as J_0):
    linear rows  = d p_t / d q_i,   angular rows = Log(R(q+e) R(q-e)^T) / (2 eps).
    """
    J_analytical = compute_world_jacobian(q)
    J_numerical = np.zeros((6, 6))

    for i in range(6):
        e = np.zeros(6)
        e[i] = eps

        Tp, Tm = fk_poe(q + e), fk_poe(q - e)
        J_numerical[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        J_numerical[3:, i] = Rot.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * eps)
    error = np.abs(J_analytical - J_numerical).max()
    print(f"Jacobian validation error at config: {error:.2e}")
    return error < 1e-6



if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    q_test = np.array([0.1, -0.2, 0.5, 0.0, 1.0, -0.5])
    print("--- Angles ---")
    print(q_test)
    
    print("\n--- World Jacobian J_0 ---")
    print(compute_world_jacobian(q_test))
    
    print("\n--- Subchain Jacobian ^3 J_5 ---")
    print(compute_subchain_jacobian_3J5(q_test))
    
    print("\nRunning Validation...")
    validate_jacobian(q_test)