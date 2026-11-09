"""
Screw-based Geometric Jacobian for ABB GoFa (CRB 15000-5/0.95).
Computes the spatial Jacobian in the world frame and the subchain Jacobian ^3 J_5.
"""

import numpy as np
import sympy as sym
from models.dh import dh_frames, dh_frames_symbolic
from forward_kinematics import fk_poe
from scipy.spatial.transform import Rotation as Rot


def jacobian_0Jt(q):
    """
    6x6 geometric Jacobian J_0(q) in the world frame, [v; w] (linear on top).
    Joint i turns about w_{k-1} through p_{k-1}, both read from the DH frames.
    """
    F = dh_frames(q) # [0T0, 0T1, ..., 0T6]
    # Reference point: tool0
    p_t = F[6][:3, 3]

    J = np.zeros((6, 6))
    for k in range(6):
        w_k, p_k = F[k][:3, 2], F[k][:3, 3]                # joint i+1 axis = w of frame k
        J[:3, k] = np.cross(w_k, p_t - p_k)                # v_i = w_k x (p_t - o_k)
        J[3:, k] = w_k                                     # w_i = w_k
    return J

def jacobian_0Jt_sym():
    F = dh_frames_symbolic()
    p_t = F[6][:3, 3]
    
    J = sym.zeros(6, 6)
    for k in range(6):
        w_k, p_k = F[k][:3, 2], F[k][:3, 3]                # joint i+1 axis = w of frame k
        J[:3, k] = w_k.cross(p_t - p_k)                    # v_i = w_k x (p_t - o_k)
        J[3:, k] = w_k                                     # w_i = w_k
    return J

def jacobian_3J5(q):
    """
    ^3 J_5 (6x6, linear first): Jacobian of the point 5
    """
    F = dh_frames(q)
    T03_inv = np.linalg.inv(F[3])
    F3 = [T03_inv @ T for T in F] # Move all transform to joint 3

    # Final point: joint 5
    p_t = F3[5][:3, 3]

    J = np.zeros((6, 6))
    for k in range(6):
        w_k, p_k = F3[k][:3, 2], F3[k][:3, 3]              # joint i+1 axis = w of frame k
        J[:3, k] = np.cross(w_k, p_t - p_k)                # v_i = w_k x (p_t - o_k)
        J[3:, k] = w_k                                     # w_i = w_k
    return J

def jacobian_3J5_sym():
    F = dh_frames_symbolic()
    R, p = F[3][:3, :3], F[3][:3, 3]
    T03_inv = sym.eye(4)
    T03_inv[:3, :3] = R.T; T03_inv[:3, 3] = -R.T * p
    F3 = [sym.simplify(T03_inv * T) for T in F] # Move all transform to joint 3
    p_t = F3[5][:3, 3]

    J = sym.zeros(6, 6)
    for k in range(6):
        w_k, p_k = F3[k][:3, 2], F3[k][:3, 3]              # joint i+1 axis = w of frame k
        J[:3, k] = w_k.cross(p_t - p_k)                    # v_i = w_k x (p_t - o_k)
        J[3:, k] = w_k                                     # w_i = w_k
    return J

def validate_jacobian(q, eps=1e-6):
    """
    Central finite differences of PoE FK, in the WORLD frame (same frame as J_0):
    linear rows  = d p_t / d q_i,   angular rows = Log(R(q+e) R(q-e)^T) / (2 eps).
    """
    J_geometrical = jacobian_0Jt(q)
    J_numerical = np.zeros((6, 6))

    for i in range(6):
        e = np.zeros(6)
        e[i] = eps

        Tp, Tm = fk_poe(q + e), fk_poe(q - e)
        J_numerical[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * eps)
        J_numerical[3:, i] = Rot.from_matrix(Tp[:3, :3] @ Tm[:3, :3].T).as_rotvec() / (2 * eps)
    error = np.abs(J_geometrical - J_numerical).max()
    print(f"Jacobian validation error at config: {error:.2e}")
    return error < 1e-6



if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    q_test = np.array([0.1, -0.2, 0.5, 0.0, 1.0, -0.5])
    print("--- Angles ---")
    print(q_test)
    
    print("\n--- World Jacobian J_0 ---")
    jacobian = jacobian_0Jt(q_test)
    print(jacobian)
    det_j = np.linalg.det(jacobian)
    print("Determinant:", det_j)
    
    print("\n--- Subchain Jacobian ^3 J_5 ---")
    jacobian_3_to_5 = jacobian_3J5(q_test)
    print(jacobian_3_to_5)
    det_j35 = np.linalg.det(jacobian_3_to_5)
    print("Determinant:", det_j35)

    # Validate the 3J5
    print("\n Are they equal?", "True" if abs(det_j - det_j35) < 1e-4 else "False")
    
    print("\nRunning Validation...")
    validate_jacobian(q_test)

    # print(sym.pprint(jacobian_0Jt_sym()[:, :3]))
    # print("....")
    # print(sym.pprint(jacobian_3J5_sym()[:, :3]))