"""
Screw-based Geometric Jacobian for ABB GoFa (CRB 15000-5/0.95).
Computes the spatial Jacobian in the world frame and the subchain Jacobian ^3 J_5.
"""

import numpy as np
from models.screw import S_LIST, OMEGA, R_AXIS, M
from forward_kinematics import fk_poe, exp_twist, skew


def compute_world_jacobian(q):
    """
    Computes the 6x6 geometric Jacobian J_0(q) expressed in the world frame,
    following the course convention [v; w] (linear velocity stacked above angular).
    """
    # 1. Get current end-effector position from PoE FK
    T_current = fk_poe(q)
    p_t = T_current[:3, 3]
    
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


def compute_subchain_jacobian_3J5(q):
    """
    Computes ^3 J_5: The wrist/subchain Jacobian up to joint 5, 
    expressed in frame 3 orientation. (Slide 34, Lec 5)
    """
    # 1. Get full world Jacobian
    J_0 = compute_world_jacobian(q)
    
    # 2. Compute rotation matrix R_03 from base to link 3 using PoE up to joint 3
    T_03 = np.eye(4)
    for i in range(3):
        T_03 = T_03 @ exp_twist(S_LIST[:, i], q[i])
    
    R_30 = T_03[:3, :3].T
    
    # 3. Build Adjoint rotation block to map spatial twist into frame 3
    Ad_30 = np.zeros((6, 6))
    Ad_30[:3, :3] = R_30
    Ad_30[3:, 3:] = R_30
    
    J_3 = Ad_30 @ J_0
    
    # 4. Return subchain up to joint 5 (first 5 columns)
    return J_3[:, :5]


def validate_jacobian(q, eps=1e-6):
    """
    Robustly validates the screw Jacobian against finite differences of PoE FK.
    """
    J_analytical = compute_world_jacobian(q)
    J_numerical = np.zeros((6, 6))
    T_base = fk_poe(q)
    
    for i in range(6):
        q_p = q.copy()
        q_p[i] += eps
        delta_T = np.linalg.inv(T_base) @ fk_poe(q_p)
        
        # Approximate spatial velocity twist vector via matrix logarithm
        from scipy.linalg import logm
        xi_mat = logm(delta_T) / eps
        w = np.array([xi_mat[2, 1], xi_mat[0, 2], xi_mat[1, 0]])
        v = xi_mat[:3, 3]
        
        J_numerical[:, i] = np.hstack([v, w])
        
    error = np.linalg.norm(J_analytical - J_numerical)
    print(f"Jacobian validation error at config: {error:.2e}")
    return error < 1e-4


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    q_test = np.array([0.1, -0.2, 0.5, 0.0, 1.0, -0.5])
    
    print("--- World Jacobian J_0 ---")
    print(compute_world_jacobian(q_test))
    
    print("\n--- Subchain Jacobian ^3 J_5 ---")
    print(compute_subchain_jacobian_3J5(q_test))
    
    print("\nRunning Validation...")
    validate_jacobian(q_test)