"""
Expected output 4 -- (q2, q3) singularity slice of the full ABB GoFa and the
distribution of its IK solutions, made interactive like Lecture 06, slide 17
("Explore the ABB IRB 4600 singularities and IK distribution").

    Left   : joint-space slice (q2, q3) [rad] with the singular curves of the
             FULL robot and the (q2, q3) of every IK solution (numbered markers).
    Middle : wrist-center section 0p5 = (x, 0, z), x >= 0, with the critical
             curves (images of the singular curves).  Crossing a curve gains or
             loses IK solutions; the shading is the number of regular IKs.
    Right  : the selected IK drawn as a stick robot.

Interaction
    * click in the middle plot      -> new wrist center, IK solved with the supplied solver
    * click a marker in the left plot -> select that IK
    * "Next IK" button / right arrow -> cycle through IKs (left arrow goes back)
    * "Random target" button / r     -> new random wrist center inside the workspace

Set-up (as in the lecture): the tool z-axis points vertically down,
0R_tool = diag(1, -1, -1).  The "wrist center" is O5 = origin of DH frame 5,
the point of axis 6 on the common normal with axis 5 (the point used for 3J5).
With the tool z-axis = axis 6 fixed, O5 = p_tool - d6 * z_tool is fixed by the
pose, so the tool target is p_tool = O5 + (0, 0, -d6).

Why a (q2, q3) slice is exact for this family
---------------------------------------------
Tool-down poses do not depend on q1 (rotation about the base axis) nor on q6
(rotation about the vertical tool axis), so (rho, z) of O5 is a function of
(q2, q3) and a discrete wrist branch.  Two families of configurations exist:

  A  q4 in {0, pi}: axis 5 is normal to the arm plane and the a5 = 0.080 m
     offset is radial:  O5 = O4 + sigma * a5 * (forward), sigma = +-1.
     On A the determinant factorises exactly like the IRB 4600:
         det(0J6) = -a2 * F * G * sin(q5),       sin(q5) = sigma * cos(q2+q3)
         F = a3 sin q3 + d4 cos q3                       (elbow)
         G = a2 sin q2 + a3 sin q23 + d4 cos q23 + sigma a5  (shoulder: O5 on base axis)
         sin q5 = 0  <=>  q2 + q3 = +-pi/2               (wrist: forearm vertical)
  B  forearm vertical (q2 + q3 = +-pi/2), q5 in {0, pi}, q4 free.  Because of the
     a5 offset, axes 4 and 6 are parallel but NOT coincident, so these are
     regular configurations (det J != 0).  Their (q2, q3) lie on the wrist lines.

Regular IK count (checked against the solver on random targets):
    A: 4 solutions inside each annulus  a2 - L3 < |O5 - (+-a5, d1)| < a2 + L3
    B: 2 solutions for every q2 with z = d1 + a2 cos q2 - tau d4 and
       | |r| - a5 | < rho < |r| + a5,  r = a2 sin q2 + tau a3,  tau = sin(q2+q3) = +-1
so the GoFa has 0, 4, 6, 8 or 10 regular IKs for a tool-down pose.

All angles are in radians.  Run from the repository root:
    python singularity_slice_ik.py            (interactive figure)
    python singularity_slice_ik.py --verify   (compare analytic count with the solver)
"""
import json
import os
import xml.etree.ElementTree as ET
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from models.dh import DH_TABLE
from models.screw import Q_LIMITS
from forward_kinematics import fk_poe, pose_errors, _rpy_to_R, _axis_angle_R
from screw_jacobian import compute_world_jacobian, dh_frames

# ============================================================================
# USER SETTINGS
# ============================================================================
IK_PLATFORM = "linux"            # "linux" or "windows"
IK_ROOT = Path(__file__).resolve().parent / "abbGofa_ik"   # folder from abbGofa_ik.zip
RANDOM_SEED = None               # int for a reproducible first target, None = random
URDF_PATH = Path(__file__).resolve().parent / "models" / "abb_gofa" / "crb15000_5_95.urdf"
SHOW_MESHES = True               # draw the URDF STL meshes in the 3D view (False = stick robot)
MESH_CELL = 0.008                # [m] vertex-clustering size used to lighten the STLs (0 = full mesh)
# ============================================================================

IK_BINARY = {
    "linux": IK_ROOT / "crb15000-ik-linux-x86_64" / "crb15000-ik",
    "windows": IK_ROOT / "crb15000-ik-windows-x86_64" / "crb15000-ik.exe",
}[IK_PLATFORM.lower()]

# --- geometry, read from the DH table -------------------------------------
D1 = DH_TABLE[0, 2]          # 0.265  base -> shoulder
A2 = DH_TABLE[1, 1]          # 0.444  upper arm
A3 = DH_TABLE[2, 1]          # 0.110  elbow offset
D4 = DH_TABLE[3, 2]          # 0.470  forearm
A5 = DH_TABLE[4, 1]          # 0.080  offset between axes 5 and 6
D6 = DH_TABLE[5, 2]          # 0.101  O5 -> flange along axis 6
L3 = np.hypot(A3, D4)        # elbow -> O4 distance
R_IN, R_OUT = abs(A2 - L3), A2 + L3

# --- fixed tool orientation: z-axis down ----------------------------------
R_TOOL = np.diag([1.0, -1.0, -1.0])
QUAT_TOOL = [0.0, 1.0, 0.0, 0.0]          # (qw, qx, qy, qz) of diag(1,-1,-1)

Q3_LO, Q3_HI = Q_LIMITS[2]               # q3 limits [-3.927, 1.484]
TOL_SING = 1e-6                           # |sin q5| below this -> family B

# Plot ranges of the wrist-center section
RHO_MAX = R_OUT + A5 + 0.06
Z_MIN, Z_MAX = D1 - R_OUT - 0.08, D1 + R_OUT + 0.08

# Branch colours, keyed by (S, E, sigma) like the lecture's IK 1..8
BRANCH_ORDER = [(s, e, g) for s in (1, -1) for e in (1, -1) for g in (1, -1)]
BRANCH_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                 "#9467bd", "#8c564b", "#e377c2", "#17becf"]
B_COLOR = "#222222"
C_ELBOW, C_SHOULDER, C_WRIST, C_BFOLD = "#e6550d", "#3182bd", "#31a354", "#31a354"


def wrap(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


# ============================================================================
# IK solver wrapper
# ============================================================================
def _ensure_executable(path):
    if not path.exists():
        raise FileNotFoundError(
            f"IK solver not found at {path}.\nUnzip abbGofa_ik.zip next to this script "
            f"or edit IK_ROOT / IK_PLATFORM at the top of the file.")
    if IK_PLATFORM.lower() == "linux" and not os.access(path, os.X_OK):
        path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def tool_target(w):
    """Wrist center O5 (x, 0, z) -> 4x4 tool0 target with the tool z-axis down."""
    T = np.eye(4)
    T[:3, :3] = R_TOOL
    T[:3, 3] = np.asarray(w, float) + D6 * R_TOOL[:, 2]
    return T


def solve_ik(w):
    """All IK solutions (radians) returned by the supplied solver for wrist center w."""
    _ensure_executable(IK_BINARY)
    T = tool_target(w)
    request = {"pose": [*T[:3, 3], *QUAT_TOOL], "frame": "tool0", "respect_limits": False}
    res = subprocess.run([str(IK_BINARY), "solve-pose"], input=json.dumps(request),
                         capture_output=True, text=True)
    if not res.stdout.strip():
        raise RuntimeError(f"IK solver gave no output (exit {res.returncode}): {res.stderr}")
    report = json.loads(res.stdout)
    if res.returncode != 0:
        raise RuntimeError(report.get("error", res.stderr))
    return report["solutions"], T


# ============================================================================
# Independent analysis of every returned configuration
# ============================================================================
def analyse_solution(sol, T_target):
    q = np.asarray(sol["q"], float)
    F = dh_frames(q)
    O4, O5 = F[4][:3, 3], F[5][:3, 3]
    fwd = np.array([np.cos(q[0]), np.sin(q[0]), 0.0])         # arm-plane forward direction

    J = compute_world_jacobian(q)
    sv = np.linalg.svd(J, compute_uv=False)
    ep, eR = pose_errors(fk_poe(q), T_target)                  # our FK vs the target

    s5 = np.sin(q[4])
    family = "B" if abs(s5) < TOL_SING else "A"
    S = int(np.sign(O5 @ fwd))                                 # shoulder: sign G
    E = int(np.sign(A3 * np.sin(q[2]) + D4 * np.cos(q[2])))    # elbow: sign F
    W = int(np.sign(s5)) if family == "A" else 0               # wrist: sign sin q5
    sigma = float((O5 - O4) @ fwd / A5)                        # +-1 on family A
    sig = int(np.sign(sigma)) if family == "A" else 0
    lo, hi = Q_LIMITS[:, 0] - 1e-12, Q_LIMITS[:, 1] + 1e-12
    return dict(q=q, family=family, S=S, E=E, W=W, sigma=sigma, sig=sig,
                det=float(np.linalg.det(J)), smin=float(sv[-1]), cond=float(sv[0] / sv[-1]),
                G=float(O5 @ fwd), F=float(A3 * np.sin(q[2]) + D4 * np.cos(q[2])), s5=float(s5),
                in_limits=bool(np.all((q >= lo) & (q <= hi))), ep=ep, eR=eR,
                frames=F, T=fk_poe(q))


def order_solutions(sols):
    """Sort like the lecture table: family A by (S, E, sigma) = (+,+,+) ... (-,-,-), then B."""
    def key(s):
        if s["family"] == "A":
            return (0, BRANCH_ORDER.index((s["S"], s["E"], s["sig"])), 0.0)
        return (1, -s["S"], s["q"][0])
    return sorted(sols, key=key)


def solution_style(s):
    if s["family"] == "A":
        return BRANCH_COLORS[BRANCH_ORDER.index((s["S"], s["E"], s["sig"]))], ("o" if s["W"] > 0 else "s")
    return B_COLOR, "D"


# ============================================================================
# Analytic singular curves (joint space) and critical curves (wrist section)
# ============================================================================
def regular_ik_count(rho, z):
    """Number of regular tool-down IK solutions for wrist center (rho, 0, z)."""
    rho, z = np.asarray(rho, float), np.asarray(z, float)
    n = np.zeros(np.broadcast(rho, z).shape, int)
    for c in (A5, -A5):                                        # family A
        d = np.hypot(rho - c, z - D1)
        n += 4 * ((d > R_IN) & (d < R_OUT))
    for tau in (1, -1):                                        # family B
        c2 = (z - D1 + tau * D4) / A2
        ok = np.abs(c2) < 1
        s2 = np.sqrt(np.clip(1 - c2 ** 2, 0, None))
        for sgn in (1, -1):
            r = np.abs(A2 * sgn * s2 + tau * A3)
            n += 2 * (ok & (rho > np.abs(r - A5)) & (rho < r + A5))
    return n


def critical_curves():
    """Critical curves in (rho, z) as lists of (rho_array, z_array)."""
    t = np.linspace(-np.pi, np.pi, 1441)
    elbow = []
    for c in (A5, -A5):                       # F = 0: elbow stretched / folded
        for R in (R_OUT, R_IN):
            x, z = c + R * np.cos(t), D1 + R * np.sin(t)
            x = np.where(x >= 0, x, np.nan)
            elbow.append((x, z))
    shoulder = [(np.array([0.0, 0.0]), np.array([Z_MIN, Z_MAX]))]   # G = 0: O5 on the base axis
    wrist, bfold = [], []
    for tau in (1, -1):                       # images of the wrist lines q2 + q3 = tau*pi/2
        r = A2 * np.sin(t) + tau * A3
        z = D1 + A2 * np.cos(t) - tau * D4
        wrist.append((np.abs(r) + A5, z))
        wrist.append((np.abs(np.abs(r) - A5), z))
        for c2 in (1, -1):                    # family-B folds at sin q2 = 0
            zf = D1 + A2 * c2 - tau * D4
            bfold.append((np.array([abs(A3 - A5), A3 + A5]), np.array([zf, zf])))
    return elbow, shoulder, wrist, bfold


def joint_space_curves(n=721):
    q2 = np.linspace(-np.pi, np.pi, n)
    Q2, Q3 = np.meshgrid(q2, q2)
    Gs = {s: A2 * np.sin(Q2) + A3 * np.sin(Q2 + Q3) + D4 * np.cos(Q2 + Q3) + s * A5
          for s in (1, -1)}
    q3_elbow = wrap(np.arctan2(-D4, A3) + np.array([0, np.pi]))
    return Q2, Q3, Gs, q3_elbow


def random_wrist_center(rng):
    while True:
        rho, z = rng.uniform(0.02, RHO_MAX), rng.uniform(Z_MIN, Z_MAX)
        if regular_ik_count(rho, z) > 0:
            return np.array([rho, 0.0, z])


# ============================================================================
# URDF meshes for the 3D view
# ============================================================================
def _read_stl(path):
    """Triangles (n, 3, 3) of a binary or ASCII STL file."""
    data = Path(path).read_bytes()
    if len(data) >= 84:
        n = int(np.frombuffer(data, "<u4", 1, 80)[0])
        if 84 + 50 * n == len(data):
            rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
            return np.frombuffer(data, rec, n, 84)["v"].astype(float)
    pts = [list(map(float, ln.split()[1:4])) for ln in data.decode(errors="ignore").splitlines()
           if ln.strip().startswith("vertex")]
    return np.asarray(pts, float).reshape(-1, 3, 3)


def _decimate(tris, cell):
    """Vertex clustering: snap vertices to a grid of size `cell`, drop collapsed triangles."""
    if cell <= 0:
        return tris
    v = tris.reshape(-1, 3)
    key = np.round(v / cell).astype(np.int64)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    mean = np.zeros((len(uniq), 3))
    np.add.at(mean, inv, v)
    mean /= np.bincount(inv, minlength=len(uniq))[:, None]
    f = inv.reshape(-1, 3)
    keep = (f[:, 0] != f[:, 1]) & (f[:, 1] != f[:, 2]) & (f[:, 0] != f[:, 2])
    f = np.unique(np.sort(f[keep], axis=1), axis=0, return_index=True)[1]
    faces = inv.reshape(-1, 3)[keep][np.sort(f)]
    return mean[faces]


class UrdfMeshModel:
    """Link meshes, colours and per-link FK read straight from the URDF."""

    def __init__(self, urdf_path, cell=MESH_CELL):
        root = ET.parse(urdf_path).getroot()
        mesh_dir = Path(urdf_path).parent
        self.links = {}                                    # name -> (local triangles, rgba)
        for link in root.findall("link"):
            vis = link.find("visual")
            if vis is None or vis.find("geometry/mesh") is None:
                continue
            fname = Path(vis.find("geometry/mesh").get("filename").split("://")[-1]).name
            path = mesh_dir / fname
            if not path.exists():
                print(f"[warn] mesh {fname} not found in {mesh_dir}; link skipped")
                continue
            tris = _decimate(_read_stl(path), cell)
            org = vis.find("origin")
            if org is not None:                           # visual origin (none in this URDF)
                R = _rpy_to_R(*map(float, org.get("rpy", "0 0 0").split()))
                tris = tris @ R.T + np.array(list(map(float, org.get("xyz", "0 0 0").split())))
            col = vis.find("material/color")
            rgba = (np.array(list(map(float, col.get("rgba").split())))
                    if col is not None else np.array([0.7, 0.7, 0.7, 1.0]))
            self.links[link.get("name")] = (tris, rgba)
        self.joints = []                                   # tree order from the root link
        js = {j.find("child").get("link"): j for j in root.findall("joint")}
        parents = {j.find("parent").get("link") for j in js.values()}
        self.root = (parents - set(js)).pop()
        order, frontier = [], [self.root]
        while frontier:
            p = frontier.pop(0)
            for c, j in js.items():
                if j.find("parent").get("link") == p:
                    order.append(j)
                    frontier.append(c)
        for j in order:
            org = j.find("origin")
            T0 = np.eye(4)
            if org is not None:
                T0[:3, :3] = _rpy_to_R(*map(float, org.get("rpy", "0 0 0").split()))
                T0[:3, 3] = list(map(float, org.get("xyz", "0 0 0").split()))
            ax = j.find("axis")
            self.joints.append(dict(parent=j.find("parent").get("link"),
                                    child=j.find("child").get("link"), type=j.get("type"), T0=T0,
                                    axis=(np.array(list(map(float, ax.get("xyz").split())))
                                          if ax is not None else None)))

    def link_transforms(self, q):
        T = {self.root: np.eye(4)}
        k = 0
        for j in self.joints:
            Tj = T[j["parent"]] @ j["T0"]
            if j["type"] in ("revolute", "continuous"):
                Tq = np.eye(4)
                Tq[:3, :3] = _axis_angle_R(j["axis"], q[k])
                Tj = Tj @ Tq
                k += 1
            T[j["child"]] = Tj
        return T

    def world_triangles(self, q, light=(0.4, -0.6, 0.7)):
        """All triangles in the world frame and their Lambert-shaded face colours."""
        Ts = self.link_transforms(q)
        tris, cols = [], []
        lvec = np.asarray(light, float) / np.linalg.norm(light)
        for name, (local, rgba) in self.links.items():
            T = Ts[name]
            wt = local @ T[:3, :3].T + T[:3, 3]
            n = np.cross(wt[:, 1] - wt[:, 0], wt[:, 2] - wt[:, 0])
            n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-15
            shade = 0.35 + 0.65 * np.abs(n @ lvec)
            c = np.clip(rgba[None, :3] * shade[:, None] * 1.15, 0, 1)
            tris.append(wt)
            cols.append(c)
        return np.concatenate(tris), np.concatenate(cols)


def load_mesh_model():
    if not SHOW_MESHES:
        return None
    try:
        model = UrdfMeshModel(URDF_PATH)
        return model if model.links else None
    except Exception as exc:                              # fall back to the stick robot
        print(f"[warn] could not load URDF meshes ({exc}); using the stick robot")
        return None


# ============================================================================
# Interactive figure
# ============================================================================
class SliceExplorer:
    def __init__(self):
        self.rng = np.random.default_rng(RANDOM_SEED)
        self.sols, self.sel, self.w = [], 0, None
        self.mesh = load_mesh_model()
        self.view = (18, -70)
        self.alpha = 1.0
        self.fig = plt.figure(figsize=(17, 10.5))
        self.fig.canvas.manager.set_window_title("ABB GoFa - singularities and IK distribution")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 0.40], width_ratios=[1, 1.1, 0.95],
                                   left=0.05, right=0.98, top=0.89, bottom=0.01,
                                   wspace=0.18, hspace=0.36)
        self.axq = self.fig.add_subplot(gs[0, 0])
        self.axw = self.fig.add_subplot(gs[0, 1])
        self.axr = self.fig.add_subplot(gs[0, 2], projection="3d")
        self.axt = self.fig.add_subplot(gs[1, :])
        self.axt.axis("off")
        self.fig.text(0.05, 0.965, "Explore the ABB GoFa singularities and IK distribution",
                      fontsize=15, fontweight="bold", va="center")
        self.fig.text(0.05, 0.935,
                      r"Tool z-axis vertically down, $^0R_t=\mathrm{diag}(1,-1,-1)$. Select a wrist "
                      r"center $^0\mathbf{p}_5=(x,0,z)$ and compare its IK branches.   "
                      r"On the $q_4\in\{0,\pi\}$ branches:  $\det({}^0J_6)=-a_2\,F\,G\,\sin q_5$,  "
                      r"$\sin q_5=\sigma\cos(q_2+q_3)$",
                      fontsize=10.5, va="center", color="0.2")
        self._draw_joint_space()
        self._draw_section()
        bx1 = self.fig.add_axes([0.80, 0.948, 0.085, 0.035])
        bx2 = self.fig.add_axes([0.895, 0.948, 0.085, 0.035])
        self.b_rand, self.b_next = Button(bx1, "Random target"), Button(bx2, "Next IK")
        self.b_rand.on_clicked(lambda _: self.set_target(random_wrist_center(self.rng)))
        self.b_next.on_clicked(lambda _: self.select(self.sel + 1))
        if self.mesh is not None:                     # "STL opacity" slider, as in the lecture
            sx = self.fig.add_axes([0.80, 0.335, 0.14, 0.02])
            self.s_alpha = Slider(sx, "STL opacity", 0.0, 1.0, valinit=1.0, valstep=0.05)
            self.s_alpha.on_changed(self._set_alpha)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.set_target(random_wrist_center(self.rng))

    # --- static layers -------------------------------------------------------
    def _draw_joint_space(self):
        ax = self.axq
        Q2, Q3, Gs, q3_elbow = joint_space_curves()
        # joint-limit gap of q3 (wrapped): (1.484, 2.356)
        ax.axhspan(Q3_HI, Q3_LO + 2 * np.pi, color="0.85", zorder=0)
        for q3e in q3_elbow:
            ax.axhline(q3e, color=C_ELBOW, lw=2)
        ax.contour(Q2, Q3, Gs[1], levels=[0], colors=C_SHOULDER, linewidths=2)
        ax.contour(Q2, Q3, Gs[-1], levels=[0], colors=C_SHOULDER, linewidths=2, linestyles="--")
        q2 = np.linspace(-np.pi, np.pi, 3)
        for k in range(-3, 4):                    # wrist lines q2 + q3 = pi/2 + k*pi
            ax.plot(q2, np.pi / 2 + k * np.pi - q2, color=C_WRIST, lw=2)
        ticks = np.linspace(-np.pi, np.pi, 5)
        labels = [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"]
        ax.set_xticks(ticks, labels)
        ax.set_yticks(ticks, labels)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect("equal")
        ax.set_xlabel(r"$q_2$ [rad]")
        ax.set_ylabel(r"$q_3$ [rad]  (wrapped to $[-\pi,\pi)$)")
        ax.set_title("full-robot singularities · $(q_2,q_3)$ slice", fontsize=11)
        handles = [
            Line2D([], [], color=C_SHOULDER, lw=2, label=r"shoulder $G=0$, $\sigma=+1$"),
            Line2D([], [], color=C_SHOULDER, lw=2, ls="--", label=r"shoulder $G=0$, $\sigma=-1$"),
            Line2D([], [], color=C_ELBOW, lw=2, label=r"elbow $F=a_3s_3+d_4c_3=0$"),
            Line2D([], [], color=C_WRIST, lw=2, label=r"wrist $\sin q_5=0$ ($q_2+q_3=\pm\pi/2$)"),
            plt.Rectangle((0, 0), 1, 1, color="0.85", label=r"$q_3$ outside joint limits"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=2, fontsize=8.5, frameon=False, handlelength=2.2)
        self.q_artists = []

    def _draw_section(self):
        ax = self.axw
        rr, zz = np.meshgrid(np.linspace(0, RHO_MAX, 700), np.linspace(Z_MIN, Z_MAX, 700))
        cnt = regular_ik_count(rr, zz)
        levels = [0, 4, 6, 8, 10]
        cmap = ListedColormap(["#ffffff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6"])
        norm = BoundaryNorm([-1, 2, 5, 7, 9, 11], cmap.N)
        im = ax.imshow(cnt, origin="lower", extent=[0, RHO_MAX, Z_MIN, Z_MAX],
                       cmap=cmap, norm=norm, interpolation="nearest", zorder=0)
        cb = self.fig.colorbar(im, ax=ax, ticks=levels, fraction=0.04, pad=0.02)
        cb.set_label("regular IK solutions")
        elbow, shoulder, wrist, bfold = critical_curves()
        for x, z in elbow:
            ax.plot(x, z, color=C_ELBOW, lw=2)
        for x, z in shoulder:
            ax.plot(x, z, color=C_SHOULDER, lw=3)
        for x, z in wrist:
            ax.plot(x, z, color=C_WRIST, lw=1.6)
        for x, z in bfold:
            ax.plot(x, z, color=C_BFOLD, lw=1.6, ls=":")
        ax.plot(0, D1, "k+", ms=10)
        ax.set_xlim(-0.015, RHO_MAX)
        ax.set_ylim(Z_MIN, Z_MAX)
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]  (y = 0, x ≥ 0)")
        ax.set_ylabel("z [m]")
        ax.set_title(r"wrist-center section $^0\mathbf{p}_5=(x,0,z)$ · click to solve IK", fontsize=11)
        handles = [
            Line2D([], [], color=C_SHOULDER, lw=3, label=r"shoulder $G=0$: O$_5$ on base axis (±4)"),
            Line2D([], [], color=C_ELBOW, lw=2, label=r"elbow $F=0$ (±4 IKs)"),
            Line2D([], [], color=C_WRIST, lw=1.6, label=r"wrist $\sin q_5=0$ image (±2 IKs)"),
            Line2D([], [], color=C_BFOLD, lw=1.6, ls=":", label=r"forearm-vertical fold $\sin q_2=0$ (±2)"),
        ]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.10),
                  ncol=2, fontsize=8.5, frameon=False, handlelength=2.2)
        self.w_marker, = ax.plot([], [], "*", color="gold", mec="k", ms=18, zorder=5)

    # --- dynamic layers ------------------------------------------------------
    def set_target(self, w):
        self.w = np.asarray(w, float)
        self.w_marker.set_data([self.w[0]], [self.w[2]])
        try:
            raw, T = solve_ik(self.w)
            self.error = None
        except Exception as exc:                      # solver error, e.g. exactly on the base axis
            raw, T, self.error = [], tool_target(self.w), str(exc)
        self.T_target = T
        self.sols = order_solutions([analyse_solution(s, T) for s in raw])
        self.sel = 0
        self._draw_markers()
        self.select(0)

    def _draw_markers(self):
        for a in self.q_artists:
            a.remove()
        self.q_artists = []
        for i, s in enumerate(self.sols):
            c, m = solution_style(s)
            x, y = wrap(s["q"][1]), wrap(s["q"][2])
            face = c if s["in_limits"] else "white"
            self.q_artists.append(self.axq.scatter(x, y, s=150, marker=m, c=face, edgecolors=c,
                                                   linewidths=2.2, zorder=4))
            self.q_artists.append(self.axq.annotate(str(i + 1), (x, y), xytext=(7, 6),
                                                    textcoords="offset points", fontsize=10,
                                                    fontweight="bold", color=c, zorder=5))
        self.sel_ring = self.axq.scatter([], [], s=420, facecolors="none",
                                         edgecolors="k", linewidths=1.5, zorder=3)
        self.q_artists.append(self.sel_ring)

    def select(self, i):
        if self.sols:
            self.sel = i % len(self.sols)
            s = self.sols[self.sel]
            self.sel_ring.set_offsets([[wrap(s["q"][1]), wrap(s["q"][2])]])
        else:
            self.sel_ring.set_offsets(np.empty((0, 2)))
        self._draw_robot()
        self._write_status()
        self.fig.canvas.draw_idle()

    def _draw_robot(self):
        ax = self.axr
        if ax.has_data():                             # keep the user's rotation of the 3D view
            self.view = (ax.elev, ax.azim)
        ax.cla()
        ax.set_title("selected IK" + (f" {self.sel + 1}" if self.sols else "") +
                     (" · URDF meshes" if self.mesh is not None else ""), fontsize=11)
        self.mesh_coll = None
        lim = 0.75
        ax.plot([-lim, lim], [0, 0], [0, 0], color="0.75", lw=0.8)
        ax.plot([0, 0], [-lim, lim], [0, 0], color="0.75", lw=0.8)
        if self.sols:
            s = self.sols[self.sel]
            c, _ = solution_style(s)
            q = s["q"]
            if self.mesh is not None:
                tris, cols = self.mesh.world_triangles(q)
                self.mesh_coll = Poly3DCollection(tris, facecolors=cols, edgecolors="none",
                                                  linewidths=0, alpha=self.alpha)
                ax.add_collection3d(self.mesh_coll)
                lw, ms = 2.5, 4                       # thin skeleton on top of the meshes
            else:
                ax.plot([0, 0], [0, 0], [0, D1], color="0.4", lw=9, alpha=0.5)
                lw, ms = 5, 6
            pts = np.array([[0, 0, 0]] + [Fk[:3, 3] for Fk in s["frames"][1:]])
            ax.plot(*pts.T, "-o", color=c, lw=lw, ms=ms, mfc="k", solid_capstyle="round",
                    zorder=10, alpha=0.35 if (self.mesh is not None and self.alpha > 0.6) else 1)
            T = s["T"]
            for k, col in enumerate("rgb"):
                a = T[:3, 3] + 0.12 * T[:3, k]
                ax.plot(*np.c_[T[:3, 3], a], color=col, lw=2.5, zorder=11)
        ax.scatter(*self.w, marker="*", s=160, c="gold", edgecolors="k", depthshade=False,
                   zorder=12)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-0.45, 1.05)
        ax.set_box_aspect((2 * lim, 2 * lim, 1.5), zoom=1.15)
        ax.view_init(elev=self.view[0], azim=self.view[1])
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

    def _set_alpha(self, val):
        self.alpha = float(val)
        if self.mesh_coll is not None:
            self.mesh_coll.set_alpha(self.alpha)
        self._draw_robot()
        self.fig.canvas.draw_idle()

    def _write_status(self):
        ax = self.axt
        ax.cla()
        ax.axis("off")
        w = self.w
        n_ana = int(regular_ik_count(w[0], w[2]))
        n_lim = sum(s["in_limits"] for s in self.sols)
        head = (f"{len(self.sols)} IKs from the solver (analytic regular count {n_ana}) · "
                f"{n_lim} within joint limits · wrist center ({w[0]:.3f}, 0, {w[2]:.3f}) m · "
                f"tool0 target ({self.T_target[0, 3]:.3f}, 0, {self.T_target[2, 3]:.3f}) m")
        if self.sols:
            s = self.sols[self.sel]
            sg = lambda v: "+" if v > 0 else ("−" if v < 0 else "0")
            head += (f"\nIK {self.sel + 1}: S {sg(s['S'])}, E {sg(s['E'])}, σ {sg(s['sig'])}, "
                     f"W {sg(s['W'])} · det J = {s['det']:.3e} · σmin(J) = {s['smin']:.4f} · "
                     f"cond(J) = {s['cond']:.1f} · FK error {max(x['ep'] for x in self.sols):.1e} m, "
                     f"{max(x['eR'] for x in self.sols):.1e} rad (all IKs, own PoE FK)")
        if self.error:
            head += f"\nsolver error: {self.error}"
        ax.text(0.0, 1.02, head, fontsize=10, va="top", ha="left", transform=ax.transAxes)

        lines = [f"{'IK':>3} {'q1':>8} {'q2':>8} {'q3':>8} {'q4':>8} {'q5':>8} {'q6':>8}  "
                 f"{'branch':<12}{'S E σ W':<9}{'det J':>11} {'σmin':>8} {'cond':>8}  limits"]
        for i, s in enumerate(self.sols):
            sg = lambda v: "+" if v > 0 else ("-" if v < 0 else "0")
            br = "A q4∈{0,π}" if s["family"] == "A" else "B q5∈{0,π}"
            mark = "▶" if i == self.sel else " "
            lines.append(f"{mark}{i + 1:>2} " + " ".join(f"{v:8.4f}" for v in s["q"]) +
                         f"  {br:<12}{sg(s['S'])} {sg(s['E'])} {sg(s['sig'])} {sg(s['W'])}  "
                         f"{s['det']:11.3e} {s['smin']:8.4f} {s['cond']:8.1f}  "
                         f"{'ok' if s['in_limits'] else 'VIOLATED'}")
        ax.text(0.0, 0.76 if self.sols else 0.8, "\n".join(lines), family="monospace",
                fontsize=8.6, va="top", ha="left", transform=ax.transAxes)
        ax.text(1.0, 0.76,
                "angles in rad · mathematical branches before joint-limit filtering\n\n"
                "markers:  ●  W+ (sin q5 > 0)    ■  W−    ◆  branch B\n"
                "filled = within joint limits · hollow = violates limits\n\n"
                "S = sign G (O5 in front of / behind the base axis)\n"
                "E = sign F (elbow) · σ = side of the a5 offset\n"
                "B: forearm vertical, q4 free. These IKs sit on the wrist\n"
                "lines but are regular: axes 4 and 6 are parallel and\n"
                "offset by a5 = 0.080 m, so they do not coincide.",
                fontsize=8.6, va="top", ha="right", transform=ax.transAxes, color="0.25")

    # --- events --------------------------------------------------------------
    def on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return                                   # zoom / pan active
        if ev.inaxes is self.axw and ev.button == 1 and ev.xdata is not None:
            self.set_target([max(ev.xdata, 1e-4), 0.0, ev.ydata])
        elif ev.inaxes is self.axq and self.sols and ev.xdata is not None:
            d = [np.hypot(wrap(s["q"][1]) - ev.xdata, wrap(s["q"][2]) - ev.ydata) for s in self.sols]
            if min(d) < 0.25:
                self.select(int(np.argmin(d)))

    def on_key(self, ev):
        if ev.key in ("right", "n"):
            self.select(self.sel + 1)
        elif ev.key in ("left", "p"):
            self.select(self.sel - 1)
        elif ev.key == "r":
            self.set_target(random_wrist_center(self.rng))


# ============================================================================
def verify_counts(n=300, seed=0):
    """Compare the analytic regular-IK count with the solver on random wrist centers."""
    rng = np.random.default_rng(seed)
    bad = 0
    for _ in range(n):
        w = np.array([rng.uniform(0.01, RHO_MAX), 0.0, rng.uniform(Z_MIN, Z_MAX)])
        raw, T = solve_ik(w)
        sols = [analyse_solution(s, T) for s in raw]
        n_reg = sum(abs(s["det"]) > 1e-9 for s in sols)
        n_ana = int(regular_ik_count(w[0], w[2]))
        ep = max([s["ep"] for s in sols], default=0.0)
        eR = max([s["eR"] for s in sols], default=0.0)
        # same tolerances as forward_kinematics.main (arccos amplifies round-off in e_R)
        if n_reg != n_ana or n_reg != len(sols) or ep > 1e-9 or eR > 1e-6:
            bad += 1
            print(f"mismatch at {w.round(4)}: solver {len(sols)}, regular {n_reg}, analytic {n_ana}, "
                  f"FK error {ep:.1e} m, {eR:.1e} rad")
    print(f"{n - bad}/{n} wrist centers: solver count == analytic count, all FK-verified")
    return bad == 0


if __name__ == "__main__":
    if "--verify" in sys.argv:
        sys.exit(0 if verify_counts() else 1)
    explorer = SliceExplorer()
    plt.show()