"""
Path-placement study of the ABB GoFa (expected outputs 6-8 of Project 1A):
  * one closed full-pose Cartesian contour and its family of placements on an
    (x, y) grid,
  * for every placement and every global tool orientation, all IK candidates of
    all poses, and for every initial branch whether it completes the contour
    continuously, or why it fails,
  * branch-resolved feasibility maps and failure maps,
  * an interactive explorer (like Lecture 06, "a closed task-space loop can
    return to the same joint configuration", but on the plane z = z0).

Task
----
One contour: an xy rectangle (as in Lecture 07), width x height = --size, in a local
frame {C} (centre at the origin, plane z = 0), N poses (N/4 per edge, corners included):
        C_T_k = [ R_0  p_k ; 0  1 ],   k = 0..N-1,  pose N = pose 0,  pose 0 = corner (-w/2, -h/2).
R_0 = fixed global tool orientation (--tool index into ORIENTATIONS, or --tool-euler).
At the default centre the inner corner lies in the 8-IK region and the outer edges in
the 4-IK region, so some branches lose their IK along the path.
Placement family: 0_T_k(x, y) = Trans(x, y, z0) . C_T_k, (x, y) on a regular grid.

Initial branches and their names
--------------------------------
Every valid IK candidate of pose 0 (our FK check, joint limits, every in-limit
2*pi representative, see ik_verification.py) is an initial branch. It is named
    S E W   shoulder: wrist centre O5 in front of (+) / behind (-) the base axis
            elbow   : sign of F = a3 sin q3 + d4 cos q3
            wrist   : sign of sin q5
  + "det+/det-" only if two solutions of the same pose share S E W,
  + "q6-/q6+"   for the two in-limit representatives q6 and q6 + 2*pi.
S E W are names, not invariants (because of a5); the sign of det J is the
invariant inside a branch. Maps are drawn per S E W name: a name is feasible at
a placement if any of its solutions completes the contour.

Outcome of a branch followed from pose 0 to pose N (first event along the path)
-------------------------------------------------------------------------------
    completes            reaches pose N (closed joint loop if it returns to its start)
    singularity margin   |det 0J_t| < --det-min at some pose
    loss of IK           no valid IK at the next pose, or fewer solutions there
                         (two solutions merge at a singularity and vanish)
    joint limit          the same solution exists at the next pose (mod 2*pi) but
                         only outside the joint limits
    discontinuity        every continuation needs a joint jump > MAX_STEP

Run from the repository root:
One tool per run (0 = tilted_down, 1 = tilted_side, 2 = down), one process, run them one after the other:
    python path_placement.py --tool 0 --no-figures      study of tool 0 only, saved in results/
    python path_placement.py --tool 1 --no-figures      then tool 1, ...
    python path_placement.py --load --tool 0 1          comparison figures from the saved results
    python path_placement.py --gui --tool 1             interactive explorer (gofa_tools/placement_explorer.py)
    python path_placement.py --tool-euler 2.6 0.3 0.8   only a custom tool
    options: --size W H --n --z
             --xlim --ylim --step --det-min --center X Y
"""
import argparse
import hashlib
import itertools
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from gofa_tools.ik_solver import solve_batch
from ik_verification import (MAX_STEP, TOL_P, TOL_R, TOL_SAME, fk_errors, in_limit_equivalents,
                             track_branches, verify_pose, wrapped_dist)

# ============================================================================
# Colours (text in ink, colour only on marks)
# ============================================================================
INK, MUTED, LIGHT = "#1f1f1e", "#6b6b68", "#e4e3df"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEW = ["".join(s) for s in itertools.product("+-", repeat=3)]       # +++, ++-, ..., ---
SEW_COLOR = dict(zip(SEW, ["#2a78d6", "#e8743c", "#1baf7a", "#d64550",
                           "#8b5fd1", "#b8860b", "#e377c2", "#17becf"]))
C_OK, C_FAIL = "#2a78d6", "#f3b1af"

# ============================================================================
# Contour: xy rectangle (local frame {C}, centre at the origin, plane z = 0)
# ============================================================================
def rectangle_points(size, n):
    """n points (n, 3) of a w x h rectangle, n/4 per edge, starting at corner (-w/2, -h/2)."""
    w, h = size[0] / 2, size[1] / 2
    V = np.array([[-w, -h], [w, -h], [w, h], [-w, h], [-w, -h]])
    s = 4 * np.arange(n) / n                       # progress, one unit per edge (as in Lecture 07)
    e, f = np.floor(s).astype(int), (s % 1)[:, None]
    return np.c_[V[e] + f * (V[e + 1] - V[e]), np.zeros(n)]


# ============================================================================
# Global tool orientations R_0
# ============================================================================
R_DOWN = np.diag([1.0, -1.0, -1.0])           # tool z-axis vertical, pointing down
ORIENTATIONS = {
    # Generic orientations (no symmetry with the arm plane). The supplied solver is
    # written for generic poses: an exactly vertical tool, or a tilt inside a
    # vertical plane through the base axis, gives poses with q1, q4 exactly at 0/pi
    # where it is much slower, can return "roots unresolved" or miss solutions.
    "tilted_down": Rot.from_euler("xyz", [np.pi - 0.3, 0.2, 0.4]).as_matrix(),      # 21 deg from vertical
    "tilted_side": Rot.from_euler("xyz", [np.pi / 2 + 0.3, 0.2, 0.4]).as_matrix(),  # 73 deg from vertical
    "down": R_DOWN,                                                                    # comparison only
}


def tilt_deg(R):
    """Angle between the tool z-axis and the downward vertical [deg]."""
    return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))


def contour_task(size, n, R0):
    """Local full poses C_T_k (n, 4, 4) of the rectangle, fixed tool orientation R0."""
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = R0
    T[:, :3, 3] = rectangle_points(size, n)
    return T


def place(T_task, x, y, z0):
    """Targets 0_T_k = Trans(x, y, z0) . C_T_k (pure translation)."""
    T = T_task.copy()
    T[:, :3, 3] += np.array([x, y, z0])
    return T


def closed_targets(T_task, x, y, z0):
    """Poses 0..N of one placement (pose N = pose 0)."""
    T = place(T_task, x, y, z0)
    return np.concatenate([T, T[:1]])


def grid(lim, step):
    return np.round(np.arange(lim[0], lim[1] + 1e-9, step), 6)


# ============================================================================
# Valid IK solutions and IK-count map of the plane z = z0
# ============================================================================
def valid_configurations(raw, T):
    """FK-checked, in-limit configurations, one per configuration modulo 2*pi."""
    ok = []
    for q in raw:
        ep, eR = fk_errors(q, T)
        if ep > TOL_P or eR > TOL_R or any(wrapped_dist(q, c) < TOL_SAME for c in ok):
            continue
        if in_limit_equivalents(q):
            ok.append(q)
    return ok


def ik_count_map(R0, z0, xs, ys):
    """Number of valid IKs of the pose (R0, (x, y, z0)) on a grid. Cached in results/."""
    h = hashlib.sha1(np.r_[R0.ravel().round(9), z0, xs, ys].tobytes()).hexdigest()[:12]
    path = f"results/ikcount_{h}.npz"
    if os.path.exists(path):
        return np.load(path)["count"]
    Ts = []
    for x in xs:
        for y in ys:
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = R0, (x, y, z0)
            Ts.append(T)
    print(f"IK count map: {len(Ts)} poses ...", flush=True)
    raw = solve_batch(Ts)
    count = np.array([len(valid_configurations(r, T)) for r, T in zip(raw, Ts)]).reshape(len(xs), len(ys))
    os.makedirs("results", exist_ok=True)
    np.savez_compressed(path, count=count)
    return count


# ============================================================================
# Initial branches: names, tracking and outcome
# ============================================================================
COMPLETE, LOSS, LIMIT, JUMP, SING, ABSENT = 0, 1, 2, 3, 4, -1
REASONS = {LOSS: "loss of IK", LIMIT: "joint limit", JUMP: "discontinuity", SING: "singularity margin"}


def branch_names(cands):
    """Unique names of the candidates of one pose (see module docstring)."""
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault(c["label"], []).append(i)
    names = [None] * len(cands)
    for lab, idx in groups.items():
        sols = {}                                                # same solution modulo 2*pi
        for i in idx:
            key = next((k for k in sols if wrapped_dist(cands[k]["q"], cands[i]["q"]) < TOL_SAME), i)
            sols.setdefault(key, []).append(i)
        for n_sol, members in enumerate(sols.values()):
            base = lab
            if len(sols) > 1:
                base += " det" + ("+" if cands[members[0]]["det"] > 0 else "-")
                if sum(1 for m in sols.values() if (cands[m[0]]["det"] > 0) == (cands[members[0]]["det"] > 0)) > 1:
                    base += f" #{n_sol + 1}"
            members.sort(key=lambda i: cands[i]["q"][5])
            for j, i in enumerate(members):
                names[i] = base + ((" q6-", " q6+")[j] if len(members) == 2 else "")
    return names


def classify(branch, fk_ok, n, det_min):
    """(outcome, pose index of the event) of one branch that starts at pose 0."""
    pts = branch["pts"]
    for k, _, _, det in pts:
        if abs(det) < det_min:
            return SING, k
    k_last = pts[-1][0]
    if k_last == n:
        return COMPLETE, n
    k_next, q_last = k_last + 1, pts[-1][1]
    if not fk_ok[k_next]:
        return LOSS, k_last
    if min(wrapped_dist(q, q_last) for q in fk_ok[k_next]) <= MAX_STEP:
        return LIMIT, k_last
    if len(fk_ok[k_next]) < len(fk_ok[k_last]):
        return LOSS, k_last
    return JUMP, k_last


def follow_branches(raw, Ts, det_min):
    """Every initial branch of one placement, followed over poses 0..N."""
    n = len(Ts) - 1
    per_pose, fk_ok = [], []
    for r, T in zip(raw, Ts):
        c, f, _ = verify_pose(r, T)
        per_pose.append(c)
        fk_ok.append(f)
    names = branch_names(per_pose[0])
    lifts = []
    for b in track_branches(per_pose, fk_ok):
        if b["pts"][0][0] != 0:
            continue
        i0 = int(np.argmin([np.abs(b["pts"][0][1] - c["q"]).max() for c in per_pose[0]]))
        code, k = classify(b, fk_ok, n, det_min)
        q = np.array([x[1] for x in b["pts"]])
        lifts.append(dict(name=names[i0], sew=per_pose[0][i0]["label"], code=code, k=k,
                          ks=np.array([x[0] for x in b["pts"]]), q=q,
                          det=np.array([x[3] for x in b["pts"]]),
                          closed=code == COMPLETE and np.abs(q[0] - q[-1]).max() < 1e-6))
    lifts.sort(key=lambda L: (SEW.index(L["sew"]), L["name"]))
    return lifts


def outcome_text(L):
    if L["code"] == COMPLETE:
        return "completes, closed joint loop" if L["closed"] else "completes, ends in another IK"
    return f"fails at pose {L['k']}: {REASONS[L['code']]}"


def summarise_placement(lifts):
    """Outcome per S E W name: completes if any of its solutions completes, otherwise
    the failure of the solution that went furthest."""
    out = np.full(len(SEW), ABSENT)
    kf = np.full(len(SEW), -1)
    closed = np.zeros(len(SEW), bool)
    for L in lifts:
        i = SEW.index(L["sew"])
        better = (out[i] == ABSENT or (L["code"] == COMPLETE and out[i] != COMPLETE)
                  or (out[i] != COMPLETE and L["code"] != COMPLETE and L["k"] > kf[i]))
        if better:
            out[i], kf[i] = L["code"], L["k"]
        closed[i] |= L["closed"]
    return out, kf, closed


# ============================================================================
# Whole family (serial, one grid row per solver call)
# ============================================================================
def _evaluate_row(T_task, x, ys, z0, det_min):
    n1 = len(T_task) + 1
    Ts = np.concatenate([closed_targets(T_task, x, y, z0) for y in ys])
    raw = solve_batch(Ts)
    return [summarise_placement(follow_branches(raw[i * n1:(i + 1) * n1], Ts[i * n1:(i + 1) * n1], det_min))
            for i in range(len(ys))]


def evaluate_family(T_task, xs, ys, z0, det_min):
    out = np.full((len(xs), len(ys), len(SEW)), ABSENT)
    kf = np.full_like(out, -1)
    closed = np.zeros(out.shape, bool)
    t0 = time.time()
    for ix, x in enumerate(xs):
        for iy, (o, k, c) in enumerate(_evaluate_row(T_task, x, ys, z0, det_min)):
            out[ix, iy], kf[ix, iy], closed[ix, iy] = o, k, c
        if (ix + 1) % 5 == 0 or ix + 1 == len(xs):
            print(f"  row {ix + 1:3d}/{len(xs)}  ({time.time() - t0:4.0f} s)", flush=True)
    return out, kf, closed


def print_summary(name, r):
    o = r["out"]
    present = o != ABSENT
    print(f"\n{name}: {o.shape[0] * o.shape[1]} placements")
    print(f"  no valid IK at pose 0           : {np.sum(~present.any(-1))}")
    print(f"  >= 1 initial branch completes   : {np.sum((o == COMPLETE).any(-1))}")
    print(f"  initial branches (S E W names)  : {present.sum()}, complete {np.sum(o == COMPLETE)} "
          f"(closed joint loop {np.sum(r['closed'] & (o == COMPLETE))})")
    for code, txt in REASONS.items():
        print(f"  fail, {txt:<19}: {np.sum(o == code)}")
    for i, lab in enumerate(SEW):
        print(f"  {lab}: present {np.sum(present[..., i]):4d}, completes {np.sum(o[..., i] == COMPLETE):4d}")


# ============================================================================
# Drawing helpers
# ============================================================================
def _seq(n):
    from matplotlib.colors import LinearSegmentedColormap
    ramp = LinearSegmentedColormap.from_list("seq", SEQ)
    return [ramp(t) for t in np.linspace(0.08, 1, n)] if n > 1 else [SEQ[3]]


def _mesh(ax, xs, ys, Z, cmap, norm):
    h = xs[1] - xs[0]
    return ax.pcolormesh(np.r_[xs - h / 2, xs[-1] + h / 2], np.r_[ys - h / 2, ys[-1] + h / 2],
                         Z.T, cmap=cmap, norm=norm, rasterized=True)


def _frame(ax, xs, ys):
    from matplotlib.patches import Circle
    ax.add_patch(Circle((0, 0), 0.05, color=INK, zorder=3))
    ax.set_aspect("equal")
    h = (xs[1] - xs[0]) / 2
    ax.set_xlim(xs[0] - h, xs[-1] + h)
    ax.set_ylim(ys[0] - h, ys[-1] + h)


def draw_plane(ax, count, xs_map, ys_map, T_task, center, xs_grid=None, ys_grid=None):
    """IK-count shading of the plane, its boundaries, base and the placed contour."""
    from matplotlib.colors import BoundaryNorm, ListedColormap
    top = max(int(count.max()), 1)
    cmap = ListedColormap([LIGHT] + _seq(top))
    norm = BoundaryNorm(np.arange(-0.5, top + 1.5), cmap.N)
    im = _mesh(ax, xs_map, ys_map, count, cmap, norm)
    ax.contour(xs_map, ys_map, count.T, levels=np.arange(0.5, count.max(), 1), colors="#d64550",
               linewidths=0.8)
    if xs_grid is not None:
        GX, GY = np.meshgrid(xs_grid, ys_grid)
        ax.plot(GX.ravel(), GY.ravel(), ".", color=INK, ms=1.2, alpha=0.35)
    _frame(ax, xs_map, ys_map)
    P = T_task[:, :3, 3] + np.array([center[0], center[1], 0])
    ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], "--", color=INK, lw=1.4)
    ax.plot(*P[0, :2], "o", color=INK, ms=5)
    ax.annotate("k = 0", P[0, :2], xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return im


def draw_lifts(ax, lifts, sel=None, full_range=True):
    """(q2, q3) projection of every lift, coloured by its S E W name."""
    for i, L in enumerate(lifts):
        bold = sel is None or i == sel
        col = SEW_COLOR[L["sew"]]
        ls = "--" if L["name"].endswith("q6+") else "-"
        ax.plot(L["q"][:, 1], L["q"][:, 2], ls, color=col, lw=2.2 if bold else 0.9, alpha=1 if bold else 0.35)
        ax.plot(L["q"][0, 1], L["q"][0, 2], "o", color=col, ms=6 if bold else 4, mec="white", mew=0.8)
        if L["code"] != COMPLETE:
            ax.plot(L["q"][-1, 1], L["q"][-1, 2], "x", color=col, ms=7, mew=2)
    if full_range:
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-3.93, 1.49)
    else:
        ax.margins(0.12)
    ax.set_xlabel("q2 [rad]")
    ax.set_ylabel("q3 [rad]")
    ax.grid(alpha=0.3)


# ============================================================================
# Report figures
# ============================================================================
def figure_definition(args, oris, out):
    """The full-pose rectangle in {C} for every orientation."""
    import matplotlib.pyplot as plt
    from gofa_tools.robot_view import draw_frame
    fig = plt.figure(figsize=(2.6 * len(oris), 2.9), constrained_layout=True)
    for j, (name, R0) in enumerate(oris.items()):
        T_task = contour_task(args.size, args.n, R0)
        ax = fig.add_subplot(1, len(oris), 1 + j, projection="3d")
        P = T_task[:, :3, 3]
        ax.plot(*np.r_[P, P[:1]].T, color=SEQ[4], lw=2)
        for k in range(0, len(T_task), max(1, len(T_task) // 8)):
            draw_frame(ax, T_task[k], length=0.2 * max(args.size), lw=1.3)
        ax.scatter(*P[0], color=INK, s=14)
        lim = 0.75 * max(args.size)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=25, azim=-60)
        ax.tick_params(labelsize=6)
        ax.set_title(f"({'abc'[j]}) {name.replace('_', ' ')}, {tilt_deg(R0):.0f}° tilt", fontsize=9, loc="left")
    fig.savefig(os.path.join(out, "contour_rectangle.pdf"))
    plt.close(fig)


def figure_example(args, name, R0, count, xs_map, ys_map, out):
    """One placement: contour in the IK-count map, lifts, robot."""
    import matplotlib.pyplot as plt
    from gofa_tools.robot_view import draw_robot, load_mesh_model, setup_axes
    T_task = contour_task(args.size, args.n, R0)
    Ts = closed_targets(T_task, *args.center, args.z)
    lifts = follow_branches(solve_batch(Ts), Ts, args.det_min)
    fig = plt.figure(figsize=(7.2, 3.0), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.15, 1])
    ax = fig.add_subplot(gs[0, 0])
    draw_plane(ax, count, xs_map, ys_map, T_task, args.center)
    span = 1.2 * max(args.size)
    ax.set_xlim(args.center[0] - span, args.center[0] + span)
    ax.set_ylim(args.center[1] - span, args.center[1] + span)
    ax.set_title("(a) contour in the plane", fontsize=9, loc="left")
    ax = fig.add_subplot(gs[0, 1])
    draw_lifts(ax, lifts, full_range=False)
    ax.set_title("(b) initial branches (q2, q3)", fontsize=9, loc="left")
    from matplotlib.lines import Line2D
    seen = list(dict.fromkeys(L["sew"] for L in lifts))
    ax.legend(handles=[Line2D([], [], color=SEW_COLOR[s], lw=2, label=s) for s in seen],
              fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False,
              title="S E W", title_fontsize=6.5)
    ax = fig.add_subplot(gs[0, 2], projection="3d")
    ax.plot(*Ts[:, :3, 3].T, "--", color="0.5", lw=1)
    done = [L for L in lifts if L["code"] == COMPLETE] or lifts
    if done:
        draw_robot(ax, done[0]["q"][0], load_mesh_model() if args.meshes else None, SEW_COLOR[done[0]["sew"]])
        ax.set_title(f"(c) branch {done[0]['name']} at k = 0", fontsize=9, loc="left")
    setup_axes(ax)
    ax.tick_params(labelsize=6)
    fig.savefig(os.path.join(out, f"example_rectangle_{name}.pdf"))
    plt.close(fig)
    print(f"\nexample placement {tuple(args.center)} m, {name}:")
    for L in lifts:
        print(f"  {L['name']:<16} {outcome_text(L)}")


def figure_family(args, oris, counts, xs_map, ys_map, out):
    """IK-count map of each orientation with a few example placements."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(oris), figsize=(7.2, 3.5), constrained_layout=True, squeeze=False)
    for ax, (name, R0) in zip(axs[0], oris.items()):
        T_task = contour_task(args.size, args.n, R0)
        im = draw_plane(ax, counts[name], xs_map, ys_map, T_task, args.center)
        P = T_task[:, :3, 3]
        for (x, y) in [(-0.3, 0.5), (0.2, -0.75), (-0.75, -0.3)]:
            ax.plot(np.r_[P[:, 0], P[0, 0]] + x, np.r_[P[:, 1], P[0, 1]] + y, "--", color=INK, lw=1)
        ax.set_title(f"{name.replace('_', ' ')}", fontsize=9)
    top = max(int(c.max()) for c in counts.values())
    cb = fig.colorbar(im, ax=axs, fraction=0.035, pad=0.02, ticks=range(0, top + 1, 2))
    cb.set_label(f"valid IK solutions at z = {args.z:.2f} m")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, "family_rectangle.pdf"))
    plt.close(fig)


def figure_feasibility(args, results, out):
    """Number of initial branches that complete the contour, per orientation."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    top = max(int((r["out"] == COMPLETE).sum(-1).max()) for r in results.values())
    top = max(top, 1)
    cmap = ListedColormap([LIGHT] + _seq(top))
    norm = BoundaryNorm(np.arange(-0.5, top + 1.5), cmap.N)
    fig, axs = plt.subplots(1, len(results), figsize=(7.2, 3.4), constrained_layout=True, squeeze=False)
    for ax, (name, r) in zip(axs[0], results.items()):
        im = _mesh(ax, r["xs"], r["ys"], (r["out"] == COMPLETE).sum(-1), cmap, norm)
        _frame(ax, r["xs"], r["ys"])
        ax.set_title(name.replace("_", " "), fontsize=9)
        ax.set_xlabel("contour centre x [m]")
    axs[0, 0].set_ylabel("contour centre y [m]")
    cb = fig.colorbar(im, ax=axs, ticks=range(top + 1), fraction=0.035, pad=0.02)
    cb.set_label("initial branches (S E W) that complete the contour")
    cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, "feasible_count_rectangle.pdf"))
    plt.close(fig)


def figure_branches(args, name, r, out):
    """Branch-resolved feasibility map: one panel per S E W name."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(["white", C_OK, C_FAIL])
    norm = BoundaryNorm([-1.5, -0.5, 0.5, 4.5], cmap.N)
    fig, axs = plt.subplots(2, 4, figsize=(7.2, 4.1), sharex=True, sharey=True, constrained_layout=True)
    for i, ax in enumerate(axs.flat):
        _mesh(ax, r["xs"], r["ys"], r["out"][..., i], cmap, norm)
        _frame(ax, r["xs"], r["ys"])
        ax.tick_params(labelsize=7)
        ax.set_title(f"S E W = {SEW[i]}  ({int((r['out'][..., i] == COMPLETE).sum())})", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("x [m]")
    for ax in axs[:, 0]:
        ax.set_ylabel("y [m]")
    fig.legend(handles=[Patch(color=C_OK, label="completes the contour"), Patch(color=C_FAIL, label="fails"),
                        Patch(facecolor="white", edgecolor=MUTED, label="not an IK of pose 0")],
               loc="outside lower center", ncol=3, frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, f"branches_rectangle_{name}.pdf"))
    plt.close(fig)


def figure_failures(args, results, out):
    """Failure maps: rows = orientations, columns = failure reasons."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch
    top = max(max(int((r["out"] == c).sum(-1).max()) for c in REASONS) for r in results.values())
    top = max(top, 1)
    cmap = ListedColormap(["white"] + _seq(top))
    norm = BoundaryNorm(np.arange(-0.5, top + 1.5), cmap.N)
    fig, axs = plt.subplots(len(results), 4, figsize=(7.2, 1.95 * len(results) + 0.7), sharex=True,
                            sharey=True, constrained_layout=True, squeeze=False)
    for row, (name, r) in zip(axs, results.items()):
        no_start = (r["out"] == ABSENT).all(-1)
        for ax, code in zip(row, REASONS):
            Z = (r["out"] == code).sum(-1)
            im = _mesh(ax, r["xs"], r["ys"], Z, cmap, norm)
            ax.contourf(r["xs"], r["ys"], no_start.T.astype(float), levels=[0.5, 1.5], colors=[LIGHT], zorder=2)
            _frame(ax, r["xs"], r["ys"])
            ax.tick_params(labelsize=7)
            ax.set_title(f"{REASONS[code]} ({int(Z.sum())})", fontsize=8)
        row[0].set_ylabel(f"{name.replace('_', ' ')}\ny [m]", fontsize=8)
    for ax in axs[-1]:
        ax.set_xlabel("x [m]")
    cb = fig.colorbar(im, ax=axs, ticks=range(top + 1), fraction=0.03, pad=0.02)
    cb.set_label("initial branches failing for this reason", fontsize=8)
    cb.outline.set_visible(False)
    fig.legend(handles=[Patch(color=LIGHT, label="no valid IK at pose 0")], loc="outside lower center",
               frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "failures_rectangle.pdf"))
    plt.close(fig)


# ============================================================================
def main():
    p = argparse.ArgumentParser(description="Path-placement study: contour, family, feasibility and failure maps.")
    p.add_argument("--tool", type=int, nargs="*", choices=range(len(ORIENTATIONS)),
                   help=f"preset tool orientations R0 by index {dict(enumerate(ORIENTATIONS))} "
                        "(default: 0, or none if --tool-euler is given)")
    p.add_argument("--tool-euler", type=float, nargs=3, metavar=("RX", "RY", "RZ"),
                   help="add a custom tool orientation (XYZ Euler angles, rad)")
    p.add_argument("--size", type=float, nargs=2, default=(0.40, 0.30), metavar=("W", "H"),
                   help="rectangle width (x) and height (y) [m]")
    p.add_argument("--n", type=int, default=128, help="poses on the rectangle (multiple of 4)")
    p.add_argument("--z", type=float, default=0.30, help="height z0 of the contour plane [m]")
    p.add_argument("--xlim", type=float, nargs=2, default=(-1.1, 1.1))
    p.add_argument("--ylim", type=float, nargs=2, default=(-1.1, 1.1))
    p.add_argument("--step", type=float, default=0.05, help="placement grid step [m]")
    p.add_argument("--map-step", type=float, default=0.02, help="resolution of the IK-count map [m]")
    p.add_argument("--det-min", type=float, default=0.01, help="singularity margin on |det 0J_t| [m^3]")
    p.add_argument("--center", type=float, nargs=2, default=(0.70, 0.20), help="example / initial centre [m]")
    p.add_argument("--no-meshes", dest="meshes", action="store_false", help="stick robot instead of STLs")
    p.add_argument("--load", action="store_true", help="no study: figures from the saved results")
    p.add_argument("--no-figures", dest="figures", action="store_false",
                   help="run the study and save the results only")
    p.add_argument("--gui", action="store_true", help="open the explorer for the first orientation")
    args = p.parse_args()

    if args.n % 4:
        p.error("--n must be a multiple of 4 (same number of poses per edge, corners included)")
    if args.tool is None:
        args.tool = [] if args.tool_euler else [0]
    names = list(ORIENTATIONS)
    oris = {names[i]: ORIENTATIONS[names[i]] for i in args.tool}
    if args.tool_euler:
        oris["custom"] = Rot.from_euler("xyz", args.tool_euler).as_matrix()
    if not oris:
        p.error("choose at least one orientation (--tool or --tool-euler)")
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    xs, ys = grid(args.xlim, args.step), grid(args.ylim, args.step)
    xs_map, ys_map = grid(args.xlim, args.map_step), grid(args.ylim, args.map_step)
    counts = {k: ik_count_map(R0, args.z, xs_map, ys_map) for k, R0 in oris.items()}

    results = {}
    for name, R0 in oris.items():
        path = f"results/feasibility_rectangle_{name}.npz"
        if args.load or args.gui:
            if os.path.exists(path):
                d = np.load(path)
                results[name] = {k: d[k] for k in d.files}
            else:
                print(f"[warn] no saved study for {name} ({path}); run it first without --load")
            continue
        T_task = contour_task(args.size, args.n, R0)
        print(f"{name} ({tilt_deg(R0):.0f}° tilt): {len(xs)} x {len(ys)} placements x {args.n} poses, "
              f"|det J| >= {args.det_min} m^3")
        o, k, c = evaluate_family(T_task, xs, ys, args.z, args.det_min)
        results[name] = dict(out=o, kfail=k, closed=c, xs=xs, ys=ys)
        np.savez_compressed(path, out=o, kfail=k, closed=c, xs=xs, ys=ys, R0=R0, z0=args.z,
                            det_min=args.det_min, size=args.size, n=args.n)

    if args.gui:
        import matplotlib.pyplot as plt
        from gofa_tools.placement_explorer import PlacementExplorer
        name = next(iter(oris))
        explorer = PlacementExplorer(args, name, oris[name], counts[name], xs_map, ys_map, xs, ys,  # noqa: F841
                                     results.get(name))
        plt.show()
        return
    if not args.figures:
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8})
    for name, r in results.items():
        print_summary(name, r)
    figure_definition(args, oris, "figures")
    figure_family(args, oris, counts, xs_map, ys_map, "figures")
    for name, R0 in oris.items():
        figure_example(args, name, R0, counts[name], xs_map, ys_map, "figures")
    if results:
        figure_feasibility(args, results, "figures")
        for name, r in results.items():
            figure_branches(args, name, r, "figures")
        figure_failures(args, results, "figures")
    print("\nfigures saved in figures/")


if __name__ == "__main__":
    main()