"""
Verification and branch organisation of the supplied CRB 15000 IK solver.

For every pose of a short Cartesian path this script
  1. calls the supplied solver (gofa_tools/ik_solver.py, batch mode, 4x4 tool0 targets: no quaternion convention to get wrong),
  2. re-checks every returned configuration with our own PoE forward kinematics,
  3. resolves the periodic representation: lists every 2*pi-equivalent of each joint
     that fits the physical joint limits (none -> the configuration is rejected).
     Each equivalent is a different physical joint position, so each one becomes
     its own joint-space candidate (only q6, range 3*pi, can have two),
  4. labels each candidate with (S, E, W) and the sign of det J,
  5. links the candidates of consecutive poses into persistent branches
     (minimum-cost matching of joint distance, Hungarian algorithm) and records
     why every branch ends: end of path, no IK, joint limit or discontinuity.

The solver is only used as a source of candidate configurations: it never
decides validity, limits, or continuity.

Run from the repository root:
    python ik_verification.py            (default straight-line path, text summary)
    python ik_verification.py --gui      (also open the interactive branch explorer)
"""
import argparse
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation as Rot

from forward_kinematics import fk_poe
from models.dh import DH_TABLE, dh_frames
from models.screw import Q_LIMITS
from screw_jacobian import jacobian_0Jt
from gofa_tools.ik_solver import solve_batch

TOL_P = 1e-9          # [m]   FK position tolerance
TOL_R = 1e-9          # [rad] FK orientation tolerance (rotation-vector norm, no arccos round-off)
TOL_LIM = 1e-12       # [rad] joint-limit tolerance, same as the solver
TOL_SAME = 1e-6       # [rad] two configurations closer than this are duplicates
MAX_STEP = 0.25       # [rad] largest joint jump (inf-norm) still accepted as continuous motion

A3, D4 = DH_TABLE[2, 1], DH_TABLE[3, 2]


# ============================================================================
# 2. Independent FK check
# ============================================================================
def fk_errors(q, T):
    """Position [m] and orientation [rad] error of our PoE FK against the target."""
    Tq = fk_poe(q)
    ep = np.linalg.norm(Tq[:3, 3] - T[:3, 3])
    eR = np.linalg.norm(Rot.from_matrix(T[:3, :3].T @ Tq[:3, :3]).as_rotvec())
    return ep, eR


# ============================================================================
# 3. Periodic representation + joint limits
# ============================================================================
def in_limit_equivalents(q):
    """All q + 2*pi*k (per joint) inside the joint limits. Empty list = limits violated.
    Only q6 ([-3pi/2, 3pi/2], range 3*pi) can have a second representative: the
    range of q3 ([-3.93, 1.48], 5.41 rad) is shorter than 2*pi."""
    per_joint = []
    for qi, (lo, hi) in zip(q, Q_LIMITS):
        k = np.arange(np.ceil((lo - TOL_LIM - qi) / (2 * np.pi)),
                      np.floor((hi + TOL_LIM - qi) / (2 * np.pi)) + 1)
        per_joint.append(qi + 2 * np.pi * k)
    return [np.array(c) for c in itertools.product(*per_joint)]


def wrapped_dist(q1, q2):
    """Joint distance modulo 2*pi (inf-norm), used to spot duplicates."""
    d = (np.asarray(q1) - q2 + np.pi) % (2 * np.pi) - np.pi
    return np.abs(d).max()


# ============================================================================
# 4. Discrete branch label
# ============================================================================
def branch_label(q):
    """(S, E, W) signs, used as branch names:
      S shoulder: wrist center O5 in front of (+) / behind (-) the base axis
      E elbow   : sign F = a3 sin q3 + d4 cos q3   (elbow up / down)
      W wrist   : sign sin q5                      (wrist flip)
    For the GoFa they are NOT invariants: because of the a5 offset, F = 0 and
    sin q5 = 0 are not exactly singular, so E or W can change without crossing a
    singularity. The invariant inside a branch is the sign of det J (see det_sign)."""
    O5 = dh_frames(q)[5][:3, 3]
    fwd = np.array([np.cos(q[0]), np.sin(q[0]), 0.0])
    sg = lambda v: "+" if v > 0 else "-"
    return sg(O5 @ fwd) + sg(A3 * np.sin(q[2]) + D4 * np.cos(q[2])) + sg(np.sin(q[4]))


# ============================================================================
# Per-pose verification
# ============================================================================
def verify_pose(raw, T):
    """Validated joint-space candidates of one pose + rejection counts.
    Every in-limit representative is its own candidate (q6 can give two).
    The FK-valid configurations are also returned, even those outside the
    limits, so that track_branches can tell a joint-limit end from a jump."""
    stats = dict(raw=len(raw), fk_fail=0, duplicate=0, limits=0, valid=0)
    cands, fk_ok = [], []
    for q in raw:
        ep, eR = fk_errors(q, T)
        if ep > TOL_P or eR > TOL_R:
            stats["fk_fail"] += 1
            continue
        if any(wrapped_dist(q, c) < TOL_SAME for c in fk_ok):
            stats["duplicate"] += 1
            continue
        fk_ok.append(q)
        reps = in_limit_equivalents(q)
        if not reps:
            stats["limits"] += 1
            continue
        stats["valid"] += 1
        label = branch_label(q)
        det = float(np.linalg.det(jacobian_0Jt(q)))       # same for every 2*pi representative
        for r in reps:
            cands.append(dict(q=r, label=label, det=det, ep=ep, eR=eR))
    stats["candidates"] = len(cands)
    return cands, fk_ok, stats


def det_sign(det):
    return "+" if det > 0 else "-"


# ============================================================================
# 5. Persistent branches along the path
# ============================================================================
def end_reason(q_last, k_next, per_pose, fk_ok_per_pose):
    """Why a branch that stopped at pose k_next - 1 could not continue."""
    if k_next >= len(per_pose):
        return "end of path"
    if not fk_ok_per_pose[k_next]:
        return "no IK"
    # The same solution exists (modulo 2*pi) close to q_last, but no in-limit
    # representative of it is close: the branch runs into a joint limit.
    if min(wrapped_dist(q, q_last) for q in fk_ok_per_pose[k_next]) <= MAX_STEP:
        return "joint limit"
    return "discontinuity"


def track_branches(per_pose, fk_ok_per_pose):
    """Link candidates of consecutive poses. Each branch is a dict with its points
    (pose index, q, label, det) and the reason it ends.
    Cost(branch, candidate) = joint jump ||q_k - q_{k-1}||_inf. Jumps above MAX_STEP
    are not continuous motion: the branch ends there."""
    branches, active = [], []                      # active = indices of branches alive at k-1
    for k, cands in enumerate(per_pose):
        new_active, used = [], set()
        if active and cands:
            C = np.array([[np.abs(c["q"] - branches[b]["pts"][-1][1]).max() for c in cands]
                          for b in active])
            # Gate first: every jump > MAX_STEP gets the same huge cost, otherwise the
            # optimiser trades a good match away to make an impossible one slightly cheaper.
            rows, cols = linear_sum_assignment(np.where(C <= MAX_STEP, C, 1e9))
            for i, j in zip(rows, cols):
                if C[i, j] <= MAX_STEP:
                    c = cands[j]
                    branches[active[i]]["pts"].append((k, c["q"], c["label"], c["det"]))
                    new_active.append(active[i])
                    used.add(j)
        for b in active:                           # branches that stopped at k-1
            if b not in new_active:
                branches[b]["end"] = end_reason(branches[b]["pts"][-1][1], k, per_pose, fk_ok_per_pose)
        for j, c in enumerate(cands):              # unmatched candidates start new branches
            if j not in used:
                branches.append(dict(pts=[(k, c["q"], c["label"], c["det"])], end=None))
                new_active.append(len(branches) - 1)
        active = new_active
    for b in active:
        branches[b]["end"] = "end of path"
    return branches


# ============================================================================
# Demo path
# ============================================================================
def default_path(n=60):
    """Straight line, fixed tilted tool orientation (a generic pose: the solver's
    formulation is for generic poses, see abbGofa_ik/.../README.md)."""
    R = Rot.from_euler("xyz", [np.pi - 0.3, 0.2, 0.4]).as_matrix()
    p0, p1 = np.array([0.45, -0.25, 0.35]), np.array([0.45, 0.25, 0.55])
    Ts = []
    for s in np.linspace(0, 1, n):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, (1 - s) * p0 + s * p1
        Ts.append(T)
    return Ts


# ============================================================================
# Interactive branch explorer
# ============================================================================
class BranchExplorer:
    """Left: timeline of every branch along the path (click one to follow it).
    Middle: joint angles and |det J| of the selected branch.
    Right: the robot following the selected branch.
    Slider / arrow keys: pose index · Play (space): animate · up/down: change branch."""

    def __init__(self, Ts, branches, mesh="auto"):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        from gofa_tools.robot_view import load_mesh_model
        self.plt = plt
        self.Ts, self.B = Ts, branches
        self.path = np.array([T[:3, 3] for T in Ts])
        self.mesh = load_mesh_model() if mesh == "auto" else mesh
        self.colors = [plt.cm.tab10(i % 10) for i in range(len(branches))]
        self.at = [{k: (q, lab, det) for k, q, lab, det in b["pts"]} for b in branches]
        self.sel, self.k = 0, branches[0]["pts"][0][0]
        self.view, self.alpha, self.playing = (18, -70), 1.0, False

        self.fig = plt.figure(figsize=(17, 9.5))
        self.fig.canvas.manager.set_window_title("ABB GoFa - persistent IK branches")
        gs = self.fig.add_gridspec(2, 3, height_ratios=[1, 0.32], width_ratios=[1, 1.05, 1],
                                   left=0.06, right=0.98, top=0.90, bottom=0.10,
                                   wspace=0.25, hspace=0.30)
        self.axb = self.fig.add_subplot(gs[0, 0])
        self.axq = self.fig.add_subplot(gs[0, 1])
        self.axd = self.axq.twinx()
        self.axr = self.fig.add_subplot(gs[0, 2], projection="3d")
        self.axt = self.fig.add_subplot(gs[1, :])
        self.axt.axis("off")
        self.status = self.axt.text(0.0, 1.0, "", va="top", ha="left", family="monospace",
                                    fontsize=9.5, transform=self.axt.transAxes)
        self.bg = None                                 # blitting background while playing
        self.fig.text(0.06, 0.965, "Persistent IK branches along the path", fontsize=15,
                      fontweight="bold", va="center")
        self.fig.text(0.06, 0.935, "Every branch is one continuous joint-space motion built from the "
                      "solver's per-pose solutions (FK-checked, joint limits, 2π representatives). "
                      "Click a branch to follow it.", fontsize=10.5, color="0.2", va="center")

        self._draw_timeline()
        self.s_pose = Slider(self.fig.add_axes([0.10, 0.035, 0.45, 0.025]), "pose", 0, len(Ts) - 1,
                             valinit=self.k, valstep=1)
        self.s_pose.on_changed(lambda v: self.set_pose(int(v)))
        self.b_play = Button(self.fig.add_axes([0.60, 0.025, 0.07, 0.045]), "Play")
        self.b_prev = Button(self.fig.add_axes([0.69, 0.025, 0.08, 0.045]), "◀ branch")
        self.b_next = Button(self.fig.add_axes([0.78, 0.025, 0.08, 0.045]), "branch ▶")
        self.b_play.on_clicked(lambda _: self.toggle_play())
        self.b_prev.on_clicked(lambda _: self.select(self.sel - 1))
        self.b_next.on_clicked(lambda _: self.select(self.sel + 1))
        if self.mesh is not None:
            self.s_alpha = Slider(self.fig.add_axes([0.90, 0.035, 0.07, 0.025]), "STL", 0.0, 1.0,
                                  valinit=1.0, valstep=0.05)
            self.s_alpha.on_changed(self._set_alpha)
        self.timer = self.fig.canvas.new_timer(interval=120)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self.select(0)

    # --- static timeline -----------------------------------------------------
    def _draw_timeline(self):
        ax = self.axb
        n = len(self.B)
        for i, b in enumerate(self.B):
            k0, k1 = b["pts"][0][0], b["pts"][-1][0]
            ax.plot([k0, k1], [i, i], color=self.colors[i], lw=7, solid_capstyle="butt")
            if b["end"] != "end of path":
                ax.plot(k1, i, "x", color="k", ms=8, mew=2)
                ax.annotate(b["end"], (k1, i), xytext=(5, 4), textcoords="offset points", fontsize=8)
        labels = [f"{i}  {b['pts'][0][2]}  det{det_sign(b['pts'][0][3])}" for i, b in enumerate(self.B)]
        ax.set_yticks(range(n), labels, family="monospace", fontsize=8.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xlim(-0.5, len(self.Ts) - 0.5)
        ax.set_xlabel("pose index along the path")
        ax.set_title("branches · label S E W · sign det J  (× = branch ends)", fontsize=11)
        ax.grid(axis="x", alpha=0.3)
        self.b_band = ax.axhspan(-0.45, 0.45, color="0.88", zorder=0)
        self.b_cursor = ax.axvline(self.k, color="k", lw=1)

    # --- selection -----------------------------------------------------------
    def select(self, i):
        self.sel = i % len(self.B)
        ks = [x[0] for x in self.B[self.sel]["pts"]]
        self.b_band.set_y(self.sel - 0.45)
        self._draw_joints()
        k = int(np.clip(self.k, ks[0], ks[-1]))
        if k != self.k:
            self.s_pose.set_val(k)                     # triggers set_pose
        else:
            self.set_pose(k)

    def _draw_joints(self):
        ax, axd = self.axq, self.axd
        ax.cla()
        axd.cla()
        b = self.B[self.sel]
        ks = np.array([x[0] for x in b["pts"]])
        qs = np.array([x[1] for x in b["pts"]])
        for j in range(6):
            ax.plot(ks, qs[:, j], color=f"C{j}", lw=2, label=f"q{j + 1}")
        axd.semilogy(ks, [abs(x[3]) for x in b["pts"]], "k--", lw=1.2, label="|det J|")
        axd.set_ylabel("|det J|  (dashed, log)")
        ax.set_xlim(-0.5, len(self.Ts) - 0.5)
        ax.set_ylim(-5, 5)
        ax.set_xlabel("pose index along the path")
        ax.set_ylabel("joint angle [rad]")
        ax.set_title(f"branch {self.sel} · joint angles", fontsize=11)
        ax.legend(loc="upper left", ncol=3, fontsize=8)
        ax.grid(alpha=0.3)
        self.q_cursor = ax.axvline(self.k, color="k", lw=1, animated=self.playing)

    # --- pose ----------------------------------------------------------------
    def set_pose(self, k):
        self.k = k
        self.b_cursor.set_xdata([k, k])
        self.q_cursor.set_xdata([k, k])
        self._draw_robot()
        self._write_status()
        if self.playing and self.bg is not None:
            self._blit()
        else:
            self.fig.canvas.draw_idle()

    def _draw_robot(self):
        from gofa_tools.robot_view import draw_frame, draw_robot, setup_axes
        ax = self.axr
        if ax.has_data():                              # keep the user's rotation of the 3D view
            self.view = (ax.elev, ax.azim)
        ax.cla()
        self.mesh_coll = None
        ax.plot(*self.path.T, color="0.5", lw=1.5)
        ax.plot(*self.path[:self.k + 1].T, color=self.colors[self.sel], lw=3)
        hit = self.at[self.sel].get(self.k)
        if hit is not None:
            self.mesh_coll = draw_robot(ax, hit[0], self.mesh, self.colors[self.sel], self.alpha)
            ax.set_title(f"branch {self.sel} at pose {self.k}", fontsize=11)
        else:
            draw_frame(ax, self.Ts[self.k])
            ax.set_title(f"branch {self.sel} does not exist at pose {self.k}", fontsize=11, color="C3")
        setup_axes(ax, self.view)

    def _write_status(self):
        b = self.B[self.sel]
        k0, k1 = b["pts"][0][0], b["pts"][-1][0]
        names = "→".join(dict.fromkeys(x[2] for x in b["pts"]))
        txt = (f"branch {self.sel}: poses {k0}–{k1} ({k1 - k0 + 1} of {len(self.Ts)}) · label {names} · "
               f"ends: {b['end']}")
        hit = self.at[self.sel].get(self.k)
        if hit is not None:
            q, lab, det = hit
            ep, eR = fk_errors(q, self.Ts[self.k])
            margin = np.minimum(q - Q_LIMITS[:, 0], Q_LIMITS[:, 1] - q)
            prev = self.at[self.sel].get(self.k - 1)
            step = np.abs(q - prev[0]).max() if prev is not None else 0.0
            txt += (f"\npose {self.k}: q = [" + ", ".join(f"{v:7.3f}" for v in q) + "] rad · label " + lab +
                    f"\ndet J = {det:.3e} · FK error {ep:.1e} m, {eR:.1e} rad · "
                    f"joint-limit margin {margin.min():.3f} rad (q{int(np.argmin(margin)) + 1}) · "
                    f"step from previous pose {step:.4f} rad")
        n_here = sum(self.k in a for a in self.at)
        txt += f"\n{n_here} of {len(self.B)} branches exist at pose {self.k}"
        self.status.set_text(txt)

    def _set_alpha(self, val):
        self.alpha = float(val)
        if self.mesh_coll is not None:
            self.mesh_coll.set_alpha(self.alpha)
            self.fig.canvas.draw_idle()

    # --- animation (blitting) and events ------------------------------------
    # A full redraw takes ~0.3 s, mostly the static timeline and joint panels.
    # While playing, only these artists are redrawn on top of a cached background.
    def _dynamic(self):
        return [self.axr, self.b_cursor, self.q_cursor, self.status, self.s_pose.ax]

    def _on_draw(self, _ev):
        """After every full redraw: cache the background without the moving parts."""
        if self.playing:
            self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            for a in self._dynamic():
                self.fig.draw_artist(a)

    def _blit(self):
        self.fig.canvas.restore_region(self.bg)
        for a in self._dynamic():
            self.fig.draw_artist(a)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause" if self.playing else "Play")
        for a in self._dynamic():                      # animated = left out of full redraws
            a.set_animated(self.playing)
        self.s_pose.drawon = not self.playing          # the slider would force full redraws
        self.bg = None
        if self.playing:
            self.fig.canvas.draw()                     # fills self.bg through _on_draw
            self.timer.start()
        else:
            self.timer.stop()
            self.fig.canvas.draw_idle()

    def _tick(self):
        ks = [x[0] for x in self.B[self.sel]["pts"]]
        k = self.k + 1 if self.k < ks[-1] else ks[0]   # loop over the branch's poses
        self.s_pose.set_val(k)

    def on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return                                     # zoom / pan active
        if ev.inaxes is self.axb and ev.ydata is not None:
            i = int(round(ev.ydata))
            if 0 <= i < len(self.B):
                self.select(i)
                ks = [x[0] for x in self.B[i]["pts"]]
                self.s_pose.set_val(int(np.clip(round(ev.xdata), ks[0], ks[-1])))

    def on_key(self, ev):
        if ev.key == "right":
            self.s_pose.set_val(min(self.k + 1, len(self.Ts) - 1))
        elif ev.key == "left":
            self.s_pose.set_val(max(self.k - 1, 0))
        elif ev.key == "down":
            self.select(self.sel + 1)
        elif ev.key == "up":
            self.select(self.sel - 1)
        elif ev.key == " ":
            self.toggle_play()


def main():
    parser = argparse.ArgumentParser(description="Verify the supplied IK solver and organise branches.")
    parser.add_argument("--steps", type=int, default=60, help="number of poses on the path")
    parser.add_argument("--gui", action="store_true", help="open the interactive branch explorer")
    args = parser.parse_args()

    Ts = default_path(args.steps)
    raw = solve_batch(Ts)
    per_pose, fk_ok_per_pose = [], []
    total = dict(raw=0, fk_fail=0, duplicate=0, limits=0, valid=0, candidates=0)
    for r, T in zip(raw, Ts):
        cands, fk_ok, st = verify_pose(r, T)
        per_pose.append(cands)
        fk_ok_per_pose.append(fk_ok)
        for key in total:
            total[key] += st[key]
    worst_ep = max((c["ep"] for cs in per_pose for c in cs), default=0.0)
    worst_eR = max((c["eR"] for cs in per_pose for c in cs), default=0.0)

    print(f"{len(Ts)} poses · {total['raw']} solver configurations")
    print(f"  poses without IK        : {sum(len(r) == 0 for r in raw)}")
    print(f"  rejected by our FK      : {total['fk_fail']}")
    print(f"  duplicates (mod 2*pi)   : {total['duplicate']}")
    print(f"  outside joint limits    : {total['limits']}")
    print(f"  valid                   : {total['valid']}   "
          f"(worst FK error {worst_ep:.1e} m, {worst_eR:.1e} rad)")
    print(f"  joint-space candidates  : {total['candidates']}   (valid configurations x in-limit 2*pi representatives)")
    print(f"  candidates per pose     : {[len(c) for c in per_pose]}")

    branches = track_branches(per_pose, fk_ok_per_pose)
    print(f"\n{len(branches)} persistent branches")
    print(f"{'id':>3} {'poses':>9} {'labels (S E W)':<16} {'det J':>5} {'q6(0)':>7} {'max |dq|':>9} {'min |det J|':>12}  end")
    for i, b in enumerate(branches):
        pts = b["pts"]
        ks = [x[0] for x in pts]
        qs = np.array([x[1] for x in pts])
        labels = "→".join(dict.fromkeys(x[2] for x in pts))     # names only, may change (see branch_label)
        signs = "→".join(dict.fromkeys(det_sign(x[3]) for x in pts))  # must stay constant
        step = np.abs(np.diff(qs, axis=0)).max() if len(pts) > 1 else 0.0
        print(f"{i:>3} {ks[0]:>4}-{ks[-1]:<4} {labels:<16} {signs:>5} {qs[0, 5]:7.3f} {step:9.4f} "
              f"{min(abs(x[3]) for x in pts):12.4f}  {b['end']}")

    if args.gui:
        import matplotlib.pyplot as plt
        explorer = BranchExplorer(Ts, branches)      # keep a reference: widgets die with it
        plt.show()


if __name__ == "__main__":
    main()