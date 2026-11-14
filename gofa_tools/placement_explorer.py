"""
Interactive explorer of the path-placement study (see path_placement.py).
Left: plane z = z0 (IK count, or with 'm' the number of completing branches).
Middle: (q2, q3) lift of every initial branch. Right: robot on the selected branch.
Run it through:  python path_placement.py --gui [--tool INDEX]
"""
import numpy as np

from gofa_tools.ik_solver import solve_batch
from path_placement import (COMPLETE, INK, LIGHT, SEW_COLOR, _frame, _mesh, _seq, closed_targets,
                            contour_task, draw_lifts, draw_plane, follow_branches, outcome_text,
                            tilt_deg)


class PlacementExplorer:
    """Left: plane z = z0 (IK count or, with 'm', the number of completing branches).
    Middle: (q2, q3) lift of every initial branch. Right: robot on the selected branch.
    Click the plane: move the contour · slider / arrows: pose · space: play ·
    up/down or buttons: branch · m: switch the map."""

    def __init__(self, args, name, R0, count, xs_map, ys_map, xs_grid, ys_grid, feas=None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button, Slider
        from gofa_tools.robot_view import load_mesh_model
        self.args, self.name = args, name
        self.T_task = contour_task(args.size, args.n, R0)
        self.count, self.xs_map, self.ys_map = count, xs_map, ys_map
        self.xs_grid, self.ys_grid, self.feas = xs_grid, ys_grid, feas
        self.mesh = load_mesh_model() if args.meshes else None
        self.center, self.sel, self.k, self.playing, self.show_feas = tuple(args.center), 0, 0, False, False
        self.view = (22, -60)

        self.fig = plt.figure(figsize=(17, 8.5))
        self.fig.canvas.manager.set_window_title("ABB GoFa - closed contour placement")
        gs = self.fig.add_gridspec(1, 3, width_ratios=[1.05, 1, 1.1], left=0.05, right=0.98,
                                   top=0.88, bottom=0.33, wspace=0.25)
        self.axp = self.fig.add_subplot(gs[0, 0])
        self.axq = self.fig.add_subplot(gs[0, 1])
        self.axr = self.fig.add_subplot(gs[0, 2], projection="3d")
        self.fig.text(0.05, 0.95, "Closed full-pose rectangle placed in the (x, y) plane", fontsize=15,
                      fontweight="bold", va="center")
        self.fig.text(0.05, 0.92, f"rectangle {args.size[0]:.2f} x {args.size[1]:.2f} m · {args.n} poses · "
                      f"{name} ({tilt_deg(R0):.0f}° tilt) · |det J| ≥ {args.det_min}. "
                      "Click the plane to move the contour · m: switch map", fontsize=10.5, color="0.25",
                      va="center")
        self.status = [self.fig.text(x, 0.25, "", family="monospace", fontsize=9, va="top") for x in (0.05, 0.52)]
        self.s_pose = Slider(self.fig.add_axes([0.10, 0.04, 0.40, 0.025]), "pose k", 0, args.n, valinit=0,
                             valstep=1)
        self.s_pose.on_changed(lambda v: self.set_pose(int(v)))
        self.b_play = Button(self.fig.add_axes([0.55, 0.03, 0.07, 0.045]), "Play")
        self.b_prev = Button(self.fig.add_axes([0.64, 0.03, 0.09, 0.045]), "◀ branch")
        self.b_next = Button(self.fig.add_axes([0.74, 0.03, 0.09, 0.045]), "branch ▶")
        self.b_play.on_clicked(lambda _: self.toggle_play())
        self.b_prev.on_clicked(lambda _: self.select(self.sel - 1))
        self.b_next.on_clicked(lambda _: self.select(self.sel + 1))
        self.timer = self.fig.canvas.new_timer(interval=150)
        self.timer.add_callback(self._tick)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.move(self.center)

    def _draw_plane(self):
        ax = self.axp
        ax.cla()
        if self.show_feas and self.feas is not None:
            from matplotlib.colors import BoundaryNorm, ListedColormap
            Z = (self.feas["out"] == COMPLETE).sum(-1)
            top = max(int(Z.max()), 1)
            cmap = ListedColormap([LIGHT] + _seq(top))
            _mesh(ax, self.feas["xs"], self.feas["ys"], Z, cmap, BoundaryNorm(np.arange(-0.5, top + 1.5), cmap.N))
            _frame(ax, self.xs_map, self.ys_map)
            P = self.T_task[:, :3, 3] + np.array([*self.center, 0])
            ax.plot(np.r_[P[:, 0], P[0, 0]], np.r_[P[:, 1], P[0, 1]], "--", color=INK, lw=1.4)
            ax.set_title("contour centres · number of initial branches that complete", fontsize=10)
        else:
            draw_plane(ax, self.count, self.xs_map, self.ys_map, self.T_task, self.center,
                       self.xs_grid, self.ys_grid)
            ax.set_title(f"plane z = {self.args.z:.2f} m · valid IKs of R0", fontsize=10)
        self.p_cursor, = ax.plot([], [], "o", color="#d64550", ms=8, zorder=5)

    def move(self, center):
        gx = self.xs_grid[np.argmin(np.abs(self.xs_grid - center[0]))]
        gy = self.ys_grid[np.argmin(np.abs(self.ys_grid - center[1]))]
        self.center = (gx, gy)
        self.Ts = closed_targets(self.T_task, gx, gy, self.args.z)
        self.lifts = follow_branches(solve_batch(self.Ts), self.Ts, self.args.det_min)
        self._draw_plane()
        self.select(0)

    def select(self, i):
        self.sel = i % max(len(self.lifts), 1)
        ax = self.axq
        ax.cla()
        if self.lifts:
            draw_lifts(ax, self.lifts, self.sel)
            self.q_cursor, = ax.plot([], [], "o", color=INK, ms=7, zorder=5)
        else:
            self.q_cursor = None
            ax.text(0.5, 0.5, "pose 0 has no valid IK", ha="center", transform=ax.transAxes)
        ax.set_title("initial branches (q2, q3) · colour = S E W · dashed = q6+", fontsize=10)
        self.set_pose(self.k)

    def set_pose(self, k):
        from gofa_tools.robot_view import draw_frame, draw_robot, setup_axes
        self.k = k
        P = self.Ts[:, :3, 3]
        self.p_cursor.set_data([P[k, 0]], [P[k, 1]])
        ax = self.axr
        if ax.has_data():
            self.view = (ax.elev, ax.azim)
        ax.cla()
        ax.plot(*P.T, "--", color="0.5", lw=1.2)
        L = self.lifts[self.sel] if self.lifts else None
        hit = None if L is None else np.flatnonzero(L["ks"] == k)
        if hit is not None and len(hit):
            q = L["q"][hit[0]]
            draw_robot(ax, q, self.mesh, SEW_COLOR[L["sew"]])
            self.q_cursor.set_data([q[1]], [q[2]])
            ax.set_title(f"branch {L['name']} at pose {k} · det J = {L['det'][hit[0]]:+.3f}", fontsize=10)
        else:
            draw_frame(ax, self.Ts[k])
            if self.q_cursor is not None:
                self.q_cursor.set_data([], [])
            ax.set_title("this branch does not reach pose {}".format(k), fontsize=10, color="#d64550")
        setup_axes(ax, self.view)
        self._write_status()
        self.fig.canvas.draw_idle()

    def _write_status(self):
        n_ok = sum(L["code"] == COMPLETE for L in self.lifts)
        lines = [f"centre ({self.center[0]:+.2f}, {self.center[1]:+.2f}, {self.args.z:.2f}) m · "
                 f"{len(self.lifts)} initial branches · {n_ok} complete the contour"]
        for i, L in enumerate(self.lifts):
            lines.append(f"{'▶' if i == self.sel else ' '} {L['name']:<16} {outcome_text(L)}")
        half = 1 + len(lines) // 2
        self.status[0].set_text("\n".join(lines[:half]))
        self.status[1].set_text("\n" + "\n".join(lines[half:]))

    def toggle_play(self):
        self.playing = not self.playing
        self.b_play.label.set_text("Pause" if self.playing else "Play")
        (self.timer.start if self.playing else self.timer.stop)()

    def _tick(self):
        self.s_pose.set_val((self.k + 1) % (self.args.n + 1))

    def on_click(self, ev):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", ""):
            return
        if ev.inaxes is self.axp and ev.xdata is not None:
            self.move((ev.xdata, ev.ydata))

    def on_key(self, ev):
        if ev.key == "right":
            self.s_pose.set_val(min(self.k + 1, self.args.n))
        elif ev.key == "left":
            self.s_pose.set_val(max(self.k - 1, 0))
        elif ev.key == "down":
            self.select(self.sel + 1)
        elif ev.key == "up":
            self.select(self.sel - 1)
        elif ev.key == " ":
            self.toggle_play()
        elif ev.key == "m":
            self.show_feas = not self.show_feas
            self._draw_plane()
            self.set_pose(self.k)
