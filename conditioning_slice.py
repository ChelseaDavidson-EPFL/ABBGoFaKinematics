"""
(q2, q3) conditioning slice of the full ABB GoFa.

Fix q1, q4, q5, q6 at a regular reference configuration and sweep (q2, q3) over
the joint limits. At each point we compute det(0J_t) of the full Jacobian
(tool0, world frame). The singular set is the curve det = 0.

Two slices are drawn with the same colour scale:
  (a) through the reference configuration q_ref (q4 = Q4_REF),
  (b) the same slice with q5 = 0.1, close to the wrist singularity.

Run from the repository root:
    python conditioning_slice.py            -> saves figures/conditioning_slice.pdf/.png
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

parser = argparse.ArgumentParser()
parser.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
parser.add_argument("--out", default="figures")
parser.add_argument("--n", type=int, default=301, help="grid points per axis")
args = parser.parse_args()
sys.path.insert(0, args.repo)

from models.screw import Q_LIMITS          # noqa: E402
from screw_jacobian import jacobian_0Jt     # noqa: E402

# Regular reference configuration [rad]
Q_REF = np.array([0.0, 0.3, -0.6, 0.6, 1.0, 0.0])
Q5_NEAR = 0.1          # q5 close to the wrist singularity

# Diverging ramp: red (det < 0) <- gray (det = 0) -> blue (det > 0)
RAMP = ["#8f2526", "#e34948", "#f3b1af", "#f0efec", "#9ec5f4", "#2a78d6", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("red_gray_blue", RAMP)
INK, MUTED, LIMIT_GRAY = "#1f1f1e", "#6b6b68", "#e4e3df"
C_SING, C_REF = "#1f1f1e", "#ffffff"


def slice_values(q_fixed, q2, q3):
    det = np.zeros((q3.size, q2.size))
    q = q_fixed.copy()
    for i, b in enumerate(q3):
        for j, a in enumerate(q2):
            q[1], q[2] = a, b
            det[i, j] = np.linalg.det(jacobian_0Jt(q))
    return det


def main():
    q2 = np.linspace(Q_LIMITS[1, 0], Q_LIMITS[1, 1], args.n)
    q3 = np.linspace(Q_LIMITS[2, 0], Q_LIMITS[2, 1], args.n)
    slices = {
        r"(a) through $q_{\mathrm{ref}}$, $q_4 = %.1f$" % Q_REF[3]: Q_REF.copy(),
        r"(b) same slice with $q_5 = %.1f$" % Q5_NEAR: np.r_[Q_REF[:4], Q5_NEAR, Q_REF[5]],
    }
    results = {k: slice_values(v, q2, q3) for k, v in slices.items()}
    vmax = max(np.abs(d).max() for d in results.values())

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True, constrained_layout=True)
    Q2, Q3 = np.meshgrid(q2, q3)
    for ax, (title, det) in zip(axes, results.items()):
        im = ax.pcolormesh(Q2, Q3, det, cmap=CMAP, vmin=-vmax, vmax=vmax, shading="auto",
                           rasterized=True)
        ax.contour(Q2, Q3, det, levels=[0], colors=C_SING, linewidths=1.4)
        ax.plot(Q_REF[1], Q_REF[2], marker="o", ms=8, mfc=C_REF, mec=INK, mew=1.5, ls="")
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        ax.set_xlabel(r"$q_2$ [rad]")
        ax.set_xlim(q2[0], q2[-1])
        ax.set_ylim(q3[0], q3[-1])
    axes[0].set_ylabel(r"$q_3$ [rad]")
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label(r"$\det({}^0J_t)$ [m$^3$]")
    cb.outline.set_visible(False)
    handles = [Line2D([], [], color=C_SING, lw=1.4, label=r"$\det({}^0J_t) = 0$"),
               Line2D([], [], marker="o", ms=7, mfc=C_REF, mec=INK, ls="",
                      label=r"$(q_2, q_3)$ of $q_{\mathrm{ref}}$")]
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)

    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"conditioning_slice.{ext}"), dpi=200)

    # Numbers for the report
    print("q_ref =", Q_REF)
    print(f"det(J) at q_ref = {np.linalg.det(jacobian_0Jt(Q_REF)):.4f}")
    for title, det in results.items():
        print(f"{title}: det in [{det.min():.4f}, {det.max():.4f}], max |det| = {np.abs(det).max():.4f}, "
              f"area with |det| < 10% of max = {100 * np.mean(np.abs(det) < 0.1 * np.abs(det).max()):.1f} %")


if __name__ == "__main__":
    main()