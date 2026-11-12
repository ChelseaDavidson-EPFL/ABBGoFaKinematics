"""
3D robot drawing shared by the interactive scripts: URDF STL meshes (with a
stick-robot fallback), the DH skeleton and the tool frame.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from forward_kinematics import fk_poe, _rpy_to_R, _axis_angle_R
from models.dh import DH_TABLE, dh_frames

URDF_PATH = Path(__file__).resolve().parent.parent / "models" / "abb_gofa" / "crb15000_5_95.urdf"
MESH_CELL = 0.008     # [m] vertex-clustering size used to lighten the STLs (0 = full mesh)
D1 = DH_TABLE[0, 2]


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

    def __init__(self, urdf_path=URDF_PATH, cell=MESH_CELL):
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


def load_mesh_model(urdf_path=URDF_PATH, cell=MESH_CELL):
    """URDF mesh model, or None (stick robot) if the meshes cannot be loaded."""
    try:
        model = UrdfMeshModel(urdf_path, cell)
        return model if model.links else None
    except Exception as exc:                              # fall back to the stick robot
        print(f"[warn] could not load URDF meshes ({exc}); using the stick robot")
        return None


# ============================================================================
# Drawing
# ============================================================================
def draw_frame(ax, T, length=0.12, lw=2.5):
    """x, y, z axes of a 4x4 frame as red, green, blue segments."""
    for k, col in enumerate("rgb"):
        a = T[:3, 3] + length * T[:3, k]
        ax.plot(*np.c_[T[:3, 3], a], color=col, lw=lw, zorder=11)


def draw_robot(ax, q, mesh=None, color="C0", alpha=1.0):
    """Robot at configuration q: meshes (if any) + DH skeleton + tool frame.
    Returns the mesh collection (None for the stick robot) so its alpha can be changed."""
    coll = None
    if mesh is not None:
        tris, cols = mesh.world_triangles(q)
        coll = Poly3DCollection(tris, facecolors=cols, edgecolors="none", linewidths=0, alpha=alpha)
        ax.add_collection3d(coll)
        lw, ms = 2.5, 4                                  # thin skeleton on top of the meshes
    else:
        ax.plot([0, 0], [0, 0], [0, D1], color="0.4", lw=9, alpha=0.5)
        lw, ms = 5, 6
    pts = np.array([[0, 0, 0]] + [F[:3, 3] for F in dh_frames(q)[1:]])
    ax.plot(*pts.T, "-o", color=color, lw=lw, ms=ms, mfc="k", solid_capstyle="round",
            zorder=10, alpha=0.35 if (mesh is not None and alpha > 0.6) else 1)
    draw_frame(ax, fk_poe(q))
    return coll


def setup_axes(ax, view=(18, -70), lim=0.75, zlim=(-0.45, 1.05)):
    """Ground cross, limits, equal-ish aspect and labels of the 3D robot view."""
    ax.plot([-lim, lim], [0, 0], [0, 0], color="0.75", lw=0.8)
    ax.plot([0, 0], [-lim, lim], [0, 0], color="0.75", lw=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect((2 * lim, 2 * lim, zlim[1] - zlim[0]), zoom=1.15)
    ax.view_init(elev=view[0], azim=view[1])
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
