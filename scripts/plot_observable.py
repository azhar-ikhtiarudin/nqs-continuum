"""Plot real-space density maps written by `save_observables`.

Usage:
    uv run python scripts/plot_observable.py                       # runs/density_example
    uv run python scripts/plot_observable.py runs/my_run --show
    uv run python scripts/plot_observable.py runs/my_run --keys n s_z --format pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from moire_mpnqs.observables import load_observables

KEYS = ("n", "n_up", "n_dn", "s_z")
LABELS = {
    "n": r"$n(\mathbf{r})$",
    "n_up": r"$n_\uparrow(\mathbf{r})$",
    "n_dn": r"$n_\downarrow(\mathbf{r})$",
    "s_z": r"$n_\uparrow - n_\downarrow$",
}


def cell_edges(cell, origin, shape):
    """Cartesian bin-corner coordinates for a map of `shape` spanning `cell` (rows = cell vectors)."""
    f1 = np.linspace(0.0, 1.0, shape[0] + 1)
    f2 = np.linspace(0.0, 1.0, shape[1] + 1)
    u, v = np.meshgrid(f1, f2, indexing="ij")
    return np.stack((u, v), axis=-1) @ cell + origin


def draw_cells(ax, primitive, origin, m1, m2, **style):
    """Outline every moire cell in an m1 x m2 tiling."""
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]], dtype=float)
    for i in range(m1):
        for j in range(m2):
            xy = (corners + [i, j]) @ primitive + origin
            ax.plot(xy[:, 0], xy[:, 1], **style)


def plot_density(obs, *, key="n", folded=False, ax=None, vlim=None):
    """Plot one map on `ax` and return the QuadMesh (for a colorbar).

    folded=False: full supercell map, with moire-cell boundaries drawn.
    folded=True : map averaged over moire cells, shown on one primitive cell.
    vlim        : optional (vmin, vmax) so several panels can share a colour scale.
    """
    if key not in KEYS:
        raise ValueError(f"key must be one of {KEYS}")
    arrays, meta = obs["arrays"], obs["meta"]
    density = arrays[f"{key}_folded" if folded else key]
    primitive, origin = arrays["primitive_cell"], arrays["origin"]
    cell = primitive if folded else arrays["supercell"]
    edges = cell_edges(cell, origin, density.shape)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4), layout="constrained")

    if key == "s_z":  # signed quantity: symmetric diverging scale centred on zero
        lim = vlim[1] if vlim else max(float(np.abs(density).max()), 1e-12)
        opts = dict(vmin=-lim, vmax=lim)
    else:
        vmin, vmax = vlim if vlim else (0.0, float(density.max()))
        opts = dict(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(edges[..., 0], edges[..., 1], density,
                         shading="flat", rasterized=True, **opts)

    m1, m2 = (1, 1) if folded else (meta["M1"], meta["M2"])
    draw_cells(ax, primitive, origin, m1, m2, color="white" if key != "s_z" else "k",
            lw=0.6, alpha=0.6)

    ax.set_aspect("equal")
    ax.set_xlabel(r"$x / a_M$")
    ax.set_ylabel(r"$y / a_M$")
    ax.set_title(LABELS[key] + (" (cell-averaged)" if folded else " (supercell)"))
    return mesh


def density_figure(obs, keys=("n", "s_z")):
    """Grid figure: one row per key, columns = (supercell, folded). Each row shares a colour scale."""
    meta = obs["meta"]
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 4.2 * len(keys)),
                            layout="constrained", squeeze=False)
    for row, key in zip(axes, keys):
        full, fold = obs["arrays"][key], obs["arrays"][f"{key}_folded"]
        if key == "s_z":
            lim = max(float(np.abs(full).max()), 1e-12)
            vlim = (-lim, lim)
        else:
            vlim = (0.0, float(full.max()))
        plot_density(obs, key=key, folded=False, ax=row[0], vlim=vlim)
        mesh = plot_density(obs, key=key, folded=True, ax=row[1], vlim=vlim)
        fig.colorbar(mesh, ax=row, label=r"density $[a_M^{-2}]$", shrink=0.9)

    fig.suptitle(
        f"N = {meta['N']} ({meta['n_up']}↑, {meta['n_dn']}↓), "
        f"{meta['M1']}×{meta['M2']} supercell, filling ν = {meta['filling']:g}, "
        f"{meta['n_samples']:,} samples"
    )
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", nargs="?", default="runs/density_example",
                help="directory containing observables.npz / observables.json")
    p.add_argument("--keys", nargs="+", default=["n", "s_z"], choices=KEYS)
    p.add_argument("--out", default=None, help="output file (default: <run_dir>/figures/density.<format>)")
    p.add_argument("--format", default="png", choices=["png", "pdf", "svg"])
    p.add_argument("--dpi", type=int, default=200)
    p.add_argument("--show", action="store_true", help="also open an interactive window")
    args = p.parse_args()

    obs = load_observables(args.run_dir)
    fig = density_figure(obs, keys=args.keys)

    out = Path(args.out) if args.out else Path(args.run_dir) / "figures" / f"density.{args.format}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"saved {out}")

    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
