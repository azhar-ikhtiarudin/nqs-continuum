"""Real-space observables, estimated from continuum VMC samples"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from moire_mpnqs.physics import Lattice

# def _integer(value, name, minimum):
#     """Validate integer-valued configuration arguments."""


class ObservableAccumulator:
    """Accumulate total and spin-resolved density histograms.

    bins_per_cell:
        Number of bins along each primitive-cell direction.
        The supercell map has shape
        (M1 * bins_per_cell, M2 * bins_per_cell).

    origin:
        Cartesian origin of the displayed cell, in a_M.
        The basic repository uses origin=(0, 0).
    """

    def __init__(self, lat: Lattice, n_up: int, n_dn: int,
                 *, bins_per_cell: int = 16, origin=None,):
        self.lat = lat
        self.n_up = n_up
        self.n_dn = n_dn
        self.bins_per_cell = bins_per_cell
        self.N = self.n_up + self.n_dn
        if self.N == 0:
            raise ValueError("at least one particle is required")
        self.origin = (np.zeros(2) if origin is None else np.asarray(origin, dtype=float))
        self.supercell = np.asarray(lat.supercell, dtype=float)
        self.inv_supercell = np.asarray(lat.inv_supercell)
        self.area = float(abs(np.linalg.det(self.supercell)))
        self.nb = (int(lat.M1)*bins_per_cell, int(lat.M2)*bins_per_cell)
        self.bin_area = self.area / np.prod(self.nb)
        self.counts = np.zeros((2, *self.nb), dtype=np.int64)
        self.n_samples = 0
        self.n_batches = 0


    def update(self, samples):
        """Add samples shaped (..., 2*N) or (..., N, 2).

        Leading dimensions can represent chains, draws, or both.
        Every configuration receives equal weight.
        """

        x = np.asarray(samples, dtype=float)
        if x.ndim >=2 and x.shape[-2:] == (self.N, 2):
            positions = x.reshape(-1, self.N, 2)
        if x.ndim >=1 and x.shape[-1] == 2*self.N:
            positions = x.reshape(-1, self.N, 2)
        
        frac = np.mod((positions-self.origin) @ self.inv_supercell, 1.0)
        indices = np.floor(frac * np.asarray(self.nb)).astype(np.int64)
        indices = np.minimum(indices, np.asarray(self.nb) - 1)
        flat_indices = indices[..., 0] * self.nb[1] + indices[..., 1]

        for spin, particle_slice in enumerate((slice(0, self.n_up), slice(self.n_up, self.N))):
            counts = np.bincount(flat_indices[:, particle_slice].ravel(), minlength=int(np.prod(self.nb)))
            self.counts[spin] += counts.reshape(self.nb)
        
        self.n_samples += positions.shape[0]
        self.n_batches += 1

    def finalize(self):
        """Return density maps, geometry, and normalization metadata."""

        n_sigma = self.counts / (self.n_samples * self.bin_area)
        arrays = {"n_up": n_sigma[0], "n_dn": n_sigma[1],
                    "n":n_sigma.sum(axis=0), "s_z": n_sigma[0]-n_sigma[1]}

        b = self.bins_per_cell
        for name in ("n_up", "n_dn", "n", "s_z"):
            arrays[f"{name}_folded"] = arrays[name].reshape(
                int(self.lat.M1), b, int(self.lat.M2), b).mean(axis=(0,2))
        arrays.update(supercell=self.supercell.copy(), primitive_cell=np.asarray([self.lat.a1, self.lat.a2], dtype=float),
                    origin=self.origin.copy())
        n_sites = int(self.lat.M1*self.lat.M2)
        return {
            "arrays": arrays,
            "meta": {
                "N": self.N,
                "n_up": self.n_up,
                "n_dn": self.n_dn,
                "M1": int(self.lat.M1),
                "M2": int(self.lat.M2),
                "bins_per_cell": self.bins_per_cell,
                "n_samples": self.n_samples,
                "n_batches": self.n_batches,
                "bin_area": float(self.bin_area),
                "area_supercell": self.area,
                "length_unit": "a_M",
                "density_unit": "a_M^-2",
                "spin_convention": "s_z = n_up - n_dn",
                "mean_density": float(arrays["n"].mean()),
                "integrated_density": float(
                    arrays["n"].sum() * self.bin_area
                ),
                "integrated_folded_density": float(
                    arrays["n_folded"].sum() * self.bin_area
                ),
                "filling": self.N / n_sites,
            },
        }


def measure(P, *, n_up=None, n_dn=None, n_batches=16, n_samples=None,
            n_discard_per_chain=50, bins_per_cell=16, origin=None):
    """Measure density using fresh samples of the current fixed state.

    This performs sampling only: no optimization or local-energy
    calculation. It advances the sampler state and replaces its cached
    samples, but does not change the wavefunction parameters.

    Explicit spin counts are optional if P contains n_up and n_dn.
    """
    if n_up is None:
        n_up = P.get("n_up")
    if n_dn is None:
        n_dn = P.get("n_dn")
    acc = ObservableAccumulator(P["lat"], n_up, n_dn, bins_per_cell=bins_per_cell, origin=origin)

    for _ in range(n_batches):
        samples = P["vs"].sample(
            n_samples=n_samples,
            n_discard_per_chain=n_discard_per_chain
        )
        acc.update(samples)

    result = acc.finalize()
    result["meta"]["n_discard_per_chain"] = n_discard_per_chain
    result["meta"]["requested_samples_per_batch"] = n_samples
    return result




def save_observables(result, out_dir):
    """Save numerical maps and JSON metadata."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "observables.npz", **result["arrays"])
    (out_dir / "observables.json").write_text(json.dumps(result["meta"], indent=2),encoding="utf-8")

def load_observables(out_dir):
    """Load maps previously written by save_observables."""
    out_dir = Path(out_dir)
    with np.load(out_dir / "observables.npz", allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    meta = json.loads((out_dir / "observables.json").read_text(encoding="utf-8"))
    return {"arrays": arrays, "meta": meta}