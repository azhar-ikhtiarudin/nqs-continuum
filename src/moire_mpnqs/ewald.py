import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc
from moire_mpnqs.physics import Lattice, lattice_vecs
import numpy as np

def ewald_coulomb_2d(x, lattice: Lattice, alpha=None, nreal=5, nrecip=None):

    area = lattice.area_supercell
    alpha = 2.5 / np.sqrt(area) if alpha is None else alpha
    if nrecip is None:
        kmax = 2*alpha*5.5
        nrecip = int(np.ceil(kmax/np.linalg.norm(lattice.G1)*1.2)) + 1

    R = lattice_vecs(lattice.A1, lattice.A2, n=nreal)   # (num_of_images, 2)
    zero_image = np.all(R==0, axis=-1)
    kvecs = lattice_vecs(lattice.G1, lattice.G2, nrecip)
    kvecs = kvecs[np.linalg.norm(kvecs, axis=1) > 1e-12] # drop k=0
    k = np.linalg.norm(kvecs, axis=1)
    kernel = np.pi / area * erfc(k/(2*alpha)) / k # (n_k,)

    x = lattice.wrap(jnp.asarray(x).reshape(-1, 2))              # real-space image sum needs wrapped inputs
    N = x.shape[0]

    # ewald real sum: d_ijR = r_i - r_j + R, exclude only i == j AND R == 0
    displacement = x[:, None, None, :] - x[None, :, None, :] + R[None, None, :, :]
    same_particle = np.eye(N, dtype=bool)
    excluded = same_particle[:, :, None] & zero_image[None, None, :]
    r2 = jnp.sum(displacement**2, axis=-1)
    r = jnp.sqrt(jnp.where(excluded, 1.0, r2))               # replace excluded before sqrt / division
    E_real = 0.5 * jnp.sum(jnp.where(excluded, 0.0, erfc(alpha * r) / r))

    # ewald reciprocal sum: |rho_k|^2 with rho_k = sum_i exp(i k.r_i)
    phase = x @ kvecs.T                                      # (N, n_k)
    rho2 = jnp.sum(jnp.cos(phase), axis=0) ** 2 + jnp.sum(jnp.sin(phase), axis=0) ** 2
    E_recip = jnp.sum(kernel * rho2)

    # self and background corrections
    E_self = -N * alpha / jnp.sqrt(jnp.pi)
    E_background = -(N**2) * jnp.sqrt(jnp.pi) / (area * alpha)

    return E_real + E_recip + E_self + E_background
