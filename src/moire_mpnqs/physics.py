import numpy as np
import jax.numpy as jnp

HARTREE_meV = 27211.386245988
BOHR_nm = 0.0529177210903

from dataclasses import dataclass
@dataclass
class Units:
    Ha_meV: float
    aB_nm: float
    aM_over_aB: float
    t_meV: float
    e2_meV: float
    rs: float

def construct_moire_units(aM_nm=10.2, eps_r=4.5, m_star=0.7, nu=2) -> Units:
    Ha = HARTREE_meV * m_star / eps_r**2      # effective Hartree [meV]
    aB = BOHR_nm * eps_r / m_star             # effective Bohr radius [nm]
    aM = aM_nm / aB                           # a_M in units of a_B*
    t = Ha / aM**2                            # hbar^2/(m* a_M^2) [meV]
    e2 = Ha / aM                              # Coulomb scale [meV * a_M]
    rho = nu / (np.sqrt(3) / 2)               # electrons per a_M^2
    rs = aM / np.sqrt(np.pi * rho)            # a / a_B*, a = 1/sqrt(pi rho)   (Eq. 4)
    return Units(Ha_meV=Ha, aB_nm=aB, aM_over_aB=aM, t_meV=t, e2_meV=e2, rs=rs)

def lattice_vecs(v1, v2, n):
    indices = np.arange(-n, n+1)
    n1, n2 = np.meshgrid(indices, indices, indexing="ij")
    return (n1[..., None]*np.asarray(v1) + n2[...,None]*np.asarray(v2)).reshape(-1, 2)

class Lattice():

    def __init__(self, M1, M2, aM=1.0):
        self.M1 = M1
        self.M2 = M2
        self.aM = aM
        self.n_sites = M1*M2
        # self.filling = 2
        # self.N = self.filling*M1*M2

        # primitive cell
        self.a1 = aM * jnp.array([1.0, 0.0])
        self.a2 = aM * jnp.array([0.5, jnp.sqrt(3.0) / 2])
        primitive_cell = jnp.stack((self.a1, self.a2), axis=0) # vectors as rows
        self.area_cell = np.abs(np.linalg.det(primitive_cell))

        # reciprocal lattice vectors
        reciprocal = 2*jnp.pi*jnp.linalg.inv(primitive_cell).T
        self.b1, self.b2 = reciprocal[0], reciprocal[1]
        self.b3 = self.b1 + self.b2
        self.b = jnp.stack((self.b1, self.b2, self.b3)) # shortest moire shell
        self.b_ext = np.stack((
            self.b1, self.b2, self.b1+self.b2, 2*self.b1+self.b2, self.b1+2*self.b2,
            self.b2-self.b1, 2*self.b1, 2*self.b2, 2*(self.b1+self.b2)
        ))

        # simulation (super)cell
        self.A1 = M1 * self.a1
        self.A2 = M2 * self.a2
        self.supercell = np.stack((self.A1, self.A2)) # rows: supercell vectors
        self.inv_supercell = np.linalg.inv(self.supercell)
        self.area_supercell = np.abs(np.linalg.det(self.supercell))
        self.G1, self.G2 = 2*np.pi*self.inv_supercell.T # = b1/M1, b2/M2
        self.G12 = np.stack((self.G1, self.G2))
        self.G = np.stack((self.G1, self.G2, self.G1+self.G2)) # shortest supercell shell


    def wrap(self, r):
        f = r @ self.inv_supercell # fractional coordinates
        return (f - jnp.floor(f)) @ self.supercell

    def sites(self):
        n1, n2 = np.meshgrid(np.arange(self.M1), np.arange(self.M2), indexing="ij")
        return n1.reshape(-1, 1)*self.a1 + n2.reshape(-1, 1)*self.a2

def periodic_dist_square(r, Gs):
    # equals |r|**2 for small r, periodic in the lattice of Gs
    G_square = jnp.sum(Gs**2, axis=-1)
    return (4.0/3.0) * jnp.sum((1-jnp.cos(r@Gs.T)) / G_square, axis=-1)

def harmonics(r, Gs):
    ph = r @ Gs.T
    # [sin(G1 . r), ..., sin(Gk . r), cos(G1 . r), ..., cos(Gk . r)]
    return jnp.concatenate((jnp.sin(ph), jnp.cos(ph)), axis=-1)

def moire_potentials(r, lattice: Lattice, V0):
    return -2.0 * V0 * jnp.sum(jnp.cos(jnp.asarray(r) @ lattice.b.T), axis=-1)

def moire_G_basis(lattice: Lattice, radius=5.0):
    Gs = lattice_vecs(lattice.b1, lattice.b2, n=int(np.ceil(radius*1.2))+1)
    nb = np.linalg.norm(lattice.b1)
    Gs = Gs[np.linalg.norm(Gs, axis=1) <= radius*nb + 1e-9]
    return Gs[np.lexsort((np.arctan2(Gs[:,1], Gs[:,0]), np.round(np.linalg.norm(Gs, axis=1), 8)))]

def noninteracting_orbitals(lattice: Lattice, n, V0_meV, t_meV, G_basis):
    nG = len(G_basis)
    if n == 0:
        return np.zeros((0,2)), np.zeros((0, nG), complex), np.zeros(0)
    
    shell = np.concatenate((lattice.b, -lattice.b))
    dG = G_basis[:, None, :] - G_basis[None, :, :]
    couple = np.any(np.linalg.norm(dG[:,:,None,:] - shell[None, None], axis=-1) < 1e-8, axis=-1)
    Vmat = -V0_meV * couple.astype(float)

    states = []
    for n1 in range(lattice.M1):
        for n2 in range(lattice.M2):
            q = n1/lattice.M1*lattice.b1 + n2/lattice.M2*lattice.b2
            # fold the momenta
            cand = q + lattice_vecs(lattice.b1, lattice.b2, 1)
            q = cand[np.argmin(np.round(np.linalg.norm(cand, axis=1), 8) * 1e3 + np.mod(np.arctan2(cand[:,1], cand[:,0]), 2*np.pi))]
            Hq = np.diag(0.5 * t_meV * np.sum((q + G_basis) ** 2, axis=1)) + Vmat
            e, v = np.linalg.eigh(Hq)
            for band in range(min(4, nG)):
                states.append((round(e[band], 9), band, n1, n2, q, v[:, band])) # for each allowed q, retain up to four lowest bands
    states.sort(key=lambda s: (s[0], s[1], s[2], s[3]))
    chosen = states[:n] 
    q = np.stack([s[4] for s in chosen])
    c = np.stack([s[5] for s in chosen]).astype(complex)
    eps = np.array([s[0] for s in chosen])
    return q, c, eps


@dataclass(frozen=True)
class RefOrbitals:
    """Non interacting reference orbitals, used to initialze the model's Bloch coefficients"""
    G_basis: np.ndarray
    q_up: np.ndarray
    c_up: np.ndarray
    eps_up: np.ndarray
    q_dn: np.ndarray
    c_dn: np.ndarray
    eps_dn: np.ndarray


def reference_orbitals(lattice: Lattice, n_up, n_dn, V0_meV, t_meV, G_radius=5.0) -> RefOrbitals:
    G_basis = moire_G_basis(lattice, G_radius)
    q_up, c_up, e_up = noninteracting_orbitals(lattice, n_up, V0_meV, t_meV, G_basis)
    q_dn, c_dn, e_dn = noninteracting_orbitals(lattice, n_dn, V0_meV, t_meV, G_basis)
    return RefOrbitals(G_basis, q_up, c_up, e_up, q_dn, c_dn, e_dn)

