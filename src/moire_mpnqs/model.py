import math
from typing import Any, Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.nn.initializers import lecun_normal

from moire_mpnqs.physics import Lattice, RefOrbitals, harmonics, periodic_dist_square

class MLP(nnx.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_layers: tuple[int, ...],
        *,
        rngs: nnx.Rngs,
        activation: Callable = nnx.gelu,
        output_activation: Callable | None = None,
        last_bias: bool = True,
        last_linear: bool = True,
        last_init: Callable | None = None,
        auto_width: bool = False,
        dtype: Any = jnp.float64,
        kwargs: dict | None = None,
    ):
        self.activation = activation
        self.output_activation = output_activation
        self.last_linear = last_linear

        kwargs = kwargs or {}

        if auto_width and len(hidden_layers) == 1:
            n_hidden = hidden_layers[0]
            qs = [ k / ( n_hidden+1 ) for k in range(n_hidden + 2) ]
            dims = [ int(round(in_dim**(1-q) * out_dim**q )) for q in qs ]
        else:
            dims = [in_dim, *hidden_layers, out_dim]

        self.layers = nnx.List()
        for k in range(len(dims)-1):
            last = k == len(dims) - 2
            use_bias = (not last or last_bias or not last_linear)
            kw = dict(kwargs)
            if last and last_init is not None:
                kw.update(kernel_init=last_init, bias_init=nnx.initializers.zeros)
            self.layers.append(nnx.Linear(
                    in_features=dims[k],
                    out_features=dims[k+1],
                    use_bias=use_bias,
                    param_dtype=dtype,
                    rngs=rngs,
                    **kw))

    def __call__(self, x):
        for k, layer in enumerate(self.layers):
            last = k == len(self.layers) - 1
            x = layer(x)
            if last:
                if self.output_activation is not None:
                    x = self.output_activation(x)
                if not self.last_linear:
                    x = self.activation(x)
            else:
                x = self.activation(x)
        return x

class LogSlaterDet(nnx.Module):
    """Block-diagonal (up | down) Slater determinant built from log-orbitals"""
    def __init__(
            self,
            n_per_spin: tuple[int, int],
            orbitals: Sequence[nnx.Module],
            return_signs: bool = False,
            add_signs: bool = True,
    ):
        self.n_per_spin = n_per_spin
        self.orbitals = nnx.List(orbitals)
        self.return_signs = return_signs
        self.add_signs = add_signs

    def __call__(self, r, dr, orbs=None, mask=True):
        assert(r.ndim == 3)
        n_up, n_dn = self.n_per_spin
        N = n_up + n_dn
        mat = jnp.concatenate([jnp.exp(orb(r, dr)) if getattr(orb, "returns_log", True)
                else orb(r, dr) for orb in self.orbitals],axis=-1,)
        # (B, N, N): rows=electrons, columns=[up orbitals | down orbitals]
        if mask:
            m = np.zeros((N, N))
            m[:n_up, :n_up] = 1.0
            m[n_up:, n_up:] = 1.0
            mat = m*mat
        if orbs is not None:
            mat = mat*jnp.exp(orbs)

        signs, logdet = jnp.linalg.slogdet(mat)
        logdet = logdet - 0.5 * math.lgamma(N+1)
        if self.add_signs:
            logdet = logdet + jnp.log(signs)
        return (signs, logdet) if self.return_signs else logdet


class Networks:
    def __init__(
        self, intermediate_dim: int, num_intermediate: int, embedding_dim: int,
        *, rngs: nnx.Rngs, activation: Callable = nnx.gelu):
        """Create networks for the MPNQS
        Args:
            intermediate_dim: dimension of the intermediate coordinates
            num_intermediate: number of hidden layers
            embedding_dim: dimension of the final coordinates of the layer
        """
        self.intermediate_dim = intermediate_dim
        self.num_intermediate = num_intermediate
        self.embedding_dim = embedding_dim
        self.activation = activation
        self.rngs = rngs

    def Net(self, in_dim: int):
        return MLP(
            in_dim=in_dim,
            out_dim=self.embedding_dim,
            hidden_layers=self.num_intermediate * (self.intermediate_dim,),
            activation=self.activation,
            rngs=self.rngs
        )


def envelope_coord(r, dr, envelope):
    """Argument of the periodic orbital envelope u_k (paper eq 7)"""
    return r if envelope == "bare" else r + jnp.real(dr) 
class Bloch_Orbital(nnx.Module):
    """log phi_a(r_i) = u_a(r_i) + i k_a.(r_i + dr_i)   (real log-amplitude: no nodes -> one band only)."""

    def __init__(self, lat: Lattice, kvecs, *, 
                    rngs: nnx.Rngs, hidden_layers=(32, 32), envelope: str = "bare"):
        assert envelope in ("bare", "backflow")
        self.envelope = envelope
        self.kvecs = jnp.asarray(kvecs)                                   # (n_orbs, 2)
        self.Gs = jnp.asarray(np.concatenate((lat.b_ext, lat.G)))         # moiré + supercell harmonics
        self.u = MLP(in_dim=2 * self.Gs.shape[0], out_dim=self.kvecs.shape[0], hidden_layers=hidden_layers,
                    rngs=rngs, last_init=nnx.initializers.zeros) 

    def __call__(self, r, dr):
        log_plane_wave = 1j * jnp.einsum("ij,...j->...i", self.kvecs, r + dr) # r @ kvecs.T
        return self.u(harmonics(
            envelope_coord(r, dr, self.envelope), self.Gs)) + log_plane_wave

class PlaneWaveBlochOrbital(nnx.Module):
    """phi, from bloch theorem"""

    returns_log = False

    def __init__(self, lattice: Lattice, kvecs, coeffs, G_basis, *, rngs: nnx.Rngs, hidden_layers=(32, 32), G_radius: float=3.0, envelope: str = "bare"):
        self.envelope = envelope
        G_basis = np.asarray(G_basis)
        keep = np.linalg.norm(G_basis, axis=1) <= G_radius * np.linalg.norm(lattice.b1) + 1e-9
        c = np.asarray(coeffs)[:, keep]
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        self.G = jnp.asarray(G_basis[keep])

        self.kvecs = jnp.asarray(kvecs)
        self.c_re = nnx.Param(jnp.asarray(c.real.T))
        self.c_im = nnx.Param(jnp.asarray(c.imag.T))
        self.Gs = jnp.asarray(np.concatenate((lattice.b_ext, lattice.G)))
        n_orb = self.kvecs.shape[0]

        self.u = MLP(in_dim=2*self.Gs.shape[0], out_dim=2*n_orb, hidden_layers=hidden_layers,
                    rngs=rngs, last_init=nnx.initializers.zeros)

    def __call__(self, r, dr):
        rplusdr = r + dr
        coeffs = self.c_re[...] + 1j * self.c_im[...]
        bloch = jnp.exp(1j * rplusdr @ self.G.T) @ coeffs # (B, N, n_orb)
        plane_wave = jnp.exp( 1j * jnp.einsum("ij,...j->...i", self.kvecs, rplusdr ))
        u = self.u(harmonics(envelope_coord(r, dr, self.envelope), self.Gs))
        n = self.kvecs.shape[0]
        return jnp.exp(u[..., :n] + 1j*u[..., n:]) * plane_wave * bloch


def kato_cusp(units, interaction: str):
    if interaction.lower() != "coulomb":
        return (0.0, 0.0)
    c = units.e2_meV / units.t_meV
    return (float(c), float(c / 3.0))

class Jastrow(nnx.Module):
    def __init__(self, lattice: Lattice, n_up: int, n_dn: int, ydim: int, cusp, *, 
                kind: str = "cusp_nn", b0: float = 0.1, hidden: int = 32, rngs: nnx.Rngs):
        self.kind = kind
        self.n_up, self.N = n_up, n_up + n_dn
        self.c = jnp.asarray(cusp, dtype=jnp.float64)                       # (anti, parallel), fixed
        inv_softplus = float(np.log(np.expm1(b0)))
        self.b_raw = nnx.Param(jnp.full((2,), inv_softplus))                 # (anti, parallel)
        self.G12 = jnp.asarray(lattice.G12)
        self.Gs = jnp.asarray(lattice.G)
        if kind == "cusp_nn":
            self.w = MLP(in_dim=2 + 3 + 2, out_dim=1, hidden_layers=(hidden,), rngs=rngs,
                        activation=nnx.tanh, last_init=nnx.initializers.zeros)
            self.v = MLP(in_dim=ydim, out_dim=1, hidden_layers=(hidden,), rngs=rngs,
                        activation=nnx.tanh, last_init=nnx.initializers.zeros)

    def __call__(self, r, y=None):
        N = self.N
        s = jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(N - self.n_up)))
        same = (s[:, None] * s[None, :]) > 0                                 # (N, N)
        iu = np.triu_indices(N, k=1)
        rij = r[:, :, None, :] - r[:, None, :, :]
        d2 = periodic_dist_square(rij, self.Gs)
        d = jnp.sqrt(d2 + 1e-24)[:, iu[0], iu[1]]                           # (B, P)
        par = same[iu]
        c = jnp.where(par, self.c[1], self.c[0])
        b = jnp.where(par, jax.nn.softplus(self.b_raw[...][1]), jax.nn.softplus(self.b_raw[...][0]))
        logJ = jnp.sum(c * d / (1.0 + d / b), axis=-1)
        if self.kind == "cusp_nn":
            rp = rij[:, iu[0], iu[1]]                                        # (B, P, 2)
            ss = jnp.broadcast_to(jnp.where(par, 1.0, -1.0), d.shape)[..., None]
            feats = jnp.concatenate((jnp.cos(rp @ self.G12.T), jnp.cos(rp @ self.Gs.T), d[..., None], ss), -1)
            logJ = logJ + jnp.sum(self.w(feats)[..., 0], axis=-1)
            if y is not None:
                logJ = logJ + jnp.sum(self.v(y)[..., 0], axis=-1)
        return logJ

ACTIVATIONS = dict(gelu=nnx.gelu, tanh=nnx.tanh, silu=nnx.silu)
class MoireMPNN(nnx.Module):
    def __init__(self, lat: Lattice, n_up: int, n_dn: int, *, h_dim: int = 16,
                hidden: int = 64, n_layers: int = 3, msg_dim: int | None = None, activation: Callable = nnx.tanh,
                moire_features: bool = False, rngs: nnx.Rngs):
        self.n_up, self.N = n_up, n_up+n_dn
        self.G12 = jnp.asarray(lat.G12)
        self.G = jnp.asarray(lat.G)
        self.b = jnp.asarray(lat.b)
        self.moire_features = moire_features
        self.n_layers = n_layers
        F = msg_dim or h_dim

        d1 = 6 + (6 if moire_features else 0)
        d2 = 6
        def mlp(i, o): return MLP(i, o, (hidden,), rngs=rngs, activation=activation)

        self.f0, self.ft0 = mlp(d1, h_dim), mlp(d2, h_dim)
        gi, gij = d1 + h_dim, d2 + h_dim
        init = lecun_normal()
        self.WQ = nnx.List([nnx.Param(init(rngs.params(), (gij, F), jnp.float64)) for _ in range(n_layers)])
        self.WK = nnx.List([nnx.Param(init(rngs.params(), (gij, F), jnp.float64)) for _ in range(n_layers)])
        self.eta = nnx.List([mlp(gij, F) for _ in range(n_layers)])
        self.f = nnx.List([mlp(gi+F, h_dim) for _ in range(n_layers)])
        self.ft = nnx.List([mlp(gij+F, h_dim) for _ in range(n_layers-1)])
        self.out_dim = gi

    def features(self, r):
        B, N, _ = r.shape
        s = jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(N-self.n_up)))
        d1 = jnp.sqrt(periodic_dist_square(r, self.G) + 1e-12)
        parts = [
            harmonics(r, self.G12),
            d1[...,None],
            jnp.broadcast_to(s[:, None], (B, N, 1))
        ]
        if self.moire_features:
            parts.insert(1, harmonics(r, self.b))
        x0 = jnp.concatenate(parts, axis=-1)
        rij = r[:, :, None, :] - r[:, None, :, :]
        eye = jnp.eye(N)
        dij = ( jnp.sqrt(jnp.where(eye > 0, 1.0, periodic_dist_square(rij, self.G), )) * (1.0 - eye) )
        ss = jnp.broadcast_to(jnp.outer(s, s), (B, N, N))
        xij0 = jnp.concatenate((harmonics(rij, self.G12), dij[..., None], ss[..., None]), axis=-1)
        return x0, xij0

    def __call__(self, r):
        N = r.shape[1]
        x0, xij0 = self.features(r)

        g = jnp.concatenate((x0, self.f0(x0)), axis=-1)
        gij = jnp.concatenate((xij0, self.ft0(xij0)), axis=-1)
        offdiag = (1.0 - jnp.eye(N))[None, :, :, None]

        for l in range(self.n_layers):
            Q = gij @ self.WQ[l][...]
            K = gij @ self.WK[l][...]
            m = nnx.gelu(jnp.einsum("bink,bnjk->bijk", Q, K)) * self.eta[l](gij)
            msum = jnp.sum(m*offdiag, axis=2)
            g = jnp.concatenate((x0, self.f[l](jnp.concatenate((g, msum), axis=-1))), axis=-1)
            if l < self.n_layers - 1:
                gij = jnp.concatenate((xij0, self.ft[l](jnp.concatenate((gij, m), axis=-1))), axis=-1)
        
        return g


class MPNQS(nnx.Module):
    def __init__(
        self,
        lat: Lattice,
        n_per_spin: tuple[int, int],
        ref: RefOrbitals,
        orbital_type: str = "bloch_pw", # "log", "bloch_pw",
        *,
        embedding_dim: int = 32,
        intermediate_dim: int = 32,
        num_intermediate: int = 2,
        attention_dim: int = 32,
        n_interactions: int = 1,
        backflow_init: float = 1e-2,
        rngs: nnx.Rngs,
        orbital_G_radius: float = 3.0,
        orbital_envelope: str = "bare",
        activation: str = "gelu",
        jastrow: str = "none",
        jastrow_cusp: tuple | None = None,
        jastrow_b0: float = 0.1,
        mpnn_kwargs: dict | None = None,
    ):
        n_up, n_dn = n_per_spin 
        assert n_up <= lat.n_sites and n_dn <= lat.n_sites, "need Bloch orbitals with nodes for >1 band"
        self.n_per_spin = (n_up, n_dn)
        self.N = n_up + n_dn
        self.cell = jnp.asarray(lat.supercell)
        self.inv_cell = jnp.asarray(lat.inv_supercell)
        self.k_vec = jnp.asarray(np.concatenate((ref.q_up, ref.q_dn)))     # (N, 2), column order of D
        self.orbital_type = orbital_type

        def make_orbital(q, c):
            if orbital_type == "log":
                return Bloch_Orbital(lat, q, rngs=rngs, envelope=orbital_envelope)
            if orbital_type == "bloch_pw":
                return PlaneWaveBlochOrbital(lat, q, c, ref.G_basis, rngs=rngs, G_radius=orbital_G_radius, envelope=orbital_envelope)
            raise ValueError(f"Unknown orbital_type: {orbital_type!r}")
        
        orbitals = [make_orbital(ref.q_up, ref.c_up)] if n_up > 0 else []
        if n_dn > 0:
            orbitals.append(make_orbital(ref.q_dn, ref.c_dn))
        
        self.slater = LogSlaterDet(self.n_per_spin, orbitals)
        kw = dict(mpnn_kwargs or {})
        kw["activation"] = ACTIVATIONS[kw.get("activation", "tanh")]
        self.MPNN = MoireMPNN(lat, n_up, n_dn, rngs=rngs, **kw)
        
        y_dim = self.MPNN.out_dim                                        

        # orbital-dependent factor exp(net(y_i, k_a)) -- not yet a symmetric Jastrow (no cusp)
        self.net = MLP(in_dim=y_dim + 2, out_dim=1, hidden_layers=(32,), rngs=rngs,
                    last_init=nnx.initializers.zeros)                

        # small backflow at init (lecun_normal gave displacements ~ a_M at step 0)
        init = nnx.initializers.normal(backflow_init)
        self.prealup = nnx.Param(init(rngs.params(), (y_dim, 2), jnp.float64))
        self.pimagup = nnx.Param(init(rngs.params(), (y_dim, 2), jnp.float64))
        self.has_seed = None
        self.jastrow_kind = jastrow
        if jastrow != "none":
            cusp = jastrow_cusp if jastrow_cusp is not None else (0.0, 0.0)
            self.jastrow = Jastrow(lat, n_up, n_dn, y_dim, cusp, kind=jastrow, b0=jastrow_b0, rngs=rngs)

    def __call__(self, x):
        batch_shape = x.shape[:-1]
        N = self.N
        r = x.reshape(-1, N, 2)
        f = r @ self.inv_cell
        r = (f - jnp.floor(f)) @ self.cell                              
        B = r.shape[0]

        y = self.MPNN(r) # (B, N, y_dim)
        dr = jnp.einsum("kd,...ik->...id", self.prealup[...] + 1j * self.pimagup[...], y) # (B, N, 2)

        jas_in = jnp.concatenate((jnp.broadcast_to(y[:, :, None, :], (B, N, N, y.shape[-1])),
                                jnp.broadcast_to(self.k_vec, (B, N, N, 2))), axis=-1)
        orbs = self.net(jas_in)[..., 0] # (B, N, N)

        logpsi = self.slater(r, dr, orbs)
        if self.has_seed:
            logpsi = logpsi + self.seed(r)
        if self.jastrow_kind != "none":
            logpsi = logpsi + self.jastrow(r, y)
        return logpsi.reshape(batch_shape)



