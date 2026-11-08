import math
from typing import Any, Callable, Sequence

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
        mat = jnp.exp(jnp.concatenate([orb(r, dr) for orb in self.orbitals], axis=-1))
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


class GraphUpdate(nnx.Module):
    def __init__(
            self, networks: Networks, node_dim: int, edge_dim: int, attention_dim: int,
            *,rngs: nnx.Rngs):
        self.attention_dim = attention_dim
        E = networks.embedding_dim

        self.phi = networks.Net(in_dim=edge_dim)
        self.f = networks.Net(in_dim=node_dim + E)    # [x0 features, hidden]
        self.f2 = networks.Net(in_dim=attention_dim)
        self.g = networks.Net(in_dim=edge_dim + E)

        init = lecun_normal()
        self.wquery = nnx.Param(init(rngs.params(), (edge_dim, attention_dim), jnp.float64))
        self.wkey = nnx.Param(init(rngs.params(), (edge_dim, attention_dim), jnp.float64))

    def __call__(self, x, xij, x0, xij0, final):
        xquery = xij @ self.wquery[...]                # (B, N, N, a)
        xkey = xij @ self.wkey[...]
        logits = jnp.einsum("...ijk,...jlk->...ilk", xquery, xkey) / jnp.sqrt(self.attention_dim)
        weight = self.f2(logits)

        message = self.phi(xij) * weight                # (B, N, N, E)
        xnew = jnp.concatenate((x0, self.f(jnp.concatenate((x, jnp.sum(message, axis=-2)), axis=-1))),
                                axis=-1)                # [x_i^(0), f(x_i, m_i)]
        if not final:
            xij = jnp.concatenate((xij0, self.g(jnp.concatenate((xij, message), axis=-1))), axis=-1)
        return xnew, xij


class MPNN(nnx.Module):
    D1 = 4 + 6 + 1 + 1   # node features: harmonics(G12) + harmonics(b) + |r|_PBC + spin
    D2 = 4 + 1 + 1       # edge features: harmonics(G12) of r_ij + |r_ij|_PBC + s_i s_j

    def __init__(self, lat: Lattice, n_up: int, n_down: int, embedding_dim: int, intermediate_dim: int,
                 num_intermediate: int, attention_dim: int, *, rngs: nnx.Rngs, n_interactions: int = 1):
        self.n_up, self.n_down = n_up, n_down
        self.embedding_dim = embedding_dim
        self.n_interactions = n_interactions

        # triangular-cell constants instead of the square-box L
        self.G12 = jnp.asarray(lat.G12)   # supercell harmonics
        self.b = jnp.asarray(lat.b)       # moiré harmonics (one-body)
        self.G = jnp.asarray(lat.G)       # shortest supercell shell, for the periodic distance

        factory = Networks(intermediate_dim, num_intermediate, embedding_dim, rngs=rngs)
        self.X = nnx.Embed(num_embeddings=2, features=embedding_dim, param_dtype=jnp.float64, embedding_init=lecun_normal(), rngs=rngs)
        node_dim, edge_dim = self.D1 + embedding_dim, self.D2 + embedding_dim
        self.layers = nnx.List([GraphUpdate(factory, node_dim, edge_dim, attention_dim, rngs=rngs)
                                for _ in range(n_interactions)])
        self.out_dim = node_dim

    def __call__(self, r):
        """r: (B, N, 2), already wrapped into the cell -> (B, N, D1 + E)."""
        B, N, _ = r.shape
        E = self.embedding_dim
        emb = self.X(jnp.arange(2))                                       # (2, E)
        s = jnp.concatenate((jnp.ones(self.n_up), -jnp.ones(self.n_down)))
        eye = jnp.eye(N)

        # one-body features: V_M breaks translation invariance
        d1 = jnp.sqrt(periodic_dist_square(r, self.G) + 1e-12)
        x0 = jnp.concatenate((harmonics(r, self.G12), harmonics(r, self.b), d1[..., None],
                            jnp.broadcast_to(s[:, None], (B, N, 1))), axis=-1)

        # edge features periodic in the triangular supercell (was sin(pi/L r) of a square box)
        rij = r[:, :, None, :] - r[:, None, :, :]
        dij = jnp.sqrt(jnp.where(eye > 0, 1.0, periodic_dist_square(rij, self.G))) * (1.0 - eye)  # safe at i=j
        ss = jnp.broadcast_to(jnp.outer(s, s), (B, N, N))
        xij0 = jnp.concatenate((harmonics(rij, self.G12), dij[..., None], ss[..., None]), axis=-1)

        x = jnp.concatenate((x0, jnp.broadcast_to(emb[0], (B, N, E))), axis=-1)
        xij = jnp.concatenate((xij0, jnp.broadcast_to(emb[1], (B, N, N, E))), axis=-1)

        for i, layer in enumerate(self.layers):
            x, xij = layer(x, xij, x0, xij0, final=(i == self.n_interactions - 1))
        return x


class Bloch_Orbital(nnx.Module):
    """log phi_a(r_i) = u_a(r_i) + i k_a.(r_i + dr_i)   (real log-amplitude: no nodes -> one band only)."""

    def __init__(self, lat: Lattice, kvecs, *, rngs: nnx.Rngs, hidden_layers=(32, 32)):
        self.kvecs = jnp.asarray(kvecs)                                   # (n_orbs, 2)
        self.Gs = jnp.asarray(np.concatenate((lat.b_ext, lat.G)))         # moiré + supercell harmonics
        self.u = MLP(in_dim=2 * self.Gs.shape[0], out_dim=self.kvecs.shape[0], hidden_layers=hidden_layers,
                    rngs=rngs, last_init=nnx.initializers.zeros) 

    def __call__(self, r, dr):
        log_plane_wave = 1j * jnp.einsum("ij,...j->...i", self.kvecs, r + dr)
        return self.u(harmonics(r, self.Gs)) + log_plane_wave


class MPNQS(nnx.Module):
    def __init__(
        self,
        lat: Lattice,
        n_per_spin: tuple[int, int],
        ref: RefOrbitals,
        *,
        embedding_dim: int = 32,
        intermediate_dim: int = 32,
        num_intermediate: int = 2,
        attention_dim: int = 32,
        n_interactions: int = 1,
        backflow_init: float = 1e-2,
        rngs: nnx.Rngs,
    ):
        n_up, n_dn = n_per_spin 
        assert n_up <= lat.n_sites and n_dn <= lat.n_sites, "need Bloch orbitals with nodes for >1 band"
        self.n_per_spin = (n_up, n_dn)
        self.N = n_up + n_dn
        self.cell = jnp.asarray(lat.supercell)
        self.inv_cell = jnp.asarray(lat.inv_supercell)
        self.k_vec = jnp.asarray(np.concatenate((ref.q_up, ref.q_dn)))     # (N, 2), column order of D
        orbitals = [Bloch_Orbital(lat, ref.q_up, rngs=rngs)]
        if n_dn > 0:
            orbitals.append(Bloch_Orbital(lat, ref.q_dn, rngs=rngs))
        self.slater = LogSlaterDet(self.n_per_spin, orbitals)

        self.MPNN = MPNN(lat, n_up, n_dn, embedding_dim=embedding_dim, intermediate_dim=intermediate_dim,
                        num_intermediate=num_intermediate, attention_dim=attention_dim,
                        n_interactions=n_interactions, rngs=rngs)
        y_dim = self.MPNN.out_dim                                        

        # orbital-dependent factor exp(net(y_i, k_a)) -- not yet a symmetric Jastrow (no cusp)
        self.net = MLP(in_dim=y_dim + 2, out_dim=1, hidden_layers=(32,), rngs=rngs,
                    last_init=nnx.initializers.zeros)                

        # small backflow at init (lecun_normal gave displacements ~ a_M at step 0)
        init = nnx.initializers.normal(backflow_init)
        self.prealup = nnx.Param(init(rngs.params(), (y_dim, 2), jnp.float64))
        self.pimagup = nnx.Param(init(rngs.params(), (y_dim, 2), jnp.float64))

    def __call__(self, x):
        batch_shape = x.shape[:-1]
        N = self.N
        r = x.reshape(-1, N, 2)
        f = r @ self.inv_cell
        r = (f - jnp.floor(f)) @ self.cell                              
        B = r.shape[0]

        y = self.MPNN(r)                                                   # (B, N, y_dim)
        dr = jnp.einsum("kd,...ik->...id", self.prealup[...] + 1j * self.pimagup[...], y)   # (B, N, 2)

        jas_in = jnp.concatenate((jnp.broadcast_to(y[:, :, None, :], (B, N, N, y.shape[-1])),
                                jnp.broadcast_to(self.k_vec, (B, N, N, 2))), axis=-1)
        orbs = self.net(jas_in)[..., 0]                                    # (B, N, N)

        return self.slater(r, dr, orbs).reshape(batch_shape)