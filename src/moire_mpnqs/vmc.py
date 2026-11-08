import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from flax import nnx

from moire_mpnqs.ewald import ewald_coulomb_2d
from moire_mpnqs.model import MPNQS
from moire_mpnqs.physics import Lattice, moire_potentials, construct_moire_units, reference_orbitals


def build_problem(V0_meV, n_up, n_dn, M=2, n_samples=128, n_discard_per_chain=8, seed=0):
    units = construct_moire_units()
    lat = Lattice(M, M)
    N = n_up + n_dn
    hilb = nk.experimental.hilbert.Particle(N=N, geometry=nk.experimental.geometry.FreeSpace(d=2))

    H_kin = nk.operator.KineticEnergy(hilb, mass=1.0 / units.t_meV)
    H_moire = nk.operator.PotentialEnergy(
        hilb, lambda x: jnp.sum(moire_potentials(x.reshape(-1, 2), lat, V0_meV)))
    H_int = nk.operator.PotentialEnergy(hilb, lambda x: units.e2_meV * ewald_coulomb_2d(x, lat))
    H = H_kin + H_moire + H_int

    ref = reference_orbitals(lat, n_up, n_dn, V0_meV, units.t_meV)
    model = MPNQS(lat, (n_up, n_dn), ref, rngs=nnx.Rngs(seed))

    sa = nk.sampler.MetropolisGaussian(hilb, sigma=0.14 / np.sqrt(N), n_chains=n_samples, sweep_size=N)
    vs = nk.vqs.MCState(sa, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
                        seed=seed, sampler_seed=seed + 1)

    # start chains uniformly in the cell (FreeSpace default is a Gaussian blob at the origin)
    frac = jax.random.uniform(jax.random.PRNGKey(seed + 7), (sa.n_chains, N, 2))
    x0 = (frac @ jnp.asarray(lat.supercell)).reshape(sa.n_chains, -1)
    vs.sampler_state = vs.sampler_state.replace(σ=x0, log_prob=sa.machine_pow * vs.log_value(x0).real)
    vs.reset()

    return dict(H=H, vs=vs, ref=ref, N=N, H_kin=H_kin, H_moire=H_moire)


def run(P, n_iter, lr=1e-3, diag_shift=1e-3, print_every=10):
    vs, N = P["vs"], P["N"]
    opt = optax.chain(optax.sgd(lr), optax.clip_by_global_norm(0.1))
    drv = nk.driver.VMC_SR(P["H"], opt, diag_shift=diag_shift, variational_state=vs)

    def cb(step, log, driver):
        e = log["Energy"]
        E = float(e.mean.real) / N
        if step % print_every == 0 or step == n_iter - 1:
            acc = float(driver.state.sampler_state.acceptance)
            print(f"it {step:4d}  E/N = {E:+.4f} ± {float(e.error_of_mean) / N:.4f} meV  acc = {acc:.2f}")
        return bool(np.isfinite(E))      # stop if NaN

    drv.run(n_iter=n_iter, callback=cb, show_progress=False)