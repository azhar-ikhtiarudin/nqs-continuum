import json
import math
import os
import time
from pathlib import Path

import flax
import jax
import jax.numpy as jnp
import netket as nk
import numpy as np
import optax
from flax import nnx

from moire_mpnqs.ewald import ewald_coulomb_2d
from moire_mpnqs.model import MPNQS, kato_cusp
from moire_mpnqs.physics import Lattice, construct_moire_units, moire_potentials, reference_orbitals


def equilibrate(vs, lat, N, n_burnin, *, seed=0, x0=None):
    if x0 is None:
        frac = jax.random.uniform(jax.random.PRNGKey(seed + 7), (vs.sampler.n_chains, N, 2))
        x0 = (frac @ jnp.asarray(lat.supercell)).reshape(vs.sampler.n_chains, N * 2)
    else:
        x0 = jnp.asarray(x0)
        expected = (vs.sampler.n_chains, N * 2)
        if x0.shape != expected:
            raise ValueError(f"Chain shape {x0.shape}; expected {expected}.")

    vs.sampler_state = vs.sampler_state.replace(
        σ=x0, log_prob=vs.sampler.machine_pow * vs.log_value(x0).real
    )
    vs.reset()
    if n_burnin:
        vs.sample(n_discard_per_chain=n_burnin)


def build_problem(
    V0_meV, n_up, n_dn, M=2, *, V0_ref=None, n_samples=512, n_chains=64,
    n_discard_per_chain=8, n_burnin=300, seed=0, model_kwargs=None,
    interaction="coulomb", chunk_size=None, initial=None,
):
    N = n_up + n_dn
    units = construct_moire_units(nu=N / M**2)
    lat = Lattice(M, M)
    hilb = nk.experimental.hilbert.Particle(N=N, geometry=nk.experimental.geometry.FreeSpace(d=2))

    H_kin = nk.operator.KineticEnergy(hilb, mass=1.0 / units.t_meV)
    H_moire = nk.operator.PotentialEnergy(
        hilb, lambda x: jnp.sum(moire_potentials(x.reshape(-1, 2), lat, V0_meV))
    )
    if interaction == "coulomb":
        potential = lambda x: units.e2_meV * ewald_coulomb_2d(x, lat)
    elif interaction == "none":
        potential = lambda x: jnp.zeros((), dtype=x.dtype)
    else:
        raise ValueError(f"Unknown interaction: {interaction!r}")

    H_int = nk.operator.PotentialEnergy(hilb, potential)
    H = H_kin + H_moire + H_int

    reference_V0 = V0_meV if V0_ref is None else V0_ref
    ref = reference_orbitals(lat, n_up, n_dn, reference_V0, units.t_meV)

    kw = dict(model_kwargs or {})
    if kw.get("jastrow", "none") != "none" and kw.get("jastrow_cusp") is None:
        kw["jastrow_cusp"] = kato_cusp(units, interaction)

    model = MPNQS(lat, (n_up, n_dn), ref, rngs=nnx.Rngs(seed), **kw)
    sampler = nk.sampler.MetropolisGaussian(
        hilb, sigma=0.14 / np.sqrt(N), n_chains=n_chains, sweep_size=N
    )
    vs = nk.vqs.MCState(
        sampler, model, n_samples=n_samples, n_discard_per_chain=n_discard_per_chain,
        seed=seed, sampler_seed=seed + 1, chunk_size=chunk_size,
    )

    if initial is not None:
        vs.parameters = flax.serialization.from_state_dict(vs.parameters, initial["params"])

    equilibrate(
        vs, lat, N, min(n_burnin, 50) if initial is not None else n_burnin,
        seed=seed, x0=None if initial is None else initial["chains"],
    )

    return dict(
        H=H, vs=vs, ref=ref, N=N, lat=lat, units=units, H_kin=H_kin, H_moire=H_moire,
        terms=dict(T=H_kin, V_moire=H_moire, V_ee=H_int),
    )


def save_checkpoint(P, path, steps):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(
        params=flax.serialization.to_state_dict(P["vs"].parameters),
        chains=np.asarray(P["vs"].sampler_state.σ),
        steps=int(steps),
    )
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(flax.serialization.msgpack_serialize(payload))
    os.replace(tmp, path)


def load_checkpoint(path):
    path = Path(path)
    if not path.exists():
        return None
    return flax.serialization.msgpack_restore(path.read_bytes())


def run(
    P, n_iter, lr=1e-3, diag_shift=1e-3, print_every=50, *, clip=0.1, history=None,
    offset=0, checkpoint_path=None, checkpoint_every=50,
):
    vs, N = P["vs"], P["N"]
    # Same ordering as nightly.
    optimizer = optax.chain(optax.sgd(lr), optax.clip_by_global_norm(clip))
    driver = nk.driver.VMC_SR(P["H"], optimizer, diag_shift=diag_shift, variational_state=vs)
    completed = offset

    def callback(step, log, drv):
        nonlocal completed
        energy = log["Energy"]
        mean = float(energy.mean.real) / N
        variance = float(energy.variance) / N
        if not np.isfinite(mean) or not np.isfinite(variance):
            raise FloatingPointError(f"Non-finite energy at step {offset + int(step)}.")

        record = dict(
            it=offset + int(step), E=mean, err=float(energy.error_of_mean) / N, var=variance,
            R_hat=float(energy.R_hat), acc=float(drv.state.sampler_state.acceptance),
            t=time.time(),
        )
        if history is not None:
            with Path(history).open("a") as stream:
                stream.write(json.dumps(record) + "\n")

        completed = record["it"] + 1
        if checkpoint_path is not None and checkpoint_every > 0 and completed % checkpoint_every == 0:
            save_checkpoint(P, checkpoint_path, completed)

        if step % print_every == 0 or step == n_iter - 1:
            print(
                f"it {record['it']:5d}  " f"E/N={mean:+.5f} ± {record['err']:.5f}  "
                f"var/N={variance:.4f}  " f"acc={record['acc']:.2f}",
                flush=True,
            )
        return True

    if n_iter > 0:
        driver.run(n_iter=n_iter, callback=callback, show_progress=False)
    if checkpoint_path is not None:
        save_checkpoint(P, checkpoint_path, completed)
    return completed


def evaluate(P, n_samples=4096, *, chunk_size=256, thin=8, n_discard_per_chain=50):
    vs, N = P["vs"], P["N"]
    old_samples = vs.n_samples
    old_chunk = vs.chunk_size
    old_sampler = vs.sampler

    try:
        if thin > 1:
            x0 = np.asarray(vs.sampler_state.σ)
            vs.sampler = old_sampler.replace(sweep_size=old_sampler.sweep_size * int(thin))
            equilibrate(vs, P["lat"], N, 0, x0=x0)

        vs.n_samples = n_samples
        vs.chunk_size = math.gcd(int(vs.n_samples), int(chunk_size))
        vs.sample(n_discard_per_chain=n_discard_per_chain)

        def stats(operator):
            result = vs.expect(operator)
            return dict(
                mean=float(result.mean.real) / N, err=float(result.error_of_mean) / N,
                var=float(result.variance) / N, R_hat=float(result.R_hat),
            )

        return dict(
            E=stats(P["H"]),
            **{name: stats(operator) for name, operator in P["terms"].items()},
            acceptance=float(vs.sampler_state.acceptance),
            n_samples=int(vs.n_samples),
            thin=int(thin),
        )
    finally:
        if thin > 1:
            x0 = np.asarray(vs.sampler_state.σ)
            vs.sampler = old_sampler
            equilibrate(vs, P["lat"], N, 0, x0=x0)
        vs.n_samples = old_samples
        vs.chunk_size = old_chunk