import jax.numpy as jnp

def moire_potential(r, a1, a2, V0):
    primitive_cell = jnp.stack((a1, a2))
    reciprocal = 2 * jnp.pi * jnp.linalg.inv(primitive_cell).T
    b1, b2 = reciprocal[0], reciprocal[1]

    # One member of each +/- pair in the shortest shell.
    G = jnp.stack((b1, b2, b1 + b2))

    phases = jnp.asarray(r) @ G.T
    return -2 * V0 * jnp.sum(jnp.cos(phases), axis=-1)
