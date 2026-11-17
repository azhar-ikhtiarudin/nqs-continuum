from moire_mpnqs.vmc import build_problem, run
from moire_mpnqs.observables import measure, save_observables

V0 = 8.0
M = 2
N_UP, N_DN = 4, 4
N_ITER = 2

P = build_problem(V0, N_UP, N_DN, M=M)
run(P, N_ITER)


obs = measure(P, n_up=N_UP, n_dn=N_DN, n_batches=16, n_samples=4096,
            n_discard_per_chain=50, bins_per_cell=16)
save_observables(obs, "runs/density_example")

print("Configurations:", obs["meta"]["n_samples"])
print("Integral over supercell:", obs["meta"]["integrated_density"])
print("Expected particle number:", P["N"])
print("Integral over primitive cell:", obs["meta"]["integrated_folded_density"])