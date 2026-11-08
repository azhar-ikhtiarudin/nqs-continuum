from moire_mpnqs.vmc import build_problem, run

V0 = 8.0
M = 2
N_UP, N_DN = 4, 4
N_ITER = 60

P = build_problem(V0, N_UP, N_DN, M=M)
run(P, N_ITER)