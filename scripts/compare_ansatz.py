"""Compare six orbital/Jastrow variants at one 2x2 Coulomb point."""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BASE = dict(
    M=2,
    n_up=4,
    n_dn=4,
    V0_meV=8.0,
    interaction="coulomb",
    V0_ref=None,
    n_samples=512,
    n_chains=64,
    n_burnin=300,
    n_discard_per_chain=8,
    seed=0,
    chunk_size=None,
)

VARIANTS = {
    "1_single_band": dict(
        model_kwargs=dict(orbital_type="log"),
    ),
    "2_multi_band": dict(
        model_kwargs=dict(orbital_type="bloch_pw"),
    ),
    "3_multi_band_no_ref": dict(
        model_kwargs=dict(orbital_type="bloch_pw"),
        V0_ref=0.0,
    ),
    "4_multi_band_cusp": dict(
        model_kwargs=dict(orbital_type="bloch_pw", jastrow="cusp"),
    ),
    "5_multi_band_cusp_nn": dict(
        model_kwargs=dict(orbital_type="bloch_pw", jastrow="cusp_nn"),
    ),
    "6_single_band_cusp_nn": dict(
        model_kwargs=dict(orbital_type="log", jastrow="cusp_nn"),
    ),
}


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def make_config(name, args):
    problem = dict(BASE)
    problem.update(VARIANTS[name])
    problem["model_kwargs"] = dict(
        problem["model_kwargs"]
    )
    if args.quick:
        problem.update(
            n_samples=64,
            n_chains=16,
            n_burnin=20,
        )

    return dict(
        problem=problem,
        training=dict(
            n_iter=5 if args.quick else args.steps,
            lr=1e-3,
            diag_shift=1e-3,
            clip=0.1,
            checkpoint_every=50,
        ),
        evaluation=dict(
            n_samples=256 if args.quick else 4096,
            chunk_size=256,
            thin=8,
        ),
    )


def trim_history(path, done):
    """Discard records after the last persisted training checkpoint."""
    if not path.exists():
        return
    records = {}
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record["it"] < done:
            records[record["it"]] = record
    text = "".join(
        json.dumps(records[it]) + "\n"
        for it in sorted(records)
    )
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def run_point(config, out):
    from moire_mpnqs.vmc import (
        build_problem,
        evaluate,
        load_checkpoint,
        run,
    )

    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "config.json"
    summary_path = out / "summary.json"

    if config_path.exists():
        previous = json.loads(config_path.read_text())
        if previous != config:
            raise ValueError(
                f"Configuration changed for {out}. "
                "Use a new --root."
            )
    else:
        write_json(config_path, config)

    if summary_path.exists():
        print(f"--- {out.name}: finished, skipping")
        return

    checkpoint_path = out / "checkpoint.mpack"
    history_path = out / "history.jsonl"
    checkpoint = load_checkpoint(checkpoint_path)
    done = 0 if checkpoint is None else int(checkpoint["steps"])

    training = dict(config["training"])
    target = training.pop("n_iter")
    if done > target:
        raise ValueError("Checkpoint exceeds requested step count.")

    trim_history(history_path, done)
    print(f"\n=== {out.name}: {done}/{target} steps", flush=True)

    P = build_problem(
        **config["problem"],
        initial=checkpoint,
    )
    completed = run(
        P,
        target - done,
        **training,
        offset=done,
        history=history_path,
        checkpoint_path=checkpoint_path,
    )
    if completed != target:
        raise RuntimeError(
            f"Training stopped at {completed}, expected {target}."
        )

    energy = evaluate(P, **config["evaluation"])
    if not all(
        np.isfinite(energy["E"][key])
        for key in ("mean", "err", "var")
    ):
        raise FloatingPointError("Non-finite final statistics.")

    summary = dict(
        config=config,
        energy=energy,
        n_params=int(P["vs"].n_parameters),
        E0_nonint=float(
            (P["ref"].eps_up.sum() + P["ref"].eps_dn.sum())
            / P["N"]
        ),
    )
    # Written last: its presence means evaluation completed.
    write_json(summary_path, summary)


def load(names, root):
    results = {}
    for name in names:
        directory = root / name
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue

        summary = json.loads(summary_path.read_text())
        history = [
            json.loads(line)
            for line in (directory / "history.jsonl")
            .read_text().splitlines()
            if line.strip()
        ]
        energy = summary["energy"]["E"]
        results[name] = dict(
            E=energy["mean"],
            err=energy["err"],
            var=energy["var"],
            n_params=summary["n_params"],
            E0=summary["E0_nonint"],
            it=np.array([row["it"] for row in history]),
            curve=np.array([row["E"] for row in history]),
        )
    return results


def print_table(results):
    print(
        f"\n{'variant':<26} {'E/N':>12} {'error':>10} "
        f"{'var/N':>12} {'params':>10}"
    )
    for name in sorted(results, key=lambda n: results[n]["E"]):
        r = results[name]
        print(
            f"{name:<26} {r['E']:12.5f} {r['err']:10.5f} "
            f"{r['var']:12.5f} {r['n_params']:10d}"
        )


def plot(results, destination):
    names = list(results)
    best = min(r["E"] for r in results.values())
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for i, name in enumerate(names):
        r = results[name]
        color = f"C{i}"

        if len(r["curve"]):
            width = max(len(r["curve"]) // 30, 1)
            smooth = np.convolve(
                r["curve"], np.ones(width) / width, mode="valid"
            )
            axes[0].plot(
                r["it"][width - 1:],
                smooth,
                color=color,
                label=name,
            )

        axes[1].errorbar(
            r["E"] - best,
            i,
            xerr=r["err"],
            fmt="o",
            color=color,
            capsize=3,
        )
        axes[2].barh(i, r["var"], color=color)

    axes[0].set(
        xlabel="SR step",
        ylabel="E/N (meV)",
        title="Training: moving average",
    )
    axes[0].legend(fontsize=7)
    axes[1].set(
        xlabel=f"E/N minus best (meV); best={best:.4f}",
        title="Final energy",
    )
    axes[1].axvline(0, color="0.6", linewidth=0.8)
    axes[2].set(
        xlabel="var(E_loc)/N (meV²)",
        title="Local-energy variance",
    )
    if all(results[name]["var"] > 0 for name in names):
        axes[2].set_xscale("log")
    for ax in axes[1:]:
        ax.set_yticks(range(len(names)), names, fontsize=8)

    fig.tight_layout()
    fig.savefig(destination, dpi=150)
    plt.close(fig)
    print(f"Saved {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--only")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")

    suffix = "_quick" if args.quick else ""
    root = args.root or Path(
        f"runs/compare_ansatz{suffix}"
    )
    names = [
        name for name in VARIANTS
        if args.only is None or args.only in name
    ]
    if not names:
        parser.error("--only did not match any variant")

    if not args.plot_only:
        for name in names:
            run_point(make_config(name, args), root / name)

    results = load(names, root)
    if not results:
        print(f"No finished variants under {root}")
        return

    print_table(results)
    plot(results, root / "compare.png")


if __name__ == "__main__":
    main()