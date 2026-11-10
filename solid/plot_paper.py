#!/usr/bin/env python3
"""Paper-style postprocessing for solid/driver NPZ output.

Figure 1:
  (a) collective overlap <Q(t)> from the equilibrated reference and saved snapshots
  (b) high-Q zoom
  (c) tau_alpha vs 1/beta, with a high-temperature Arrhenius fit
  (d) relaxation spectrum chi''(omega) from the overlap

Figure 2:
  (a) MSD
  (b,c) van Hove distributions at two betas

The lattice driver stores configurations, not labelled particle trajectories.
Figure 1 is therefore exact for the saved snapshots. Figure 2 is only emitted if
the input NPZ also carries precomputed MSD/van-Hove arrays:

  msd           (nbeta, nt)
  msd_time      (nt,)
  van_hove      (nbeta, nt, nx)
  van_hove_x    (nx,)
  van_hove_time (nt,)       [optional, for printing the selected time]

Accepted Figure-1 input:
  master aging_data.npz from solid/aging_driver.py:
    betas, beta_i_initial, beta_i_equilibrated, beta_i_snapshots,
    beta_i_snapshot_sweeps, ...
  or one beta_*/run.npz:
    initial, equilibrated, snapshots, snapshot_sweeps, scalars
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullLocator, MaxNLocator

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.bbox": "tight",
})

Float = np.float64
_TRAPZ = getattr(np, "trapezoid", getattr(np, "trapz", None))


def load_runs(path: Path) -> list[dict]:
    z = np.load(path, allow_pickle=False)
    if "betas" in z.files:
        out: list[dict] = []
        for i, beta in enumerate(np.asarray(z["betas"], dtype=float)):
            p = f"beta_{i}_"
            if p + "snapshots" not in z.files:
                continue
            d = {"beta": float(beta)}
            for key in ("initial", "equilibrated", "final", "snapshots",
                        "snapshot_sweeps", "energy", "accept_local",
                        "accept_nonlocal", "scalars", "meta"):
                if p + key in z.files:
                    d[key] = z[p + key]
            out.append(d)
        return out
    beta = float(z["scalars"][0]) if "scalars" in z.files else float(z["meta"][0])
    d = {"beta": beta}
    for key in ("initial", "equilibrated", "final", "snapshots",
                "snapshot_sweeps", "energy", "accept_local",
                "accept_nonlocal", "scalars", "meta"):
        if key in z.files:
            d[key] = z[key]
    return [d]


def _types(a: np.ndarray) -> np.ndarray:
    return np.asarray(a, dtype=np.uint8) & 3


def overlap_curve(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return t, mean Q, standard error, per-chain Q.

    Q0 = (N1^2 + N2^2) / (Nocc*N), with Nocc the occupied sites in the
    reference configuration. A vacancy never matches a particle type.
    """
    ref = _types(run.get("equilibrated", run["initial"]))
    snaps = _types(run["snapshots"])
    times = np.asarray(run["snapshot_sweeps"], dtype=np.float64)
    if snaps.ndim != 3:
        raise ValueError("snapshots must have shape (nsnap, chains, N)")
    ns, chains, N = snaps.shape
    q = np.zeros((ns, chains), dtype=np.float64)
    for c in range(chains):
        rc = ref[c]
        n1 = int(np.count_nonzero(rc == 1))
        n2 = int(np.count_nonzero(rc == 2))
        nocc = n1 + n2
        if nocc == 0:
            continue
        q0 = (n1 * n1 + n2 * n2) / float(nocc * N)
        for k in range(ns):
            same = np.count_nonzero((rc != 0) & (snaps[k, c] == rc))
            q[k, c] = (same / float(nocc) - q0) / (1.0 - q0)
    mean = q.mean(axis=1)
    sem = q.std(axis=1, ddof=1) / math.sqrt(chains) if chains > 1 else np.zeros(ns)
    return (np.concatenate(([0.0], times)),
            np.concatenate(([1.0], mean)),
            np.concatenate(([0.0], sem)), q)


def tau_alpha(t: np.ndarray, q: np.ndarray) -> float:
    """First Q <= 1/e crossing with log-log interpolation; NaN if censored."""
    m = (t >= 0) & np.isfinite(q)
    t, q = t[m], q[m]
    if len(t) < 2:
        return float("nan")
    target = math.log(1.0 / math.e)
    if q[0] <= math.exp(target):
        return float(t[0])
    for i in range(1, len(t)):
        if q[i] <= math.exp(target):
            if t[i - 1] <= 0 or t[i] <= 0 or q[i - 1] <= 0 or q[i] <= 0:
                return float(t[i])
            x0, x1 = math.log(t[i - 1]), math.log(t[i])
            y0, y1 = math.log(q[i - 1]), math.log(q[i])
            x = x0 + (target - y0) * (x1 - x0) / (y1 - y0)
            return float(math.exp(x))
    return float("nan")


def stretched_fit(t: np.ndarray, q: np.ndarray) -> tuple[float, float] | None:
    m = (q > 0.05) & (q < 0.95) & (t > 0)
    if np.count_nonzero(m) < 3:
        return None
    x = np.log(t[m])
    y = np.log(-np.log(np.clip(q[m], 1e-12, 1.0 - 1e-12)))
    slope, intercept = np.polyfit(x, y, 1)
    if not np.isfinite(slope) or slope <= 0:
        return None
    return float(math.exp(-intercept / slope)), float(slope)


def spectrum(t: np.ndarray, q: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """chi''(omega) = integral dlogtau (omega*tau/(1+(omega*tau)^2)) dQ/dlogtau."""
    m = (t > 0) & np.isfinite(q)
    t, q = t[m], q[m]
    if len(t) < 2:
        return np.full_like(omega, np.nan)
    fit = stretched_fit(t, q)
    if fit is not None:
        tau, beta = fit
        tmax = max(float(t.max()), 20.0 * tau)
        td = np.geomspace(max(0.5, float(t.min())), tmax, 512)
        qd = np.exp(-np.power(td / tau, beta))
    else:
        td = np.geomspace(max(0.5, float(t.min())), float(t.max()), 512)
        qd = np.interp(np.log(td), np.log(t), q)
    lx = np.log(td)
    dq = np.gradient(qd, lx)
    out = np.empty_like(omega)
    for i, w in enumerate(omega):
        kern = (w * td) / (1.0 + (w * td) ** 2)
        out[i] = -_TRAPZ(dq * kern, lx)
    return out


def _colors(betas: Iterable[float]) -> dict[float, tuple]:
    b = sorted(set(float(x) for x in betas))
    cmap = plt.get_cmap("viridis")
    return {x: cmap(i / max(1, len(b) - 1)) for i, x in enumerate(b)}


def figure_energy(runs: list[dict], out: Path) -> None:
    """Total energy per chain against production sweep for every beta."""
    rows = [r for r in runs if "energy" in r]
    if not rows:
        return
    n = len(rows)
    cols = 4
    nrows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(3.05 * cols, 2.55 * nrows),
                             squeeze=False)
    for i, run in enumerate(rows):
        ax = axes[i // cols][i % cols]
        e = np.asarray(run["energy"], dtype=float)
        if e.ndim == 1:
            e = e[None, :]
        equil = int(np.asarray(run.get("meta", [0, 0, 0, 0, 0]))[4])
        t = np.arange(1, e.shape[1] + 1, dtype=float) + equil
        for c in range(e.shape[0]):
            ax.plot(t, e[c], color="0.70", lw=0.55, alpha=0.45)
        ax.plot(t, e.mean(axis=0), color="C0", lw=1.15)
        ax.set_xlim(t[0] * 0.85, t[-1] * 1.08)
        ax.set_title(rf"$\beta={run['beta']:g}$", fontsize=8.5)
        ax.set_xlabel("production sweep", fontsize=8)
        ax.set_ylabel("total energy", fontsize=8)
        ax.tick_params(labelsize=7)
        if len(t) > 1:
            ax.xaxis.set_major_locator(MaxNLocator(4))
    for j in range(n, nrows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    fig.savefig(out / "total_energy.png", dpi=200)
    fig.savefig(out / "total_energy.pdf")
    plt.close(fig)


def figure1(runs: list[dict], out: Path) -> dict[float, float]:
    runs = sorted(runs, key=lambda r: r["beta"])
    colors = _colors([r["beta"] for r in runs])
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 8.1))
    ax = axes.ravel()
    taus: dict[float, float] = {}
    curves: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for run in runs:
        b = run["beta"]
        c = colors[b]
        t, q, qe, _ = overlap_curve(run)
        m = t > 0
        curves.append((b, t, q, qe))
        taus[b] = tau_alpha(t, q)
        ax[0].errorbar(t[m], q[m], yerr=np.maximum(qe[m], 0),
                       color=c, marker="o", ms=2.6, lw=1.15, capsize=0,
                       label=rf"$\beta={b:g}$")
        ax[1].errorbar(t[m], q[m], yerr=np.maximum(qe[m], 0),
                       color=c, marker="o", ms=2.6, lw=1.15, capsize=0)

    xs = np.concatenate([t[t > 0] for _, t, _, _ in curves])
    xlim = (xs.min() * 0.8, xs.max() * 1.35)
    ax[0].set_xscale("log")
    ax[0].set_xlim(*xlim)
    ax[0].set_xlabel(r"$t$ (sweeps)")
    ax[0].set_ylabel(r"$\langle Q(t)\rangle$")
    ax[0].set_ylim(-0.03, 1.03)
    handles, labels = ax[0].get_legend_handles_labels()
    ax[0].set_title("(a) Collective overlap", loc="left")
    ax[1].set_xscale("log")
    ax[1].set_xlim(*xlim)
    ax[1].set_xlabel(r"$t$ (sweeps)")
    ax[1].set_ylabel(r"$\langle Q(t)\rangle$")
    ax[1].set_ylim(0.80, 1.005)
    ax[1].set_title("(b) Plateau zoom", loc="left")

    pts = [(1.0 / b, taus[b]) for b in taus if np.isfinite(taus[b]) and taus[b] > 0]
    if pts:
        x, y = np.asarray(pts).T
        ax[2].loglog(x, y, "o-", color="black", ms=4, lw=1.1)
        if len(x) >= 3:
            m = x <= np.quantile(x, 0.5)
            if np.count_nonzero(m) >= 2:
                coef = np.polyfit(x[m], np.log10(y[m]), 1)
                xx = np.linspace(x[m].min(), x[m].max(), 80)
                ax[2].loglog(xx, 10 ** np.polyval(coef, xx), "--",
                             color="0.45", lw=1.0, label="Arrhenius fit")
                ax[2].legend(frameon=False, loc="upper left")
        ax[2].set_xlim(x.min() * 0.75, x.max() * 1.3)
    cens = [b for b, v in taus.items() if not np.isfinite(v)]
    if cens:
        ax[2].text(0.02, 0.98, "censored: β=" + ",".join(f"{b:g}" for b in cens),
                   transform=ax[2].transAxes, va="top", fontsize=7)
    ax[2].set_xscale("log")
    ax[2].set_xlabel(r"$1/\beta$")
    ax[2].set_ylabel(r"$\tau_\alpha$ (sweeps)")
    ax[2].set_title("(c) Relaxation time", loc="left")

    for b, t, q, _ in curves:
        m = t > 0
        if np.count_nonzero(m) < 2:
            continue
        w = np.geomspace(0.02 / float(t[m].max()), 20.0 / float(t[m].min()), 90)
        ax[3].loglog(w, spectrum(t, q, w), color=colors[b], lw=1.25,
                     label=rf"$\beta={b:g}$")
    ax[3].set_xlabel(r"$\omega$")
    ax[3].set_ylabel(r"$\chi''(\omega)$")

    ax[3].set_title("(d) Relaxation spectrum", loc="left")

    fig.tight_layout(w_pad=1.4, h_pad=1.2, rect=(0, 0.055, 1, 1))
    fig.legend(handles, labels, frameon=False, ncol=4, loc="lower center",
               bbox_to_anchor=(0.5, 0.002))
    fig.savefig(out / "fig1_overlap_tau_spectrum.png", dpi=200)
    fig.savefig(out / "fig1_overlap_tau_spectrum.pdf")
    plt.close(fig)
    return taus

def figure2(z: np.lib.npyio.NpzFile, out: Path,
            fig2_betas: list[float] | None) -> bool:
    if "msd" not in z.files or "msd_time" not in z.files:
        print("Figure 2 skipped: input has no 'msd'/'msd_time'. "
              "The lattice driver stores configurations, not trajectories.")
        return False
    beta = (np.asarray(z["betas"], dtype=float) if "betas" in z.files
            else np.arange(np.asarray(z["msd"]).shape[0], dtype=float))
    msd = np.asarray(z["msd"], dtype=float)
    mt = np.asarray(z["msd_time"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = _colors(beta)
    for i, b in enumerate(beta):
        axes[0].loglog(mt, msd[i], color=colors[float(b)], lw=1.2,
                       label=f"β={b:g}")
    axes[0].set_xlabel("t (sweeps)")
    axes[0].set_ylabel(r"$\Delta^2(t)$")
    axes[0].legend(frameon=False, fontsize=7)
    axes[0].set_title("(a) MSD", loc="left")

    if "van_hove" in z.files and "van_hove_x" in z.files:
        vh = np.asarray(z["van_hove"], dtype=float)
        x = np.asarray(z["van_hove_x"], dtype=float)
        ti = vh.shape[1] // 2
        if "van_hove_time" in z.files:
            print("van Hove t index", ti, "t =",
                  np.asarray(z["van_hove_time"])[ti])
        idxs = ([int(np.argmin(np.abs(beta - b))) for b in fig2_betas]
                if fig2_betas else [int(np.argmin(beta)), int(np.argmax(beta))])
        for j, i in enumerate(idxs[:2]):
            axes[1 + j].semilogy(x, np.maximum(vh[i, ti], 1e-30),
                                 color=colors[float(beta[i])], lw=1.2)
            axes[1 + j].set_xlabel(r"$\Delta x$")
            axes[1 + j].set_ylabel(r"$P_t(\Delta x)$")
            axes[1 + j].set_title(f"({'b' if j == 0 else 'c'}) β={beta[i]:g}",
                                  loc="left")
    else:
        for j in (1, 2):
            axes[j].axis("off")
            axes[j].text(0.5, 0.5, "van_hove\nmissing", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(out / "fig2_msd_vanhove.png", dpi=180)
    fig.savefig(out / "fig2_msd_vanhove.pdf")
    plt.close(fig)
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, default=Path("paper_plots"))
    ap.add_argument("--betas", type=float, nargs="*", default=None)
    ap.add_argument("--skip-fig1", action="store_true")
    ap.add_argument("--skip-fig2", action="store_true")
    ap.add_argument("--fig2-betas", type=float, nargs="*", default=None)
    args = ap.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.input)
    if args.betas:
        runs = [r for r in runs if any(abs(r["beta"] - b) < 1e-12 for b in args.betas)]
    if not args.skip_fig1:
        taus = figure1(runs, args.output_dir)
        print("tau_alpha:", {f"{b:g}": (None if not np.isfinite(t) else float(t))
                             for b, t in taus.items()})
    figure_energy(runs, args.output_dir)
    if not args.skip_fig2:
        z = np.load(args.input, allow_pickle=False)
        figure2(z, args.output_dir, args.fig2_betas)
    print("wrote", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
