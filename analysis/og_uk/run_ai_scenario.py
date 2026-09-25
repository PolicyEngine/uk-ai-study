"""UK AI scenario, 1-sector, 2026-2030 — Anthropic "substantial".

Baseline vs AI as a PROPER counterfactual: the AI arm inherits the baseline's
initial household assets and government debt, and only contemporaneous choices
and the terminal steady state respond. This follows the wiring in
``oguk.api.run_transition_path``:

    baseline:  _build_specs(..., base_dir,   base_dir, baseline=True)
    reform:    _build_specs(..., reform_dir, base_dir, baseline=False)

An earlier version of this script passed ``(d, d, baseline=True)`` for BOTH
arms, which gave the AI economy its own AI steady state as initial conditions
and reset its initial debt. Debt and capital were then already +12.19% and
+22.11% apart at t=0 (PR #15 review, finding C1).

Parameters
----------
epsilon 1.0   both arms. OG-UK's single-sector default; at Cobb-Douglas gamma
              is the capital share, so a labour-share target is exact.
gamma   0.35 -> 0.389   Anthropic Table 3: labour share falls 3.9pp.
Z       1.0  -> Z_AI     solved so that the JOINT (gamma, Z) change delivers
              Anthropic's +3.1% measured TFP at baseline factor prices.

Note on Z: changing gamma alone already raises output at fixed inputs, because
Y = Z*K**gamma*L**(1-gamma). Setting Z = 1.031 on top of the gamma change
therefore overshoots the intended 3.1% technology gain (review finding C2).
``solve_Z_for_tfp_target`` normalises the two jointly.
"""

import json
import os
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
from dask.distributed import Client, LocalCluster

import ogcore.TPI as TPI

TPI.ENFORCE_SOLUTION_CHECKS = False
from ogcore import SS  # noqa: E402
import oguk.api as api  # noqa: E402

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
OUT = str(RESULTS / "ai_scenario_1sector.json")

EPS, G_BASE, G_AI = 1.0, 0.35, 0.389
TFP_TARGET = 0.031          # Anthropic Table 3 panel d, "substantial"
START = 2026
RAMP_YEARS = 4              # 2026 -> 2030


def solve_Z_for_tfp_target(K, L, g0, g1, target):
    """Z such that the joint (gamma, Z) change is a `target` technology gain.

    The gain is measured at fixed baseline inputs, so

        Z * K**g1 * L**(1-g1) / (K**g0 * L**(1-g0)) = 1 + target

    Changing gamma alone moves output even with Z held at 1, which is why Z
    cannot simply be set to 1 + target.
    """
    gamma_only = (K ** g1 * L ** (1 - g1)) / (K ** g0 * L ** (1 - g0))
    return (1.0 + target) / gamma_only


def run(name, gamma, z_path, base_dir, out_dir, baseline, client):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    p = api._build_specs(
        START, None, out_dir, base_dir, baseline=baseline,
        age_specific="pooled", multi_sector=False,
        param_overrides={"epsilon": [EPS], "gamma": [gamma]},
    )
    p.TPI_outer_method = "anderson"
    if z_path is not None:
        p.Z = z_path
        print(f"  Z 2026-2030: {np.round(z_path[:5, 0], 5).tolist()}", flush=True)
    print(f"  baseline={baseline}  base_dir={'own' if base_dir == out_dir else 'INHERITED'}",
          flush=True)
    ss = SS.run_SS(p, client=client)
    with open(os.path.join(out_dir, "SS", "SS_vars.pkl"), "wb") as f:
        pickle.dump(ss, f)
    print(f"  SS labour share = {float(ss['w'] * ss['L'] / ss['Y']):.4f}", flush=True)
    p.alpha_G = np.full(p.T + p.S, float(ss["G"] / ss["Y"]))
    TPI.run_TPI(p, client=client)
    with open(os.path.join(out_dir, "TPI", "TPI_vars.pkl"), "rb") as f:
        tpi = pickle.load(f)
    rc = np.abs(np.asarray(tpi["resource_constraint_error"])).reshape(p.T, -1).max(1)
    n = 5
    print(f"OK {(time.time()-t0)/60:.1f} min | RC max {rc[:n].max():.2e}", flush=True)
    return {
        "gamma": gamma, "elapsed_min": (time.time() - t0) / 60,
        "ss_labour_share": float(ss["w"] * ss["L"] / ss["Y"]),
        "rc": rc[:n].tolist(),
        **{v: np.asarray(tpi[v])[:n].tolist()
           for v in ("Y", "K", "L", "w", "r", "C", "I", "G", "D", "total_tax_revenue")
           if v in tpi},
    }


def main():
    cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=4,
                           dashboard_address=None)
    client = Client(cluster)
    base_dir = tempfile.mkdtemp()
    for sub in ("SS", "TPI"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    res = {}
    # --- baseline: writes the SS the AI arm will inherit from -------------
    res["baseline"] = run("baseline", G_BASE, None, base_dir, base_dir, True, client)

    # --- Z solved against the baseline's own initial inputs ---------------
    K0 = res["baseline"]["K"][0]
    L0 = res["baseline"]["L"][0]
    Z_AI = solve_Z_for_tfp_target(K0, L0, G_BASE, G_AI, TFP_TARGET)
    T_len = None
    print(f"\nZ solved for a {TFP_TARGET:.1%} joint technology gain at baseline inputs:"
          f"\n  gamma-only effect at fixed K,L = "
          f"{(solve_Z_for_tfp_target(K0,L0,G_BASE,G_AI,0.0)**-1 - 1)*100:+.4f}%"
          f"\n  Z_AI = {Z_AI:.6f}  (naive 1+target would be {1+TFP_TARGET:.3f})", flush=True)

    # --- AI arm: own output dir, but INHERITS the baseline as its start ----
    reform_dir = tempfile.mkdtemp()
    for sub in ("SS", "TPI"):
        os.makedirs(os.path.join(reform_dir, sub), exist_ok=True)
    p_tmp = api._build_specs(START, None, reform_dir, base_dir, baseline=False,
                             age_specific="pooled", multi_sector=False,
                             param_overrides={"epsilon": [EPS], "gamma": [G_AI]})
    T = p_tmp.T + p_tmp.S
    up = np.minimum(np.arange(T) / RAMP_YEARS, 1.0)[:, None]
    z_path = 1.0 + up * (Z_AI - 1.0)
    res["ai"] = run("ai", G_AI, z_path, base_dir, reform_dir, False, client)

    res["assumptions"] = {
        "epsilon": EPS, "gamma_base": G_BASE, "gamma_ai": G_AI,
        "tfp_target": TFP_TARGET, "Z_ai_solved": Z_AI,
        "Z_shape": f"linear ramp over {RAMP_YEARS} years, flat after",
        "counterfactual": "AI arm inherits baseline initial assets and debt "
                          "(baseline=False, baseline_dir=base_dir)",
    }
    json.dump(res, open(OUT, "w"), indent=1)
    Db, Da = res["baseline"]["D"][0], res["ai"]["D"][0]
    Kb, Ka = res["baseline"]["K"][0], res["ai"]["K"][0]
    print(f"\nInitial-condition check (should be ~0):"
          f"\n  debt gap at t=0    {(Da/Db-1)*100:+.4f}%"
          f"\n  capital gap at t=0 {(Ka/Kb-1)*100:+.4f}%", flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
