"""Two AI scenarios for the UK, run separately, on a corrected counterfactual.

  anthropic  Korinek et al. (2026) "substantial": labour share falls 3.9pp and
             measured TFP rises 3.1% by 2030.
  obr        OBR March 2026 EFO, technological-displacement case: "higher trend
             productivity fully offsets lower employment to leave the LEVEL of
             GDP unchanged", with a lower labour share and higher profit share.

Both shocks enter the same way: gamma raises the capital weight (automation),
Z carries productivity. They differ in what Z is solved against —

  anthropic  Z solved so the JOINT (gamma, Z) change is a +3.1% technology gain
             at baseline factor prices.
  obr        Z solved so the GDP LEVEL is unchanged at 2030, which is the
             defining feature of their scenario.

Each scenario is run twice, with gamma STEPPED and gamma RAMPED, so the effect
of the time-varying-gamma patch on the analysis can be read directly. The ramp
arms require the firm.py patch (see firm_gamma_tv.patch); the step arms do not.

Counterfactual wiring follows oguk.api.run_transition_path: the baseline writes
its steady state, and every shocked arm inherits it via
``baseline_dir=base_dir, baseline=False`` so initial assets and debt are common
(PR #15 review, finding C1).

Usage:
    python run_scenarios.py                    # all four shocked arms
    python run_scenarios.py --only anthropic   # one scenario, both shapes
    python run_scenarios.py --shapes step      # skip the arms needing the patch
"""

import argparse
import json
import os
import pickle
import platform
import subprocess
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

EPS = 1.0
G_BASE = 0.35
START = 2026
RAMP_YEARS = 4
NPER = 5

# Anthropic Table 3, "substantial", 2030.
ANTHROPIC = {"labour_share_fall_pp": 3.9, "tfp_gain": 0.031}
# OBR March 2026 EFO Box 2.2. GDP level unchanged is the defining feature;
# the labour-share fall is not published as a number, so we impose the same
# automation intensity as the Anthropic arm and let Z do the offsetting.
OBR = {"gdp_level_change": 0.0}


def gamma_for_labour_share_fall(g0, fall_pp):
    """At epsilon=1 the labour share is exactly 1-gamma, so this is algebra."""
    return 1.0 - ((1.0 - g0) - fall_pp / 100.0)


def gamma_only_gain(K, L, g0, g1):
    """Output change at FIXED inputs from moving gamma alone, Z held at 1."""
    return (K ** g1 * L ** (1 - g1)) / (K ** g0 * L ** (1 - g0)) - 1.0


def provenance():
    def ver(pkg):
        try:
            import importlib.metadata as m
            return m.version(pkg)
        except Exception:
            return None

    def sha(path):
        try:
            return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                                  capture_output=True, text=True).stdout.strip() or None
        except Exception:
            return None

    return {
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "packages": {p: ver(p) for p in
                     ("ogcore", "policyengine", "policyengine-uk", "numpy", "pandas")},
        "oguk_git_sha": sha(str(Path(api.__file__).resolve().parents[1])),
        "enforce_solution_checks": TPI.ENFORCE_SOLUTION_CHECKS,
        "note": "ENFORCE_SOLUTION_CHECKS disabled; full RC and Euler diagnostics "
                "are recorded per arm below.",
    }


def diagnostics(tpi, p):
    """Full-path solver diagnostics, not just the first few periods."""
    rc = np.abs(np.asarray(tpi["resource_constraint_error"])).reshape(p.T, -1).max(1)
    out = {
        "rc_first5": rc[:NPER].tolist(),
        "rc_max_interior": float(np.nanmax(rc[1:p.T - 1])),
        "rc_argmax_interior": int(np.nanargmax(rc[1:p.T - 1]) + 1),
        "rc_nonfinite_periods": int((~np.isfinite(rc)).sum()),
    }
    for key, label in (("euler_savings", "euler_savings"),
                       ("euler_labor_leisure", "euler_labour")):
        if key in tpi:
            a = np.abs(np.asarray(tpi[key]))
            out[f"{label}_max"] = float(np.nanmax(a))
    return out


def run_arm(name, gamma_val, ramp, z_terminal, base_dir, client, baseline=False):
    t0 = time.time()
    out_dir = base_dir if baseline else tempfile.mkdtemp()
    for sub in ("SS", "TPI"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    print(f"\n=== {name} ===", flush=True)
    p = api._build_specs(START, None, out_dir, base_dir, baseline=baseline,
                         age_specific="pooled", multi_sector=False,
                         param_overrides={"epsilon": [EPS], "gamma": [gamma_val]})
    p.TPI_outer_method = "anderson"
    T = p.T + p.S
    up = np.minimum(np.arange(T) / RAMP_YEARS, 1.0)[:, None]
    if ramp:
        # time-varying gamma: needs the firm.py patch
        p.gamma = G_BASE + up * (gamma_val - G_BASE)
        print(f"  gamma ramps {G_BASE} -> {gamma_val}", flush=True)
    if z_terminal is not None:
        p.Z = 1.0 + up * (z_terminal - 1.0)
        print(f"  Z ramps 1.0 -> {z_terminal:.6f}", flush=True)
    print(f"  baseline={baseline}, initial state "
          f"{'own' if baseline else 'INHERITED from baseline'}", flush=True)
    ss = SS.run_SS(p, client=client)
    with open(os.path.join(out_dir, "SS", "SS_vars.pkl"), "wb") as f:
        pickle.dump(ss, f)
    p.alpha_G = np.full(p.T + p.S, float(ss["G"] / ss["Y"]))
    TPI.run_TPI(p, client=client)
    with open(os.path.join(out_dir, "TPI", "TPI_vars.pkl"), "rb") as f:
        tpi = pickle.load(f)
    rec = {
        "gamma_terminal": gamma_val, "gamma_shape": "ramp" if ramp else "step",
        "Z_terminal": z_terminal, "elapsed_min": (time.time() - t0) / 60,
        "ss_labour_share": float(ss["w"] * ss["L"] / ss["Y"]),
        "g_n_used": np.asarray(p.g_n)[:NPER].tolist(),
        "g_y_annual_used": float(p.g_y_annual) if np.ndim(p.g_y_annual) == 0
        else float(np.asarray(p.g_y_annual).ravel()[0]),
        "diagnostics": diagnostics(tpi, p),
        **{v: np.asarray(tpi[v])[:NPER].tolist()
           for v in ("Y", "K", "L", "w", "r", "C", "I", "G", "D", "total_tax_revenue")
           if v in tpi},
    }
    print(f"OK {rec['elapsed_min']:.1f} min | SS labour share "
          f"{rec['ss_labour_share']:.4f} | RC interior max "
          f"{rec['diagnostics']['rc_max_interior']:.2e}", flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["anthropic", "obr"], default=None)
    ap.add_argument("--shapes", choices=["step", "ramp", "both"], default="both")
    args = ap.parse_args()

    cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=4,
                           dashboard_address=None)
    client = Client(cluster)
    base_dir = tempfile.mkdtemp()
    for sub in ("SS", "TPI"):
        os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    res = {"provenance": provenance()}
    res["baseline"] = run_arm("baseline", G_BASE, False, None, base_dir, client,
                              baseline=True)
    K0, L0 = res["baseline"]["K"][0], res["baseline"]["L"][0]

    g_ai = gamma_for_labour_share_fall(G_BASE, ANTHROPIC["labour_share_fall_pp"])
    gonly = gamma_only_gain(K0, L0, G_BASE, g_ai)
    # Anthropic: joint (gamma, Z) change = +3.1% technology gain at fixed inputs
    z_anth = (1.0 + ANTHROPIC["tfp_gain"]) / (1.0 + gonly)
    # OBR: GDP level unchanged at fixed inputs, so Z exactly offsets gamma
    z_obr = 1.0 / (1.0 + gonly)

    print(f"\ngamma for a {ANTHROPIC['labour_share_fall_pp']}pp labour-share fall: "
          f"{g_ai:.4f}\ngamma-only output gain at baseline inputs: {gonly*100:+.4f}%"
          f"\n  Z solved, anthropic (+3.1% joint) : {z_anth:.6f}"
          f"\n  Z solved, obr (GDP level unchanged): {z_obr:.6f}", flush=True)

    shapes = [False, True] if args.shapes == "both" else [args.shapes == "ramp"]
    plan = [("anthropic", z_anth), ("obr", z_obr)]
    if args.only:
        plan = [x for x in plan if x[0] == args.only]
    for scen, z in plan:
        for ramp in shapes:
            key = f"{scen}_{'ramp' if ramp else 'step'}"
            try:
                res[key] = run_arm(key, g_ai, ramp, z, base_dir, client)
            except Exception as e:
                res[key] = {"error": f"{type(e).__name__}: {e}"}
                print(f"FAILED {type(e).__name__}: {e}", flush=True)
            json.dump(res, open(RESULTS / "scenarios.json", "w"), indent=1)

    res["assumptions"] = {
        "epsilon": EPS, "gamma_base": G_BASE, "gamma_shocked": g_ai,
        "gamma_only_output_gain": gonly,
        "Z_anthropic": z_anth, "Z_obr": z_obr,
        "anthropic_target": "labour share -3.9pp and +3.1% measured TFP (Table 3)",
        "obr_target": "labour share -3.9pp with the GDP LEVEL unchanged "
                      "(March 2026 EFO Box 2.2)",
        "counterfactual": "shocked arms inherit baseline initial assets and debt",
    }
    json.dump(res, open(RESULTS / "scenarios.json", "w"), indent=1)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
