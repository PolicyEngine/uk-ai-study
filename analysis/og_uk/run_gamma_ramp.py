"""FIRST TEST OF THE PATCH: gamma RAMPED 0.35 -> 0.389 over 2026-2030.

Previously impossible — gamma was indexed [m] with no time dimension. With the
firm.py patch it follows Z's convention: [-1] in SS, [: p.T] along the path.

This is Anthropic's design properly: automation BUILDS to 2030 instead of
jumping once, so both paths start from a common origin AND keep the
automation channel.
"""
import json, time, os, pickle, tempfile
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)
import numpy as np
from dask.distributed import Client, LocalCluster
import ogcore.TPI as TPI
TPI.ENFORCE_SOLUTION_CHECKS = False
from ogcore import SS
import oguk.api as api

EPS, G0, G1, Z1, YEARS = 1.0, 0.35, 0.389, 1.031, 4
OUT = str(RESULTS / "ai_scenario_gamma_ramp.json")

cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=4, dashboard_address=None)
client = Client(cluster)
res = {"assum": {"epsilon": EPS, "gamma": f"{G0} ramp-> {G1}", "Z": f"1.0 ramp-> {Z1}"}}
for name, ramp in (("baseline", False), ("ai", True)):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    try:
        d = tempfile.mkdtemp()
        os.makedirs(f"{d}/SS", exist_ok=True); os.makedirs(f"{d}/TPI", exist_ok=True)
        p = api._build_specs(2026, None, d, d, baseline=True, age_specific="pooled",
                             multi_sector=False, param_overrides={"epsilon": [EPS], "gamma": [G0]})
        p.TPI_outer_method = "anderson"
        T = p.T + p.S
        up = np.minimum(np.arange(T) / YEARS, 1.0)[:, None]
        if ramp:
            p.gamma = G0 + up * (G1 - G0)          # (T, 1)  <-- the new capability
            p.Z = 1.0 + up * (Z1 - 1.0)
            print(f"  gamma path 2026-2030: {np.round(p.gamma[:5,0],5).tolist()}", flush=True)
            print(f"  gamma terminal: {p.gamma[-1,0]:.5f}", flush=True)
        else:
            p.gamma = np.full((T, 1), G0)
        ss = SS.run_SS(p, client=client)
        pickle.dump(ss, open(f"{d}/SS/SS_vars.pkl", "wb"))
        print(f"  SS labour share = {float(ss['w']*ss['L']/ss['Y']):.4f}", flush=True)
        p.alpha_G = np.full(p.T + p.S, float(ss["G"] / ss["Y"]))
        TPI.run_TPI(p, client=client)
        tpi = pickle.load(open(f"{d}/TPI/TPI_vars.pkl", "rb"))
        rc = np.abs(np.asarray(tpi["resource_constraint_error"])).reshape(p.T, -1).max(1)
        n = 5
        res[name] = {"elapsed_min": (time.time()-t0)/60, "rc": rc[:n].tolist(),
                     "ss_labour_share": float(ss["w"]*ss["L"]/ss["Y"]),
                     **{v: np.asarray(tpi[v])[:n].tolist()
                        for v in ("Y","K","L","w","r","C","I","G","D","total_tax_revenue") if v in tpi}}
        print(f"OK {(time.time()-t0)/60:.1f} min | RC max {rc[:n].max():.2e}", flush=True)
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("\nGRAMP DONE", flush=True)
