"""UK AI scenario, 1-sector, 2027-2030 — Z RAMPED, gamma stepped.

Mirrors Anthropic's own design (Table 1):
  kappa, alpha, theta RAMP from a mid-2026 anchor to their 2030 values -> Z ramps
  k (automation share) is "held constant"                              -> gamma steps

  epsilon 0.5   both runs        Anthropic Table 1 (gross complements)
  gamma   0.607 -> 0.627         constant automation intensity, stepped once
  Z       1.0 -> 1.031 by 2030   RAMPED: Anthropic Table 3d measured TFP +3.1%

Both paths start from the same 2027 point and fan out.
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

EPS, G_BASE, G_AI, Z_2030 = 1.0, 0.35, 0.389, 1.031
# eps=1.0  OG-UK's own single-sector default; at Cobb-Douglas gamma IS the capital share
# gamma    0.35 -> 0.389  =>  labour share 0.65 -> 0.611 exactly (Anthropic 3.9pp fall)
# Z        ramped to 1.031 (Anthropic Table 3d measured TFP +3.1%)
YEARS = 4                      # 2026 -> 2030
OUT = str(RESULTS / "final1s.json")

cluster = LocalCluster(processes=False, n_workers=1, threads_per_worker=4, dashboard_address=None)
client = Client(cluster)
res = {"assumptions": {"epsilon": EPS, "gamma_base": G_BASE, "gamma_ai": G_AI,
                       "Z_2030": Z_2030, "Z_shape": "linear ramp 2027-2030, flat after"}}
for name, g, ramp in (("baseline", G_BASE, False), ("ai", G_AI, True)):
    t0 = time.time()
    print(f"\n=== {name}: gamma={g}, Z={'ramp->'+str(Z_2030) if ramp else '1.0'} ===", flush=True)
    try:
        d = tempfile.mkdtemp()
        os.makedirs(f"{d}/SS", exist_ok=True); os.makedirs(f"{d}/TPI", exist_ok=True)
        p = api._build_specs(2026, None, d, d, baseline=True, age_specific="pooled",
                             multi_sector=False,
                             param_overrides={"epsilon": [EPS], "gamma": [g]})
        p.TPI_outer_method = "anderson"
        if ramp:
            T = p.T + p.S
            z = np.ones((T, 1))
            up = np.minimum(np.arange(T) / YEARS, 1.0)      # 0 at 2027 -> 1 at 2030, flat after
            z[:, 0] = 1.0 + up * (Z_2030 - 1.0)
            p.Z = z
            print(f"  Z path 2026-2030: {np.round(z[:5,0],5).tolist()}", flush=True)
        ss = SS.run_SS(p, client=client)
        pickle.dump(ss, open(f"{d}/SS/SS_vars.pkl", "wb"))
        print(f"  SS labour share = {float(ss['w']*ss['L']/ss['Y']):.4f}", flush=True)
        p.alpha_G = np.full(p.T + p.S, float(ss["G"] / ss["Y"]))
        TPI.run_TPI(p, client=client)
        tpi = pickle.load(open(f"{d}/TPI/TPI_vars.pkl", "rb"))
        rc = np.abs(np.asarray(tpi["resource_constraint_error"])).reshape(p.T, -1).max(1)
        n = 5
        res[name] = {"gamma": g, "elapsed_min": (time.time()-t0)/60,
                     "ss_labour_share": float(ss["w"]*ss["L"]/ss["Y"]), "rc": rc[:n].tolist(),
                     **{v: np.asarray(tpi[v])[:n].tolist()
                        for v in ("Y","K","L","w","r","C","I","G","D","total_tax_revenue") if v in tpi}}
        print(f"OK {(time.time()-t0)/60:.1f} min | RC max {rc[:n].max():.2e}", flush=True)
    except Exception as e:
        res[name] = {"error": f"{type(e).__name__}: {e}"}
        print(f"FAILED {type(e).__name__}: {e}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1)
print("\nFINAL1S DONE", flush=True)
