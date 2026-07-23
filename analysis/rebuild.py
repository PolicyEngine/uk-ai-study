#!/usr/bin/env python3
"""Attested clean-room rebuild driver for results/ (issue #1, R2-1).

Replaces the ad-hoc regenerate_all.sh sequence with a complete, explicit
manifest/DAG of every tracked file under results/ and the analysis script
that produces it.

Modes
-----
  python analysis/rebuild.py --dry-run     Print the topologically ordered
                                           DAG, per-stage commands, declared
                                           outputs and dependencies. No work.
  python analysis/rebuild.py --check       Verify the current results/ tree
                                           against the manifest (and, if
                                           present, against
                                           results/BUILD_MANIFEST.json
                                           hashes). Exits non-zero and lists
                                           every missing, empty, unmapped
                                           (tracked-but-not-in-manifest) and
                                           stale file.
  python analysis/rebuild.py               Full clean-room rebuild:
                                           1. `git worktree add --detach` a
                                              pristine checkout of HEAD into
                                              a temp build root;
                                           2. symlink the (untracked) data/
                                              directory into it;
                                           3. run every stage in topological
                                              order with cwd = build root, so
                                              all scripts' relative
                                              Path("results"/...) writes land
                                              in the temp root, never in the
                                              working tree;
                                           4. validate all manifest outputs
                                              exist and are non-empty;
                                           5. atomically publish: one
                                              `rsync -a --delete` of the
                                              built results/ over the repo's
                                              results/ (non-generated files
                                              preserved via copy-in before
                                              publish);
                                           6. write results/BUILD_MANIFEST.json
                                              (git commit, timestamp, per-file
                                              sha256, per-script sha256, seed
                                              policy).
  Options: --stages A B   run only these stages (plus nothing else; deps are
                          NOT auto-added — for debugging),
           --keep-build   keep the temp worktree on failure/success.

Design notes
------------
* Scripts are never edited: every analysis script resolves output paths
  either relative to cwd (Path("results"/...)) or relative to its own file
  (ROOT = parents[1]). Both resolve inside the worktree build root, so the
  clean-room build needs no RESULTS_DIR plumbing and is immune to the
  parallel edits in uk_ai_study/shocks.py, incidence_scenarios.py and
  tax_composition.py: the manifest declares file NAMES and ordering only,
  not file contents, seeds beyond the scripts' own fixed conventions, or
  scenario internals.
* Stage stdout/stderr is teed to results/robustness/sensitivity_run.log in
  the build root (that tracked log file is thereby regenerated too).
* results/EXTENSIONS_SUMMARY.md is hand-written documentation, not a build
  product; it is declared NON_GENERATED and carried through publish.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ATTESTATION = "BUILD_MANIFEST.json"
BUILD_LOG = "robustness/sensitivity_run.log"

# Tracked files under results/ that no analysis script produces (documentation
# etc.). Preserved verbatim across an atomic publish.
NON_GENERATED = []

PY = sys.executable or "python3"

# ---------------------------------------------------------------------------
# Manifest: stage -> command, declared outputs (relative to results/),
# declared data inputs (relative to repo root), and upstream stage deps
# ("after" = stages whose outputs this stage reads or overwrites).
# NOTE: the preset list mirrors uk_ai_study.shocks.PRESETS at the time of
# writing; run_all.py iterates PRESETS at runtime, and --check will flag any
# divergence (missing or extra tracked files), so a PRESETS change surfaces
# loudly rather than silently.
# ---------------------------------------------------------------------------
FRS_INPUTS = ["data/frs_2024_25.h5", "data/frs_2024_25/UKDA-9563-tab/tab/adult.tab"]

MANIFEST = [
    {
        "stage": "presets",
        "cmd": [PY, "analysis/run_all.py"],
        "outputs": [
            "central.json", "low.json", "high.json",
            "central_youth_tilted.json",
            "central_ripple.json", "central_ripple_low.json",
            "central_ripple_high.json",
            "wage_margin_central.json", "wage_margin_pss.json",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "jr16_figs",
        "cmd": [PY, "analysis/replicate_jr16.py", "figs"],
        "outputs": [
            "jr16/fig4_1_transition_by_decile.csv",
            "jr16/fig4_2_wage_gain_by_decile.csv",
            "jr16/fig4_3_capital_by_decile.csv",
            "jr16/fig4_4_decomposition.csv",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "jr16_grid",
        "cmd": [PY, "analysis/replicate_jr16.py", "grid"],
        "outputs": ["jr16/grid.csv"],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "incidence",
        "cmd": [PY, "analysis/incidence_scenarios.py"],
        "outputs": [
            "incidence/exposure.json", "incidence/junior.json",
            "incidence/uniform.json", "incidence/compression.json",
            "incidence/uniform_compound.json",
            "incidence/compression_compound.json",
            "incidence/summary.csv",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "measured_incidence",
        "cmd": [PY, "analysis/measured_incidence.py"],
        "outputs": ["incidence/klein_top_loaded.json", "incidence/summary_five.csv"],
        "inputs": FRS_INPUTS,
        "after": ["incidence"],  # extends the four-family summary to five
    },
    {
        "stage": "appendix_fast",
        "cmd": [PY, "analysis/appendix.py", "fast"],
        "outputs": [
            "appendix/baseline_distributions.csv",
            "appendix/b1_market_income_less_capital.png",
            "appendix/b2_capital_income.png",
            "appendix/b3_benefits.png",
            "appendix/b4_tax_and_ni.png",
            "appendix/b5_disposable_income.png",
            "appendix/exposure_incidence_by_decile.png",
            "appendix/complementarity_incidence_by_decile.png",
            "appendix/job_loss_by_major_group.csv",
            "appendix/b7_uniform_vs_ai_decile.csv",
            "appendix/b7_uniform_vs_ai_decile.png",
            "appendix/b8_alternative_index_decile.csv",
            "appendix/b8_alternative_index_decile.png",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "appendix_decomp",
        "cmd": [PY, "analysis/appendix.py", "decomp"],
        "outputs": ["appendix/decomposition_ci.csv"],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "appendix_grids",
        "cmd": [PY, "analysis/appendix.py", "grids"],
        "outputs": ["appendix/grid_deciles_capital.csv"],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "robustness",
        "cmd": [PY, "analysis/robustness.py", "all"],
        "outputs": [
            "robustness/central_draws.csv",
            "robustness/exposure_sensitivity.json",
            "robustness/uniform_comparator.json",
            "robustness/age_exposure_moments.csv",
            "robustness/frs_vs_ashe_occupation.csv",
            # also writes robustness/central_monte_carlo.json, which the
            # monte_carlo stage recomputes and owns (declared there).
        ],
        "inputs": FRS_INPUTS,
        "after": [],
        "log": True,
    },
    {
        "stage": "monte_carlo",
        "cmd": [PY, "analysis/monte_carlo_families.py"],
        "outputs": [
            "robustness/incidence_draws_five.csv",
            "robustness/policy_draws.csv",
            "robustness/incidence_monte_carlo.json",
            "robustness/policy_monte_carlo.json",
            "robustness/central_monte_carlo.json",
        ],
        "inputs": FRS_INPUTS,
        "after": ["robustness"],  # supersedes robustness' central_monte_carlo.json
        "log": True,
    },
    {
        "stage": "index_sensitivity",
        "cmd": [PY, "analysis/index_sensitivity.py"],
        "outputs": [
            "robustness/index_sensitivity_full.json",
            "robustness/index_sensitivity.png",
        ],
        "inputs": FRS_INPUTS + ["data/raw"],
        "after": [],
        "log": True,
    },
    {
        "stage": "duration_takeup",
        "cmd": [PY, "analysis/sensitivity_duration_takeup.py"],
        "outputs": [
            "robustness/duration_takeup_draws.csv",
            "robustness/duration_takeup_sensitivity.json",
        ],
        "inputs": FRS_INPUTS,
        "after": ["monte_carlo"],  # reads robustness/incidence_draws_five.csv
        "log": True,
    },
    {
        "stage": "duration_reforms",
        "cmd": [PY, "analysis/sensitivity_duration_reforms.py"],
        "outputs": [
            "robustness/duration_reform_draws.csv",
            "robustness/duration_reform_ranking.json",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
        "log": True,
    },
    {
        "stage": "wage_tier",
        "cmd": [PY, "analysis/sensitivity_wage_tier.py"],
        "outputs": [
            "robustness/measured_wage_tier_draws.csv",
            "robustness/measured_wage_tier_sensitivity.json",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
        "log": True,
    },
    {
        "stage": "mixed_wage_adjustment",
        "cmd": [PY, "analysis/sensitivity_mixed_wage_adjustment.py"],
        "outputs": [
            "robustness/mixed_adjustment.csv",
            "robustness/survivor_wage_grid.csv",
            "robustness/mixed_wage_adjustment.json",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
        "log": True,
    },
    {
        "stage": "enhanced_dataset",
        "cmd": [PY, "analysis/sensitivity_enhanced_dataset.py"],
        "outputs": [
            "robustness/enhanced_dataset_cells.csv",
            "robustness/enhanced_dataset_central.json",
        ],
        "inputs": FRS_INPUTS + ["data/enhanced_frs_2023_24.h5"],
        "after": ["presets"],  # reads results/central.json
        "log": True,
    },
    {
        "stage": "paper_scenarios",
        "cmd": [PY, "analysis/paper_scenarios.py"],
        "outputs": [
            "paper_scenarios/brynjolfsson_canaries_2025.json",
            "paper_scenarios/hosseini_lichtinger_2026.json",
            "paper_scenarios/hosseini_lichtinger_2026_adopter_scaled.json",
            "paper_scenarios/klein_teeselink_2025.json",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "policy",
        "cmd": [PY, "analysis/policy_counterfactuals.py"],
        "outputs": [
            "policy/R1_wage_insurance_exposure.json",
            "policy/R1_wage_insurance_junior.json",
            "policy/R1_wage_insurance_uniform.json",
            "policy/R2_uc_circuit_breaker_exposure.json",
            "policy/R3_cap_suspension_taper_cut_exposure.json",
            "policy/summary.csv",
            "policy/policy_reforms.png",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "caseloads",
        "cmd": [PY, "analysis/caseloads.py"],
        "outputs": [
            "caseloads/central.json", "caseloads/low.json", "caseloads/high.json",
            "caseloads/incidence_exposure.json", "caseloads/incidence_junior.json",
            "caseloads/incidence_uniform.json", "caseloads/incidence_compression.json",
            "caseloads/summary.csv", "caseloads/caseloads.png",
        ],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "tax_composition",
        "cmd": [PY, "analysis/tax_composition.py"],
        "outputs": [
            "tax_composition/composition_grid.csv",
            "tax_composition/recycling_case.json",
            "tax_composition/revenue_shortfall_phi.png",
        ],
        "inputs": FRS_INPUTS,
        "after": ["jr16_grid"],  # reads results/jr16/grid.csv
    },
    {
        "stage": "gender",
        "cmd": [PY, "analysis/gender.py"],
        "outputs": ["appendix/gender_incidence.json"],
        "inputs": FRS_INPUTS,
        "after": [],
    },
    {
        "stage": "geo_impact",
        "cmd": [PY, "analysis/geo_impact.py"],
        "outputs": [
            "geo/constituency_impacts.csv",
            "geo/region_summary.csv",
            "geo/hexmap_income_change.png",
            "geo/hexmap_displacement.png",
            "geo/imputation_notes.json",
        ],
        "inputs": FRS_INPUTS + [
            "data/enhanced_frs_2023_24.h5",
            "data/parliamentary_constituency_weights.h5",
            "data/constituencies_2024.csv",
        ],
        "after": [],
    },
    {
        "stage": "geo_choropleth",
        "cmd": [PY, "analysis/geo_choropleth.py"],
        "outputs": ["geo/map_income_change.png", "geo/map_displacement.png"],
        "inputs": ["data/uk_constituencies_2024.geojson"],
        "after": ["geo_impact"],  # reads results/geo/constituency_impacts.csv
    },
    {
        "stage": "figures",
        "cmd": [PY, "analysis/figures.py"],
        "outputs": [
            "jr16/fig4_1_transition.png", "jr16/fig4_2_wages.png",
            "jr16/fig4_3_capital.png", "jr16/fig4_4_decomposition.png",
            "jr16/fig4_5_disposable_grid.png", "jr16/fig4_6_exchequer_grid.png",
            "jr16/fig4_7_gini_grid.png",
            "appendix/b9_grid_by_decile.png",
            "appendix/b11_grid_no_capital.png",
            "appendix/decomposition_ci.png",
            "incidence/incidence_families.png",
        ],
        "inputs": [],
        # presentation layer: reads jr16 CSVs, appendix CSVs, incidence JSONs
        "after": ["jr16_figs", "jr16_grid", "appendix_decomp",
                  "appendix_grids", "incidence", "measured_incidence"],
    },
    {
        "stage": "tex_values",
        "cmd": [PY, "analysis/emit_tex_values.py"],
        "outputs": [],
        "repo_outputs": ["paper/values_generated.tex"],
        "inputs": [],
        "after": [
            "presets", "jr16_figs", "jr16_grid", "incidence",
            "measured_incidence", "monte_carlo", "policy", "caseloads",
            "tax_composition", "enhanced_dataset", "figures",
        ],
    },
    {
        "stage": "paper_pdf",
        "cmd": ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error",
                "-cd", "paper/main.tex"],
        "outputs": [],
        "repo_outputs": ["paper/main.pdf"],
        "inputs": [],
        "after": ["tex_values", "geo_choropleth"],
    },
]


# ---------------------------------------------------------------------------
# DAG utilities
# ---------------------------------------------------------------------------
def topo_order(manifest):
    by_name = {s["stage"]: s for s in manifest}
    order, seen, visiting = [], set(), set()

    def visit(name):
        if name in seen:
            return
        if name in visiting:
            raise SystemExit(f"cycle in manifest at stage {name}")
        visiting.add(name)
        for dep in by_name[name]["after"]:
            visit(dep)
        visiting.discard(name)
        seen.add(name)
        order.append(name)

    for s in manifest:
        visit(s["stage"])
    return [by_name[n] for n in order]


def all_outputs(manifest):
    out = []
    for s in manifest:
        out.extend(s["outputs"])
    dupes = {o for o in out if out.count(o) > 1}
    if dupes:
        raise SystemExit(f"manifest declares duplicate outputs: {sorted(dupes)}")
    return out


def tracked_results_files():
    txt = subprocess.run(
        ["git", "ls-files", "results/"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    return [line[len("results/"):] for line in txt.splitlines() if line.strip()]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
def dry_run(manifest):
    ordered = topo_order(manifest)
    outputs = all_outputs(manifest)
    tracked = tracked_results_files()
    generated_tracked = [t for t in tracked
                         if t not in NON_GENERATED
                         and t not in (ATTESTATION, BUILD_LOG)]
    mapped = [t for t in generated_tracked if t in outputs]
    print(f"DAG: {len(ordered)} stages, {len(outputs)} declared outputs")
    print(f"Coverage: {len(mapped)}/{len(generated_tracked)} generated tracked "
          f"files mapped ({len(NON_GENERATED)} declared non-generated: "
          f"{NON_GENERATED}; build log: {BUILD_LOG})")
    for i, s in enumerate(ordered, 1):
        after = f"  after: {', '.join(s['after'])}" if s["after"] else ""
        print(f"\n[{i:2d}] {s['stage']}{after}")
        print(f"     cmd: {' '.join(s['cmd'])}")
        print(f"     inputs: {', '.join(s['inputs']) or '(upstream results only)'}")
        print(f"     outputs ({len(s['outputs'])}): "
              + ", ".join(s["outputs"]))
    unmapped = [t for t in generated_tracked if t not in outputs]
    extra = [o for o in outputs if o not in generated_tracked]
    if unmapped:
        print(f"\nWARNING tracked-but-unmapped: {unmapped}")
    if extra:
        print(f"\nNOTE declared-but-untracked (will become newly tracked): {extra}")


def check(manifest, results_dir: Path = RESULTS, verify_hashes: bool = True) -> int:
    outputs = all_outputs(manifest)
    tracked = tracked_results_files()
    problems = []

    for rel in outputs:
        p = results_dir / rel
        if not p.exists():
            problems.append(f"MISSING   {rel}")
        elif p.stat().st_size == 0:
            problems.append(f"EMPTY     {rel}")

    known = set(outputs) | set(NON_GENERATED) | {ATTESTATION, BUILD_LOG}
    for t in tracked:
        if t not in known:
            problems.append(f"UNMAPPED  {t}  (tracked in git, not in manifest)")

    att_path = results_dir / ATTESTATION
    if verify_hashes:
        if not att_path.exists():
            problems.append(f"NO-ATTEST {ATTESTATION} absent — tree is not an "
                            "attested clean-room build")
        else:
            att = json.loads(att_path.read_text())
            for rel, digest in att.get("files", {}).items():
                p = results_dir / rel
                if not p.exists():
                    problems.append(f"DELETED   {rel}  (attested file absent)")
                elif sha256(p) != digest:
                    problems.append(f"STALE     {rel}  (hash != attestation)")
            actual = {
                str(p.relative_to(results_dir))
                for p in results_dir.rglob("*")
                if p.is_file() and p.name != ATTESTATION
            }
            attested = set(att.get("files", {}))
            for rel in sorted(actual - attested):
                problems.append(f"EXTRA     {rel}  (not in attested tree)")
            for rel in outputs:
                if rel not in att.get("files", {}):
                    problems.append(f"UNATTESTED {rel}  (not in attestation)")
            for rel, digest in att.get("inputs", {}).items():
                p = ROOT / rel
                if not p.exists():
                    problems.append(f"INPUT-MISSING {rel}")
                elif sha256(p) != digest:
                    problems.append(f"INPUT-STALE {rel}  (hash != attestation)")

    if problems:
        print(f"CHECK FAILED: {len(problems)} problem(s)")
        for p in problems:
            print("  " + p)
        return 1
    print(f"CHECK OK: {len(outputs)} manifest outputs present and non-empty; "
          f"{len(tracked)} tracked files all accounted for")
    return 0


def build(manifest, keep_build=False, only_stages=None):
    ordered = topo_order(manifest)
    if only_stages:
        unknown = set(only_stages) - {s["stage"] for s in ordered}
        if unknown:
            raise SystemExit(f"unknown stages: {sorted(unknown)}")
        ordered = [s for s in ordered if s["stage"] in only_stages]

    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                            capture_output=True, text=True, check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        print("WARNING: working tree is dirty; the clean-room build uses "
              "committed HEAD, not uncommitted edits:\n" + dirty)

    tmp = Path(tempfile.mkdtemp(prefix="uk-ai-rebuild-"))
    wt = tmp / "worktree"
    print(f"build root: {wt}")
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), "HEAD"],
                   cwd=ROOT, check=True)
    ok = False
    try:
        # data/ is untracked (large survey files): share it read-only by symlink
        if (ROOT / "data").exists() and not (wt / "data").exists():
            (wt / "data").symlink_to(ROOT / "data")
        build_env = os.environ.copy()
        build_env["PYTHONPATH"] = str(wt)
        probe = subprocess.run(
            [
                PY,
                "-c",
                "import pathlib, uk_ai_study; "
                "p=pathlib.Path(uk_ai_study.__file__).resolve(); "
                "w=pathlib.Path.cwd().resolve(); "
                "assert p.is_relative_to(w), (p, w); print(p)",
            ],
            cwd=wt,
            env=build_env,
            capture_output=True,
            text=True,
        )
        if probe.returncode:
            raise SystemExit(
                "clean-room import isolation failed:\n"
                + probe.stdout
                + probe.stderr
            )
        print("isolated package import:", probe.stdout.strip())
        # clean-room: build results from EMPTY
        build_results = wt / "results"
        if build_results.exists():
            shutil.rmtree(build_results)
        for sub in ["jr16", "appendix", "robustness", "incidence", "policy",
                    "caseloads", "paper_scenarios", "tax_composition", "geo"]:
            (build_results / sub).mkdir(parents=True, exist_ok=True)

        log_path = build_results / BUILD_LOG
        started = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        for i, s in enumerate(ordered, 1):
            print(f"\n=== [{i}/{len(ordered)}] {s['stage']}: {' '.join(s['cmd'])}")
            with open(log_path, "a") as log:
                log.write(f"\n=== stage {s['stage']}: {' '.join(s['cmd'])}\n")
                log.flush()
                proc = subprocess.run(s["cmd"], cwd=wt, env=build_env,
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True)
                log.write(proc.stdout)
            sys.stdout.write(proc.stdout[-2000:])
            if proc.returncode != 0:
                raise SystemExit(f"stage {s['stage']} failed "
                                 f"(rc={proc.returncode}); full log: {log_path}")
            missing = [o for o in s["outputs"]
                       if not (build_results / o).exists()
                       or (build_results / o).stat().st_size == 0]
            if missing:
                raise SystemExit(f"stage {s['stage']} succeeded but declared "
                                 f"outputs missing/empty: {missing}")
            missing_repo = [
                o for o in s.get("repo_outputs", [])
                if not (wt / o).exists() or (wt / o).stat().st_size == 0
            ]
            if missing_repo:
                raise SystemExit(
                    f"stage {s['stage']} succeeded but repo outputs "
                    f"missing/empty: {missing_repo}"
                )

        # full-manifest validation before publish (skip when --stages subset)
        if not only_stages:
            rc = check(manifest, results_dir=build_results, verify_hashes=False)
            if rc != 0:
                raise SystemExit("built tree failed manifest validation; "
                                 "not publishing")

        # carry non-generated documentation files through the publish
        for rel in NON_GENERATED:
            src = RESULTS / rel
            if src.exists():
                dst = build_results / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

        # attestation, written into the build tree so it publishes atomically
        files = {}
        for p in sorted(build_results.rglob("*")):
            if p.is_file() and p.name != ATTESTATION:
                files[str(p.relative_to(build_results))] = sha256(p)
        input_paths = sorted(
            {
                rel
                for stage in ordered
                for rel in stage.get("inputs", [])
                if (wt / rel).is_file()
            }
        )
        attestation = {
            "git_commit": commit,
            # The detached source worktree is constructed from committed HEAD
            # and import-isolation is asserted before any stage runs. Dirt in
            # the invoking checkout cannot enter the build and is recorded
            # separately for auditability.
            "git_dirty_at_build": False,
            "invoking_checkout_dirty_at_start": bool(dirty),
            "started_utc": started,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stages": [{"stage": s["stage"], "cmd": s["cmd"]} for s in ordered],
            "script_sha256": {
                str(f.relative_to(wt)): sha256(f)
                for f in sorted((wt / "analysis").glob("*.py"))
            } | {
                str(f.relative_to(wt)): sha256(f)
                for f in sorted((wt / "uk_ai_study").glob("*.py"))
            },
            "inputs": {rel: sha256(wt / rel) for rel in input_paths},
            "presentation_files": {
                rel: sha256(wt / rel)
                for stage in ordered
                for rel in stage.get("repo_outputs", [])
            },
            "seed_policy": ("headline single-draw results use seed 0; "
                            "Monte Carlo families use the scripts' internal "
                            "fixed seed sequences (paired seeds 0..19)"),
            "non_generated": NON_GENERATED,
            "files": files,
        }
        (build_results / ATTESTATION).write_text(json.dumps(attestation, indent=2))

        # atomic publish: single rsync --delete of the validated build tree
        if only_stages:
            print("\n--stages subset build: NOT publishing (partial tree). "
                  f"Outputs left in {build_results}")
        else:
            subprocess.run(
                ["rsync", "-a", "--delete", "--checksum",
                 str(build_results) + "/", str(RESULTS) + "/"],
                check=True,
            )
            for rel in {
                rel
                for stage in ordered
                for rel in stage.get("repo_outputs", [])
            }:
                src = wt / rel
                dst = ROOT / rel
                staged = dst.with_name(dst.name + ".rebuild-new")
                shutil.copy2(src, staged)
                os.replace(staged, dst)
            print(f"\nPUBLISHED {len(files)} files to {RESULTS} "
                  f"(attestation: results/{ATTESTATION})")
        ok = True
    finally:
        if keep_build or not ok:
            print(f"build tree kept at {wt}")
        else:
            subprocess.run(["git", "worktree", "remove", "--force", str(wt)],
                           cwd=ROOT, check=False)
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="print the DAG and coverage; run nothing")
    ap.add_argument("--check", action="store_true",
                    help="verify results/ against manifest and attestation")
    ap.add_argument("--no-hashes", action="store_true",
                    help="with --check: skip attestation hash comparison")
    ap.add_argument("--stages", nargs="*", default=None,
                    help="run only these stages (no publish)")
    ap.add_argument("--keep-build", action="store_true",
                    help="keep the temp worktree after a successful build")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(MANIFEST)
    elif args.check:
        sys.exit(check(MANIFEST, verify_hashes=not args.no_hashes))
    else:
        build(MANIFEST, keep_build=args.keep_build, only_stages=args.stages)


if __name__ == "__main__":
    main()
