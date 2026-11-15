#!/usr/bin/env python3
"""Verified restamp of committed draw checkpoints after non-behavioural edits.

Why. Every draw-level checkpoint CSV carries a ``code_fingerprint`` (see
``uk_ai_study/checkpoint.py``), and ``emit_tex_values.py`` refuses draws whose
stamp differs from the current code. The fixes for the Fable 5.2 review of
PR #18 change files that the fingerprint covers without changing what any
simulation computes: the fingerprint now also hashes the sibling modules a
stage imports (finding M8), every script takes ``PERIOD`` from
``uk_ai_study.data`` instead of redeclaring it (M7), duplicated constants come
from one place, and docstrings, comments and one local variable name were
reworded. Rerunning about 40 hours of simulation to refresh a hash would add
nothing, so this script restamps instead, and only after proving each edit is
non-behavioural.

What it proves, per stamped CSV (``--check`` proves and writes nothing;
``--apply`` also rewrites the stamps and the log):

1. Every row carries the fingerprint the OLD code (``--old-ref``, default the
   PR head before these fixes) gives for the producing script. So the draws
   were current at the old commit.
2. Every file in the old or new fingerprint set (the package, the script and,
   under the new rule, its imported sibling modules) normalises to the same
   AST at the old commit and now. Normalising strips docstrings, inlines
   module-level numeric/string constants however they are defined or
   imported (so ``PERIOD = 2026`` and ``from uk_ai_study.data import PERIOD``
   are equal when ``data.PERIOD == 2026``), applies the per-file local
   renames in ``RENAMES`` and blanks the bodies of the entry points in
   ``EXEMPT_FUNCTIONS``, which no stage imports. ``uk_ai_study/checkpoint.py``
   is exempt as a whole: it holds the stamp logic, not simulation code.
3. Every sibling module newly covered by the fingerprint was last changed in
   git no later than the commit that last wrote the CSV, so the committed
   draws were produced with the helper code as it stood at the old commit.

Only then is the old fingerprint token replaced by the new one, byte for byte
(no other character of the CSV changes), and ``results/RESTAMP_LOG.json``
records, per CSV, the old and new fingerprints and the verdict for each file.
A restamp is a disclosed substitute for a rerun under unchanged code, not a
rerun: the README and the PR body say so.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "analysis"))

from uk_ai_study.checkpoint import code_fingerprint, local_imports  # noqa: E402

LOG = ROOT / "results" / "RESTAMP_LOG.json"
DEFAULT_OLD_REF = "02956bdbf68638221a9af5e14d5989f58c56d95c"

#: stamped checkpoint -> producing script (None: wage_cut's own stamp rule)
STAMPED = {
    "results/jr16/grid.csv": "replicate_jr16.py",
    "results/appendix/grid_deciles_capital.csv": "appendix.py",
    "results/robustness/incidence_draws_five.csv": "monte_carlo_families.py",
    "results/robustness/policy_draws.csv": "monte_carlo_families.py",
    "results/robustness/central_cost_components_draws.csv": "monte_carlo_families.py",
    "results/robustness/duration_takeup_draws.csv": "sensitivity_duration_takeup.py",
    "results/robustness/duration_reform_draws.csv": "sensitivity_duration_reforms.py",
    "results/robustness/measured_wage_tier_draws.csv": "sensitivity_wage_tier.py",
    "results/robustness/dividend_coverage_draws.csv": "sensitivity_dividend_coverage.py",
    "results/tax_composition/composition_sims.csv": "tax_composition.py",
    "results/next_runs/obr_benchmark_draws.csv": "next_runs.py",
    "results/next_runs/capital_variants_draws.csv": "next_runs.py",
    "results/takeup/takeup_draws.csv": "displaced_takeup_runs.py",
    "results/wage_cut/wage_cut_draws.csv": "wage_cut_household.py",
}
WAGE_CUT = "wage_cut_household.py"

#: files whose edits cannot change a simulation, with the reason
EXEMPT_FILES = {
    "uk_ai_study/checkpoint.py": "stamp and fingerprint logic only (finding M8); "
                                 "no simulation calls it",
}
#: per-file top-level functions whose bodies are not compared, with the reason
EXEMPT_FUNCTIONS = {
    "analysis/policy_counterfactuals.py": {
        "main": "script entry point that writes results/policy/; its plotting "
                "moved to figures.py (finding M2). No stage imports main()",
    },
}
#: per-file renames of local names (old -> new), applied to the old source
RENAMES = {
    "analysis/incidence_scenarios.py": {"elite": "top_tier"},
}


# ---------------------------------------------------------------------------
# AST normalisation
# ---------------------------------------------------------------------------
def _strip_docstrings(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]


def const_value(node: ast.AST):
    """Value of a literal numeric/string expression, else raise ValueError."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, str)) \
            and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = const_value(node.operand)
        return -v if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        a, b = const_value(node.left), const_value(node.right)
        if isinstance(a, str) or isinstance(b, str):
            raise ValueError("string arithmetic")
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        return a / b
    raise ValueError("not a literal")


def module_constants(tree: ast.Module) -> dict:
    """Module-level ``NAME = literal`` bindings assigned exactly once."""
    counts: dict[str, int] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            counts[node.id] = counts.get(node.id, 0) + 1
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and counts[node.targets[0].id] == 1:
            try:
                out[node.targets[0].id] = const_value(node.value)
            except ValueError:
                pass
    return out


def _package_constants(root: Path, module: str) -> dict:
    path = root / (module.replace(".", "/") + ".py")
    if not module.startswith("uk_ai_study") or not path.is_file():
        return {}
    return module_constants(ast.parse(path.read_text()))


def normalise(src: str, root: Path, rel: str, renames: dict | None = None,
              exempt_functions=()) -> str:
    """Canonical AST dump of ``src`` (file ``rel`` of the tree at ``root``)."""
    tree = ast.parse(src)
    _strip_docstrings(tree)
    if renames:
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in renames:
                node.id = renames[node.id]
            elif isinstance(node, ast.arg) and node.arg in renames:
                node.arg = renames[node.arg]
    consts = module_constants(tree)
    # constants imported from the package resolve to their literal values
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            src_consts = _package_constants(root, node.module)
            for alias in node.names:
                if alias.name in src_consts:
                    consts[alias.asname or alias.name] = src_consts[alias.name]
    # never inline a name that is also bound locally somewhere
    bound = {n.arg for n in ast.walk(tree) if isinstance(n, ast.arg)}
    for node in tree.body:
        for sub in ast.walk(node):
            if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                for s in ast.walk(sub):
                    if isinstance(s, ast.Name) and isinstance(s.ctx, ast.Store):
                        bound.add(s.id)
    consts = {k: v for k, v in consts.items() if k not in bound}
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 \
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id in consts:
            continue
        if isinstance(node, ast.ImportFrom):
            node.names = [a for a in node.names if (a.asname or a.name) not in consts]
            if not node.names:
                continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in exempt_functions:
            node.body = [ast.Pass()]
        body.append(node)
    tree.body = body

    class Inline(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in consts:
                return ast.copy_location(ast.Constant(consts[node.id]), node)
            return node

    tree = Inline().visit(tree)
    return ast.dump(tree, include_attributes=False)


def compare(rel: str, old_root: Path, new_root: Path) -> str:
    """'identical', 'equivalent' or 'exempt: ...'; raises on a real change."""
    if rel in EXEMPT_FILES:
        return "exempt: " + EXEMPT_FILES[rel]
    old_p, new_p = old_root / rel, new_root / rel
    if not old_p.is_file() or not new_p.is_file():
        raise SystemExit(f"{rel}: present in only one of the two trees")
    old_s, new_s = old_p.read_text(), new_p.read_text()
    if old_s == new_s:
        return "identical"
    ex = EXEMPT_FUNCTIONS.get(rel, {})
    a = normalise(old_s, old_root, rel, RENAMES.get(rel), ex)
    b = normalise(new_s, new_root, rel, None, ex)
    if a != b:
        raise SystemExit(f"{rel}: behavioural change between the old ref and now; "
                         "the draws it covers need a rerun, not a restamp")
    return "equivalent" + (f" (bodies of {sorted(ex)} not compared: "
                           + "; ".join(ex.values()) + ")" if ex else "")


# ---------------------------------------------------------------------------
# Fingerprints at the old ref (old rule: package + script)
# ---------------------------------------------------------------------------
def old_fingerprint(old_root: Path, script: str) -> str:
    h = hashlib.sha256()
    files = sorted((old_root / "uk_ai_study").glob("*.py")) + [old_root / "analysis" / script]
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    fp = "c" + h.hexdigest()[:16]
    if script == WAGE_CUT:  # its own rule: package + script, then next_runs.py
        h = hashlib.sha256(fp.encode())
        h.update((old_root / "analysis" / "next_runs.py").read_bytes())
        fp = "c" + h.hexdigest()[:16]
    return fp


def new_fingerprint(script: str) -> str:
    if script == WAGE_CUT:
        import wage_cut_household as wc

        return wc.fingerprint()
    return code_fingerprint(ROOT / "analysis" / script)


def covered_files(old_root: Path, script: str) -> set[str]:
    rels = {f"uk_ai_study/{p.name}" for p in (ROOT / "uk_ai_study").glob("*.py")}
    rels |= {f"uk_ai_study/{p.name}" for p in (old_root / "uk_ai_study").glob("*.py")}
    rels.add(f"analysis/{script}")
    rels |= {f"analysis/{p.name}" for p in local_imports(ROOT / "analysis" / script)}
    if script == WAGE_CUT:
        rels.add("analysis/next_runs.py")
    return rels


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def materialise(ref: str, dest: Path) -> None:
    data = subprocess.run(["git", "archive", ref, "analysis", "uk_ai_study"], cwd=ROOT,
                          capture_output=True, check=True).stdout
    with tarfile.open(fileobj=BytesIO(data)) as tar:
        tar.extractall(dest, filter="data")


def restamp_text(text: str, old: str, new: str, rows: int) -> str:
    """Replace the fingerprint token on every data row and nowhere else."""
    lines = text.split("\n")
    header = lines[0].split(",")
    col = header.index("code_fingerprint")
    out, n = [lines[0]], 0
    for line in lines[1:]:
        if not line:
            out.append(line)
            continue
        cells = line.split(",")
        if len(cells) != len(header) or cells[col] != old:
            raise SystemExit(f"unexpected row shape or stamp: {line[:120]}")
        cells[col] = new
        out.append(",".join(cells))
        n += 1
    if n != rows:
        raise SystemExit(f"restamped {n} rows, expected {rows}")
    return "\n".join(out)


def run(old_ref: str, apply: bool) -> dict:
    import pandas as pd

    log = {"old_ref": old_ref, "new_head_at_restamp": git("rev-parse", "HEAD"),
           "restamped_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rule": __doc__.split("\n\n")[0].strip(), "files": {}}
    with tempfile.TemporaryDirectory() as tmp:
        old_root = Path(tmp)
        materialise(old_ref, old_root)
        verdicts: dict[str, str] = {}
        new_texts = {}
        for rel, script in STAMPED.items():
            d = pd.read_csv(ROOT / rel, dtype={"code_fingerprint": str})
            seen = set(d["code_fingerprint"])
            old_fp = old_fingerprint(old_root, script)
            new_fp = new_fingerprint(script)
            if seen == {old_fp}:
                state = "pending"
            elif seen == {new_fp}:
                # already restamped: the proofs below are rerun, and the log
                # must say this checkpoint was restamped from old_fp
                prev = json.loads(LOG.read_text())["files"].get(rel, {}) if LOG.exists() else {}
                if prev.get("old_fingerprint") != old_fp or prev.get("new_fingerprint") != new_fp:
                    raise SystemExit(f"{rel}: carries the current fingerprint but {LOG.name} "
                                     "does not record its restamp from the old ref")
                state = "restamped"
            else:
                raise SystemExit(f"{rel}: stamps {sorted(seen)} match neither the old-ref "
                                 f"fingerprint {old_fp} nor the current one {new_fp}")
            files = sorted(covered_files(old_root, script))
            for f in files:
                if f not in verdicts:
                    verdicts[f] = compare(f, old_root, ROOT)
            old_set = {f"uk_ai_study/{p.name}" for p in (old_root / "uk_ai_study").glob("*.py")}
            old_set |= {f"analysis/{script}"} | ({"analysis/next_runs.py"} if script == WAGE_CUT else set())
            csv_commit = git("log", "-1", "--format=%H", old_ref, "--", rel)
            helpers = []
            for f in sorted(set(files) - old_set):
                h_commit = git("log", "-1", "--format=%H", old_ref, "--", f)
                ok = subprocess.run(["git", "merge-base", "--is-ancestor", h_commit, csv_commit],
                                    cwd=ROOT).returncode == 0
                if not ok:
                    raise SystemExit(f"{rel}: newly covered helper {f} changed (at {h_commit[:9]}) "
                                     f"after the draws were last written ({csv_commit[:9]})")
                helpers.append({"file": f, "last_changed": h_commit[:9]})
            text = (ROOT / rel).read_text()
            if state == "pending" and new_fp != old_fp:
                new_texts[rel] = restamp_text(text, old_fp, new_fp, len(d))
            log["files"][rel] = {"script": f"analysis/{script}", "rows": len(d), "state": state,
                                 "old_fingerprint": old_fp, "new_fingerprint": new_fp,
                                 "csv_last_written": csv_commit[:9],
                                 "newly_covered_helpers": helpers, "covered_files": files}
        log["file_verdicts"] = verdicts
    if apply and new_texts:
        for rel, text in new_texts.items():
            (ROOT / rel).write_text(text)
        for r in log["files"].values():
            r.pop("state")
        LOG.write_text(json.dumps(log, indent=2) + "\n")
    elif apply:
        print("nothing to restamp: every checkpoint already carries the current fingerprint")
    return log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old-ref", default=DEFAULT_OLD_REF)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="prove equivalence; write nothing")
    g.add_argument("--apply", action="store_true", help="prove, then restamp and write the log")
    args = ap.parse_args()
    log = run(args.old_ref, apply=args.apply)
    for rel, r in log["files"].items():
        print(f"{rel}: {r['old_fingerprint']} -> {r['new_fingerprint']} ({r['rows']} rows"
              + (f", {r['state']})" if "state" in r else ")"))
    for f, v in sorted(log["file_verdicts"].items()):
        print(f"  {f}: {v}")
    print("APPLIED" if args.apply else "CHECK OK (nothing written)")


if __name__ == "__main__":
    main()
