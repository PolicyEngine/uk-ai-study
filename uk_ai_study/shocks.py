"""ESRI JR16 shock mechanics (eqs 3.4 / 3.5) on a person table.

Employment shock (eq 3.4): the aggregate number of displaced workers is
``displacement_rate x employed``; it is allocated across SOC major groups in
proportion to ``employment x mean C-AIOE`` of the group, then realised by
random draws with UNIFORM ordering keys within each group (the survey weight
enters only through quota consumption, so a represented person's inclusion
probability does not depend on their record's grossing weight — #1,
finding 6).

Wage shock (eq 3.5): surviving workers receive percentage uplifts
proportional to complementarity (theta), normalised by the
EMPLOYMENT-weighted mean theta over baseline workers — JR16-literal, per the
estimand decision on uk-ai-study#1 (finding 5).

Capital shock: interest and dividend income scaled by the ratio of the
shocked to the baseline return (JR16: 1.005% -> 1.405%, i.e. +0.4pp on the
return, a factor of ~1.398).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BASELINE_CAPITAL_RETURN = 0.01005
CAPITAL_RETURN_INCREASE = 0.004


@dataclass(frozen=True)
class ShockScenario:
    name: str
    displacement_rate: float
    wage_uplift: float
    capital_return_increase: float = CAPITAL_RETURN_INCREASE
    youth_displacement_multiplier: float = 1.0  # >1 tilts draws toward ages 16-24


#: Scenario presets (overridable). The capital shock (+0.4pp on the return)
#: is ON in every preset, as in all JR16 scenarios.
#: central — 7% displacement / +2.6% wages: JR16's central calibration, which
#:   converts Briggs & Kodnani (2023) task-exposure and productivity figures
#:   into displacement and wage rates (JR16 sec 3.2).
#: low — 1% displacement, no wage uplift. JR16 sec 3.2 attributes ~1% to
#:   Acemoglu (2025), but his 0.9-1.1% is a ten-year GDP figure, not an
#:   employment effect (uk-ai-study#1, finding 11) — read this as a
#:   sensitivity case, not an evidence-anchored lower bound.
#: high — Brynjolfsson, Chandar & Chen: 13% per early drafts; the Nov 2025
#:   version reports 16%. Cohort-specific relative decline treated as an
#:   economy-wide absolute rate — an upper bound in both respects.
#: central_youth_tilted — central with Klein Teeselink (2025) junior/total
#:   employment-effect ratio 5.8/4.5 as the youth multiplier.
PRESETS = {
    "central": ShockScenario("central", 0.07, 0.026),
    "low": ShockScenario("low", 0.01, 0.0),
    "high": ShockScenario("high", 0.13, 0.026),
    "central_youth_tilted": ShockScenario(
        "central_youth_tilted", 0.07, 0.026, youth_displacement_multiplier=5.8 / 4.5
    ),
}


def draw_displaced(
    persons: pd.DataFrame,
    scenario: ShockScenario,
    seed: int = 0,
) -> np.ndarray:
    """Boolean displaced mask per eq 3.4 (employees only)."""
    rng = np.random.default_rng(seed)
    employed = persons["employment_income"].to_numpy() > 0
    exposure = persons["exposure"].to_numpy()
    # JR16 normalises C-AIOE so the least-exposed sector scores 0 (and thus
    # receives no eq 3.4 job losses); the raw standardised score is negative
    # for low-exposure groups, which would corrupt the quota weights.
    exposure = exposure - exposure[employed].min()
    weight = persons["weight"].to_numpy()
    # Employees without an observed SOC code form their own pseudo-group
    # (carrying their mean-imputed exposure), so the displacement universe is
    # ALL employees and matches the wage-uplift universe (#1, finding 7).
    group = np.where(
        np.isfinite(persons["soc_major_group"].to_numpy()),
        persons["soc_major_group"].to_numpy(),
        -1.0,
    )
    total_quota = scenario.displacement_rate * float(weight[employed].sum())
    groups = np.unique(group[employed])
    # group quotas proportional to employment x mean exposure
    emp_w = {g: float(weight[employed & (group == g)].sum()) for g in groups}
    exp_g = {
        g: float(np.average(exposure[employed & (group == g)], weights=weight[employed & (group == g)]))
        for g in groups
    }
    raw = {g: emp_w[g] * exp_g[g] for g in groups}
    if sum(raw.values()) <= 0:
        # degenerate case (uniform exposure): allocate by employment alone
        raw = {g: emp_w[g] for g in groups}
        exposure = np.ones_like(exposure)
    scale = total_quota / sum(raw.values())
    displaced = np.zeros(len(persons), dtype=bool)

    age = persons["age"].to_numpy()
    for g in groups:
        members = np.flatnonzero(employed & (group == g))
        quota = raw[g] * scale
        if quota <= 0:
            continue
        # uniform ordering keys within group: exposure is constant within a
        # 1-digit group, and weighting the ordering as well as the quota
        # consumption would double-count the survey weight, making a
        # represented person's risk depend on their record's grossing
        # weight (#1, finding 6)
        p = np.ones(len(members))
        if scenario.youth_displacement_multiplier != 1.0:
            p = p * np.where(age[members] < 25, scenario.youth_displacement_multiplier, 1.0)
        p = p / p.sum()
        # draw members (weighted, without replacement) until the weighted
        # quota fills; the quota-crossing person is included with probability
        # equal to the remaining quota fraction, so the expected displaced
        # weight equals the quota exactly
        chosen = rng.choice(members, size=len(members), replace=False, p=p)
        cum = np.cumsum(weight[chosen])
        displaced[chosen[cum <= quota]] = True
        crossing = np.searchsorted(cum, quota)
        if crossing < len(chosen) and cum[crossing] > quota:
            shortfall = quota - (cum[crossing - 1] if crossing else 0.0)
            if rng.random() < shortfall / weight[chosen[crossing]]:
                displaced[chosen[crossing]] = True
    return displaced


def apply_shocks(
    persons: pd.DataFrame,
    scenario: ShockScenario,
    seed: int = 0,
) -> pd.DataFrame:
    """Shocked copy of the person table: employment, wage and capital shocks.

    Expects columns: employment_income, savings_interest_income,
    dividend_income, exposure, complementarity, soc_major_group, age, weight.
    """
    shocked = persons.copy()
    displaced = draw_displaced(persons, scenario, seed=seed)
    shocked["displaced"] = displaced

    employment = shocked["employment_income"].to_numpy(dtype=float)
    survivors = (employment > 0) & ~displaced

    # eq 3.5: person-level % wage change = wage_uplift * theta_i / theta_bar,
    # with theta_bar the EMPLOYMENT-weighted mean theta over baseline workers
    # (JR16-literal; deterministic across draws) — estimand decision on
    # uk-ai-study#1, finding 5
    theta = shocked["complementarity"].to_numpy(dtype=float)
    weight = shocked["weight"].to_numpy(dtype=float)
    baseline_workers = employment > 0
    theta_bar = float(
        (theta * weight)[baseline_workers].sum() / weight[baseline_workers].sum()
    )
    uplift = np.zeros_like(employment)
    if theta_bar > 0:
        uplift[survivors] = scenario.wage_uplift * (theta[survivors] / theta_bar) * employment[survivors]
    employment_shocked = np.where(displaced, 0.0, employment + uplift)
    shocked["employment_income"] = employment_shocked

    capital_factor = (BASELINE_CAPITAL_RETURN + scenario.capital_return_increase) / BASELINE_CAPITAL_RETURN
    for column in ("savings_interest_income", "dividend_income"):
        shocked[column] = shocked[column].to_numpy(dtype=float) * capital_factor
    return shocked


@dataclass(frozen=True)
class WageMarginScenario:
    """Wage-margin scenario: same aggregate earnings loss as displacement,
    delivered as occupation-level wage CUTS with no job loss.

    Motivated by equilibrium task models (Acemoglu & Restrepo 2022): the AI
    shock arrives mostly as RELATIVE WAGE DECLINES for workers who stay
    employed, not as full unemployment.

    aggregate_earnings_loss_share: the weighted fall in aggregate employee
    earnings as a share of baseline aggregate employee earnings. Default 0.07
    matches the central displacement preset: eq 3.4 removes a weighted head-
    count of ``displacement_rate x employees``, drawn with uniform ordering
    keys within groups and quotas ∝ employment x exposure, so the expected
    earnings removed is displacement_rate x aggregate earnings up to the
    (second-order) covariance between within-draw selection and earnings —
    across seeds the central draw removes ~7% of the aggregate employee wage
    bill. We therefore calibrate the cuts to exactly 7% of the baseline
    employee wage bill of the same universe (all employees with positive
    employment_income).

    gradient: "caioe" (cut ∝ max(0, C-AIOE normalised so the least-exposed
    employed person scores 0 — the same min-shift normalisation eq 3.4 uses
    for quota weights) or "pss" (cut ∝ a per-major-group weight column, e.g.
    the PSS column of uk_soc2020_major_group_genai_expertise.csv).

    wage_uplift: the eq 3.5 survivor uplift, applied ON TOP of the cut
    (net % change = uplift_i - cut_i, varying by occupation). Set to 0.0 to
    run the cut without the uplift.
    """

    name: str
    aggregate_earnings_loss_share: float = 0.07
    gradient: str = "caioe"
    wage_uplift: float = 0.026
    capital_return_increase: float = CAPITAL_RETURN_INCREASE


#: Wage-margin presets: capital shock ON, as in all presets.
WAGE_MARGIN_PRESETS = {
    "wage_margin_central": WageMarginScenario("wage_margin_central", gradient="caioe"),
    "wage_margin_pss": WageMarginScenario("wage_margin_pss", gradient="pss"),
}

PSS_CSV_NAME = "uk_soc2020_major_group_genai_expertise.csv"


def load_pss_weights(column: str = "pss") -> "pd.Series":
    """Per-major-group PSS weights from the packaged genai-expertise csv.

    Returns a Series indexed by major group (1-9). Raises FileNotFoundError
    with a clear message if the csv has not been built yet.
    """
    from importlib import resources

    path = resources.files("uk_ai_study") / "data" / PSS_CSV_NAME
    try:
        table = pd.read_csv(str(path))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"uk_ai_study/data/{PSS_CSV_NAME} not found: the PSS gradient "
            "requires the genai-expertise crosswalk (built separately). "
            "Either add the file or pass gradient_weights explicitly."
        )
    matches = [c for c in table.columns if c.lower() == column.lower()]
    if not matches:
        raise ValueError(
            f"{PSS_CSV_NAME} has no '{column}' column; found {list(table.columns)}."
        )
    return table.set_index("soc2020_major_group")[matches[0]]


def _gradient_values(
    persons: pd.DataFrame,
    scenario: WageMarginScenario,
    gradient_weights=None,
) -> np.ndarray:
    """Non-negative per-person gradient g_i (relative cut sizes, unscaled)."""
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    if gradient_weights is None and scenario.gradient == "caioe":
        # min-shift normalisation, exactly as eq 3.4's quota weights: only
        # occupations more exposed than the least-exposed employed person
        # lose; the least-exposed group takes a zero cut.
        exposure = persons["exposure"].to_numpy(dtype=float)
        return np.maximum(0.0, exposure - exposure[employed].min())
    if gradient_weights is None:
        if scenario.gradient != "pss":
            raise ValueError(f"unknown gradient: {scenario.gradient!r}")
        gradient_weights = load_pss_weights()
    # per-major-group weights: accept any dict/Series keyed by major group
    # (1-9 or FRS 1000-9000); employees without a SOC code get the
    # employment-weighted mean weight (mirroring mean-imputed exposure)
    weights = pd.Series(gradient_weights, dtype=float)
    if (weights < 0).any():
        raise ValueError("gradient weights must be non-negative")
    codes = pd.to_numeric(persons["soc_major_group"], errors="coerce")
    codes = codes.where(codes < 10, codes / 1000)
    g = codes.map(weights).to_numpy(dtype=float)
    w = persons["weight"].to_numpy(dtype=float)
    matched = employed & np.isfinite(g)
    mean_g = float(np.average(g[matched], weights=w[matched])) if matched.any() else 0.0
    return np.where(np.isfinite(g), g, mean_g)


def apply_wage_margin_shock(
    persons: pd.DataFrame,
    scenario: WageMarginScenario,
    gradient_weights=None,
) -> pd.DataFrame:
    """Shocked copy of the person table under the wage-margin family.

    Every employee keeps their job (``displaced`` is all False; hours,
    pension contributions and employment_status are untouched downstream —
    ``build_shocked_simulation``'s displacement transition no-ops). Each
    employee's employment_income is scaled by ``1 + uplift_i - cut_i`` where:

    - cut_i = k * g_i, with g_i the gradient (see WageMarginScenario) and k
      the single calibration constant chosen so the weighted aggregate
      earnings removed equals ``aggregate_earnings_loss_share`` x baseline
      aggregate employee earnings (earnings-equivalence with the central
      displacement draw);
    - uplift_i = wage_uplift * theta_i / theta_bar — eq 3.5 with the same
      employment-weighted theta_bar as apply_shocks (with no displacement,
      the survivor universe is all baseline workers).

    Only employment_income changes, mirroring the existing eq 3.5 uplift,
    which also leaves hours and pension contributions untouched for workers
    who keep their jobs. The capital shock applies as in every scenario.
    """
    shocked = persons.copy()
    earnings = shocked["employment_income"].to_numpy(dtype=float)
    weight = shocked["weight"].to_numpy(dtype=float)
    employed = earnings > 0
    shocked["displaced"] = np.zeros(len(shocked), dtype=bool)

    g = _gradient_values(persons, scenario, gradient_weights)
    gradient_bill = float((g * earnings * weight)[employed].sum())
    total_bill = float((earnings * weight)[employed].sum())
    if scenario.aggregate_earnings_loss_share > 0 and gradient_bill <= 0:
        raise ValueError(
            "wage-margin gradient is zero on the entire employed wage bill; "
            "cannot calibrate the aggregate earnings loss."
        )
    k = (
        scenario.aggregate_earnings_loss_share * total_bill / gradient_bill
        if gradient_bill > 0
        else 0.0
    )
    cut = k * g
    if (cut[employed] > 1.0).any():
        raise ValueError(
            "calibrated wage-margin cut exceeds 100% of earnings for some "
            "workers; the gradient is too concentrated for this loss share."
        )

    theta = shocked["complementarity"].to_numpy(dtype=float)
    theta_bar = float(np.average(theta[employed], weights=weight[employed]))
    uplift = np.zeros_like(earnings)
    if scenario.wage_uplift and theta_bar > 0:
        uplift[employed] = scenario.wage_uplift * theta[employed] / theta_bar

    factor = np.where(employed, 1.0 + uplift - cut, 1.0)
    shocked["employment_income"] = np.maximum(0.0, earnings * factor)

    capital_factor = (
        BASELINE_CAPITAL_RETURN + scenario.capital_return_increase
    ) / BASELINE_CAPITAL_RETURN
    for column in ("savings_interest_income", "dividend_income"):
        shocked[column] = shocked[column].to_numpy(dtype=float) * capital_factor
    return shocked


# --- ripple family (reduced-form Acemoglu & Restrepo 2022 propagation) ------


@dataclass(frozen=True)
class RippleScenario(ShockScenario):
    """Displacement scenario plus a reduced-form 'ripple' on survivors.

    After the central displacement draw, displaced workers seek work
    elsewhere and bid down the wages of NON-displaced workers in destination
    occupations (Acemoglu & Restrepo 2022 propagation, reduced form):

    1. Each displaced worker's labour supply is routed across destination
       SOC2020 major groups by a 9x9 row-stochastic matrix R (origin ->
       destination, zero diagonal). Default R is EMPLOYMENT-PROPORTIONAL
       routing (see analysis/build_ripple_routing.py: no pairwise
       retraining-cost matrix exists in the Hosseini & Lichtinger release,
       so R[o, l] ∝ destination employment share, ASHE 2025 Table 14,
       excluding the origin group).
    2. For each destination group l the proportional labour-supply inflow is
       s_l = (weighted displaced routed to l) / (weighted baseline employment
       of l). Non-displaced workers in l take a wage cut of
       ``labour_demand_elasticity`` (eta) x s_l, composed additively with the
       eq 3.5 uplift exactly as WageMarginScenario composes cut and uplift:
       net income = baseline x (1 + uplift_rate - cut_rate). With eta = 0 the
       scenario reproduces apply_shocks exactly.
    3. Displaced workers themselves are unchanged (fully displaced).

    Displaced workers without an observed SOC code have no origin row: their
    weight is routed by unconditional employment shares (the R rows are
    near-identical, so this is innocuous). Non-displaced employees without a
    SOC code belong to no destination group and take no ripple cut.
    """

    labour_demand_elasticity: float = 0.3  # eta: dlog(wage)/dlog(supply inflow)


#: Ripple presets = central preset + ripple, eta in {0.3 central, 0.15, 0.5}.
RIPPLE_PRESETS = {
    "central_ripple": RippleScenario("central_ripple", 0.07, 0.026),
    "central_ripple_low": RippleScenario(
        "central_ripple_low", 0.07, 0.026, labour_demand_elasticity=0.15
    ),
    "central_ripple_high": RippleScenario(
        "central_ripple_high", 0.07, 0.026, labour_demand_elasticity=0.5
    ),
}

RIPPLE_ROUTING_CSV_NAME = "uk_soc2020_major_group_ripple_routing.csv"


def load_ripple_routing() -> pd.DataFrame:
    """9x9 row-stochastic routing matrix R, indexed 1-9, columns 1-9."""
    from importlib import resources

    path = resources.files("uk_ai_study") / "data" / RIPPLE_ROUTING_CSV_NAME
    table = pd.read_csv(str(path)).set_index("origin_major_group")
    table.columns = [int(c.replace("dest_", "")) for c in table.columns]
    matrix = table.reindex(index=range(1, 10), columns=range(1, 10))
    if matrix.isna().any().any():
        raise ValueError(f"{RIPPLE_ROUTING_CSV_NAME} is not a full 9x9 matrix")
    if not np.allclose(matrix.to_numpy().sum(axis=1), 1.0, atol=1e-9):
        raise ValueError(f"{RIPPLE_ROUTING_CSV_NAME} rows must sum to 1")
    return matrix


def compute_inflow_shares(
    persons: pd.DataFrame,
    displaced: np.ndarray,
    routing: pd.DataFrame,
) -> pd.Series:
    """Per-destination-group proportional labour-supply inflow s_l.

    s_l = (weighted displaced routed to l) / (weighted baseline employment
    of l). Baseline employment of l counts all employees observed in group l
    (displaced and not) — the pre-shock workforce.
    """
    weight = persons["weight"].to_numpy(dtype=float)
    employed = persons["employment_income"].to_numpy(dtype=float) > 0
    codes = pd.to_numeric(persons["soc_major_group"], errors="coerce")
    codes = codes.where(codes < 10, codes / 1000)  # accept 1-9 or FRS 1000-9000
    group = codes.to_numpy(dtype=float)

    dest = routing.columns
    inflow = pd.Series(0.0, index=dest)
    for o in routing.index:
        w_o = float(weight[displaced & (group == o)].sum())
        if w_o:
            inflow = inflow + w_o * routing.loc[o]
    # displaced without an observed SOC code: no origin row — route by
    # unconditional destination employment shares (mean of R rows,
    # renormalised, which collapses to employment shares for the
    # employment-proportional R up to the excluded-origin correction)
    w_nan = float(weight[displaced & ~np.isfinite(group)].sum())
    if w_nan:
        shares = routing.to_numpy().mean(axis=0)
        inflow = inflow + w_nan * shares / shares.sum()

    base_emp = pd.Series(
        {l: float(weight[employed & (group == l)].sum()) for l in dest}
    )
    return (inflow / base_emp.replace(0.0, np.nan)).fillna(0.0)


def apply_ripple_shocks(
    persons: pd.DataFrame,
    scenario: RippleScenario,
    seed: int = 0,
    routing: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """apply_shocks plus the ripple wage cut on non-displaced workers.

    The cut for a non-displaced employee in destination group l is
    eta x s_l of BASELINE earnings, subtracted from the apply_shocks result
    — i.e. net income = baseline x (1 + uplift_rate - eta x s_l), the same
    additive rate composition as apply_wage_margin_shock. eta = 0 reproduces
    apply_shocks bit-for-bit. Displaced workers are unchanged (still zero).
    """
    shocked = apply_shocks(persons, scenario, seed=seed)
    eta = scenario.labour_demand_elasticity
    if eta == 0.0:
        return shocked
    if routing is None:
        routing = load_ripple_routing()

    displaced = shocked["displaced"].to_numpy()
    s = compute_inflow_shares(persons, displaced, routing)

    earnings = persons["employment_income"].to_numpy(dtype=float)
    codes = pd.to_numeric(persons["soc_major_group"], errors="coerce")
    codes = codes.where(codes < 10, codes / 1000)
    cut_rate = codes.map(eta * s).fillna(0.0).to_numpy(dtype=float)
    survivors = (earnings > 0) & ~displaced
    new = shocked["employment_income"].to_numpy(dtype=float).copy()
    new[survivors] = np.maximum(
        0.0, new[survivors] - cut_rate[survivors] * earnings[survivors]
    )
    shocked["employment_income"] = new
    return shocked


#: The transition contract (#1, finding 4 / decision 2): "displaced" means
#: fully out of work. Besides employment_income = 0, these person-level
#: inputs are zeroed so displaced workers do not remain in_work (hours > 0
#: keeps UC childcare, tax-free childcare and extended childcare paying),
#: do not keep deducting pension contributions from zero earnings, and do
#: not draw statutory pay.
TRANSITION_ZEROED_VARIABLES = (
    "hours_worked",
    "employee_pension_contributions",
    "pension_contributions_via_salary_sacrifice",
    "statutory_maternity_pay",
    "statutory_paternity_pay",
    "statutory_sick_pay",
)

SHOCKED_INCOME_VARIABLES = (
    "employment_income",
    "savings_interest_income",
    "dividend_income",
)


def build_shocked_simulation(dataset, baseline_sim, shocked_table, period):
    """One shared constructor for the shocked simulation (every pipeline).

    Sets the shocked income inputs from ``shocked_table`` and applies the
    full displacement transition to displaced persons.
    """
    from policyengine_uk import Microsimulation

    sim = Microsimulation(dataset=dataset)
    for column in SHOCKED_INCOME_VARIABLES:
        sim.set_input(column, period, shocked_table[column].to_numpy(dtype=float))
    displaced = shocked_table["displaced"].to_numpy()
    for var in TRANSITION_ZEROED_VARIABLES:
        values = baseline_sim.calculate(var, period=period, map_to="person").values.astype(float)
        values[displaced] = 0.0
        sim.set_input(var, period, values)
    status = baseline_sim.calculate("employment_status", period=period, map_to="person").values.astype(object)
    status[displaced] = "UNEMPLOYED"
    # A rejected set_input here would silently leave displaced workers
    # EMPLOYED (with zero hours), changing benefit entitlements in every
    # result — fail hard rather than warn.
    sim.set_input("employment_status", period, status)
    applied = sim.calculate("employment_status", period=period, map_to="person").values
    if not (applied[displaced].astype(str) == "UNEMPLOYED").all():
        raise RuntimeError(
            "employment_status transition not applied: displaced persons are "
            "not all UNEMPLOYED in the shocked simulation."
        )
    return sim
