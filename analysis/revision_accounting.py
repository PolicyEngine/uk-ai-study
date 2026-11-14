"""Post-processing for the Fiscal Studies revision (13 Nov 2026).

Every function here is pure arithmetic on results that already exist under
``results/``; nothing runs a simulation. ``emit_tex_values.py`` calls these
functions and writes their outputs as macros, and
``tests/test_revision_accounting.py`` checks them on fake data.

Draw bases (each quantity stays on one basis):

* break-even values: the survivor-wage grid is seed 0; the capital and
  duration break-evens use the 20 paired draws of the channel split and of
  the duration sensitivity;
* the factor-income account is seed 0 throughout, because seed 0 is the only
  draw with the displaced wage bill, the survivor-wage grid and the
  tax-composition run on one basis;
* cost per displaced worker is the per-draw ratio over the 20 channel draws;
* the exposure-index comparison is regression-adjusted to the 50-draw mean
  displaced count with the slope from the 50 exposure-proportional draws.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: JR16 return-to-capital shock: +0.4pp on a 1.005 per cent base
BASE_RETURN_PCT = 1.005
RETURN_SHOCK_PP = 0.4
CAPITAL_SCALE = (BASE_RETURN_PCT + RETURN_SHOCK_PP) / BASE_RETURN_PCT - 1
#: corporation tax main rate used in the tax-composition exercise
CT_RATE = 0.25


# ---------------------------------------------------------------------------
# B9: break-even values
# ---------------------------------------------------------------------------
def zero_crossing(x, y) -> float:
    """Linear interpolation of the x at which y crosses zero.

    ``x`` must be sorted; the crossing must lie inside the grid (no
    extrapolation), otherwise ValueError.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    for i in range(len(x) - 1):
        if y[i] == 0:
            return float(x[i])
        if y[i] * y[i + 1] < 0:
            return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))
    if y[-1] == 0:
        return float(x[-1])
    raise ValueError("no sign change inside the grid")


def breakeven_survivor_wage(grid: pd.DataFrame) -> float:
    """Survivor-wage effect (per cent) at which the seed-0 central cost is zero."""
    g = grid.sort_values("survivor_wage_effect_pct")
    return zero_crossing(g["survivor_wage_effect_pct"], g["exchequer_cost_bn"])


def breakeven_capital(central_cost_bn: float, capital_channel_cost_bn: float) -> dict:
    """Capital-return shock at which the central cost is zero.

    The capital channel is assumed linear in the shock: the cost without it
    is ``central - capital_channel`` and each 0.4pp adds the channel again.
    Returns the shock in percentage points of return and as a proportional
    uplift to interest and dividend income.
    """
    if capital_channel_cost_bn >= 0:
        raise ValueError("the capital channel must raise net receipts")
    no_capital = central_cost_bn - capital_channel_cost_bn
    pp = RETURN_SHOCK_PP * no_capital / -capital_channel_cost_bn
    return {"pp": pp, "uplift_pct": 100 * pp / BASE_RETURN_PCT,
            "cost_without_capital_bn": no_capital}


def breakeven_duration(full_year_cost_bn: float, half_retention_cost_bn: float) -> dict:
    """Out-of-work months at which the central cost is zero.

    Linear in the earnings-retention share r between the full-year case
    (r = 0, 12 months out of work) and the 50 per cent retention hybrid
    (r = 0.5, read as six months).
    """
    if not (full_year_cost_bn > 0 > half_retention_cost_bn):
        raise ValueError("break-even needs a cost at r=0 and a gain at r=0.5")
    r = 0.5 * full_year_cost_bn / (full_year_cost_bn - half_retention_cost_bn)
    return {"retention": r, "months_out_of_work": 12 * (1 - r)}


# ---------------------------------------------------------------------------
# B10: factor-income account (seed 0)
# ---------------------------------------------------------------------------
def survivor_gain_bn(grid: pd.DataFrame, wage_bill_lost_bn: float, gross_share: float) -> float:
    """Pounds gained by survivors from the central +2.6 per cent uplift.

    The survivor-wage grid holds the displaced fixed; the difference in net
    employee earnings between the +2.6 and 0 per cent rows, times the
    baseline employee wage bill, is the survivors' gain.
    """
    bill = wage_bill_lost_bn / gross_share
    g = grid.set_index("survivor_wage_effect_pct")["net_employment_income_change_pct"]
    return float((g.loc[2.6] - g.loc[0.0]) / 100 * bill)


def factor_account(
    *,
    wage_bill_lost_bn: float,
    survivor_gain_bn: float,
    employer_ni_net_bn: float,
    dividends_bn: float,
    interest_bn: float,
    channel_disp_bn: float,
    channel_wage_bn: float,
    channel_capital_bn: float,
    central_cost_bn: float,
    dividend_tax_on_uplift_bn: float,
    phi: float,
    payout: float,
    uk_household_share: float,
) -> dict:
    """Where the displaced wage bill goes, and the closed government balance.

    Profits rise by ``phi`` times the displaced wage bill net of survivor
    gains, pay corporation tax at 25 per cent, and a share ``payout`` of
    post-tax profit is distributed, of which ``uk_household_share`` reaches
    UK households (the rest goes to overseas owners and pension funds, whose
    returns the household simulation does not tax). Household dividends are
    taxed at the average rate the simulated capital channel implies.

    The closed balance replaces the exogenous capital channel (and its tax)
    with corporation tax plus dividend tax on the implied payout; the
    displacement and survivor-wage channels and their interaction are kept.
    All flows are seed-0 pounds a year; a negative cost is a net gain.
    """
    net_labour = wage_bill_lost_bn - survivor_gain_bn
    profits = phi * net_labour
    ct = CT_RATE * profits
    post_tax = profits - ct
    paid = payout * post_tax
    household_div = uk_household_share * paid
    div_uplift = CAPITAL_SCALE * dividends_bn
    div_rate = dividend_tax_on_uplift_bn / div_uplift
    div_tax = div_rate * household_div
    interaction = central_cost_bn - (channel_disp_bn + channel_wage_bn + channel_capital_bn)
    closed_cost = channel_disp_bn + channel_wage_bn + interaction - ct - div_tax
    return {
        "wage_bill_lost_bn": wage_bill_lost_bn,
        "survivor_gain_bn": survivor_gain_bn,
        "net_labour_income_lost_bn": net_labour,
        "employer_ni_net_bn": employer_ni_net_bn,
        "capital_uplift_assumed_bn": CAPITAL_SCALE * (dividends_bn + interest_bn),
        "dividend_uplift_assumed_bn": div_uplift,
        "interest_uplift_assumed_bn": CAPITAL_SCALE * interest_bn,
        "dividend_tax_rate": div_rate,
        "profits_bn": profits,
        "corporation_tax_bn": ct,
        "post_tax_profits_bn": post_tax,
        "paid_out_bn": paid,
        "household_dividends_bn": household_div,
        "household_dividend_tax_bn": div_tax,
        "interaction_bn": interaction,
        "closed_cost_bn": closed_cost,
        "headline_cost_bn": central_cost_bn,
    }


# ---------------------------------------------------------------------------
# B11: absorption (stabilisation) and cost per displaced worker
# ---------------------------------------------------------------------------
def absorption_rates(
    *,
    wage_bill_lost_bn: float,
    disp_gov_balance_bn: float,
    disp_narrow_cash_bn: float,
    survivor_grid: pd.DataFrame,
    gross_share: float,
) -> dict:
    """Pence of Exchequer cost per pound of gross earnings removed.

    Job loss: the displacement-only run (seed 0) on the government balance
    (includes employer NI) and on the narrow household basis (income tax,
    employee NI and benefits; the Dolls et al. stabilisation coefficient).
    Wage cuts: the marginal cost of cutting survivors' wages, from the
    survivor-wage grid rows 0 and -2.6 per cent (seed 0, government balance).
    """
    bill = wage_bill_lost_bn / gross_share
    g = survivor_grid.set_index("survivor_wage_effect_pct")
    earn = (g.loc[0.0, "net_employment_income_change_pct"]
            - g.loc[-2.6, "net_employment_income_change_pct"]) / 100 * bill
    cost = g.loc[-2.6, "exchequer_cost_bn"] - g.loc[0.0, "exchequer_cost_bn"]
    return {
        "job_loss_gov_balance": disp_gov_balance_bn / wage_bill_lost_bn,
        "job_loss_household": disp_narrow_cash_bn / wage_bill_lost_bn,
        "wage_cut_gov_balance": float(cost / earn),
        "wage_cut_earnings_bn": float(earn),
    }


def cost_per_displaced(draws: pd.DataFrame, run: str = "channel_displacement") -> dict:
    """Per-draw cost per displaced worker (pounds), mean and SD across draws."""
    d = draws[draws["run"] == run]
    ratio = d["exchequer_cost_bn"] * 1e9 / (d["displaced_weighted_m"] * 1e6)
    return {"mean_gbp": float(ratio.mean()), "sd_gbp": float(ratio.std(ddof=1)), "draws": int(len(d))}


def family_displacement_cost_per_worker(
    draws: pd.DataFrame, other_channels_bn: float
) -> pd.Series:
    """Displacement-channel cost per displaced worker by family (pounds).

    Adds back the survivor-wage and capital channels (``other_channels_bn``,
    the negative sum of their 20-draw costs) to each family draw, assuming
    the channels are additive, and divides by that draw's displaced count.
    """
    disp = draws["exchequer_cost_bn"] - other_channels_bn
    ratio = disp * 1e9 / (draws["displaced_weighted_m"] * 1e6)
    return ratio.groupby(draws["family"]).mean()


# ---------------------------------------------------------------------------
# B12: index comparison adjusted to the 7 per cent count
# ---------------------------------------------------------------------------
def count_slope(draws: pd.DataFrame, col: str, family: str = "exposure") -> tuple[float, float]:
    """OLS slope of ``col`` on displaced millions across one family's draws,
    and that family's mean displaced count."""
    d = draws[draws["family"] == family]
    x, y = d["displaced_weighted_m"].to_numpy(), d[col].to_numpy()
    slope = float(np.polyfit(x, y, 1)[0])
    return slope, float(x.mean())


def count_adjusted(value: float, displaced_m: float, slope: float, target_m: float) -> float:
    """Shift a single-draw result to the target displaced count."""
    return value - slope * (displaced_m - target_m)


# ---------------------------------------------------------------------------
# M4/item 6: Monte Carlo standard error of the mean
# ---------------------------------------------------------------------------
def mc_standard_error(sd: float, n: int) -> float:
    return sd / np.sqrt(n)
