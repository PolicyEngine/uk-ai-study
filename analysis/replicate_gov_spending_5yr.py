"""Replicate: effect of +£5bn/year government consumption on UK GDP over 5 years.

Uses the OBR macroeconomic model emulator (`obr_macro`) directly — the same
engine the `macromod` MCP `score_reform` tool wraps. We call `run_reform`
ourselves so we can set the reporting window to a full five years.

Key detail the MCP tool cannot express
--------------------------------------
`obr_macro.run_reform` has TWO different horizons:
  * `periods`  -> how many quarters the shock is *applied*
  * `end`      -> the last quarter that is *reported* (default "2027Q4")
The MCP `score_reform` tool only exposes `periods`, so it is structurally
capped at 12 reported quarters (2025Q1-2027Q4) no matter what `periods` you
pass. To get a genuine 5-year path you must set `end` yourself, which is only
possible by calling `run_reform` directly, as below.

£5bn/year of real government consumption = £1,250m per quarter (CGG is in
£m/quarter). Five years = 20 quarters, reported through 2029Q4.

Run:
    python analysis/replicate_gov_spending_5yr.py
"""

from obr_macro import run_reform

SHOCK_PER_QUARTER_M = 1250.0   # £1.25bn/quarter = £5bn/year
YEARS = 5
QUARTERS = YEARS * 4           # 20
START = "2025Q1"
END = "2029Q4"                 # 5-year reporting window (default is 2027Q4)


def main() -> None:
    df = run_reform(
        name="+£5bn/yr government consumption (CGG), 5 years",
        var="CGG",
        shock=SHOCK_PER_QUARTER_M,
        start=START,
        end=END,
        periods=QUARTERS,          # apply the shock for all 20 quarters
        investment_closure=False,  # not a corporation-tax reform
    )

    # Per-quarter path
    cols = ["period", "delta_gdp_bn", "pct_gdp", "delta_cons_m", "delta_if_m"]
    print(df[cols].to_string(index=False,
                             float_format=lambda x: f"{x:8.4f}"))

    # Headline summaries
    cumulative_bn = df["delta_gdp_bn"].sum()
    peak = df.loc[df["pct_gdp"].abs().idxmax()]
    # Multiplier = extra GDP per £ of extra spending, on the steady-state quarter
    multiplier = df["delta_gdp_bn"].iloc[-1] / (SHOCK_PER_QUARTER_M / 1000)

    print("\n--- Headline (5-year window) ---")
    print(f"Quarters reported            : {len(df)}")
    print(f"Cumulative GDP gain          : £{cumulative_bn:,.1f}bn")
    print(f"Peak effect                  : {peak['pct_gdp']:.3f}% of GDP "
          f"({peak['period']})")
    print(f"Steady-state impact multiplier: {multiplier:.2f}")


if __name__ == "__main__":
    main()
