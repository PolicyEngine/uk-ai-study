import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

OUT = "/private/tmp/claude-501/-Users-janansadeqian-uk-ai-study/a5115e49-a10d-45a0-82e2-4590cb42ccca/scratchpad/wip-pitch"
REPO = "/Users/janansadeqian/uk-ai-study"

INK = "#1a1a1a"
MUTED = "#6b6660"
BG = "#fffdfa"
FONT = "Georgia, serif"

FAMILIES = [
    ("exposure", "Exposure-proportional", "#2f5f8f"),
    ("junior", "Junior-concentrated", "#3fc1c9"),
    ("compression", "Expertise compression", "#1f6f6b"),
    ("uniform", "Uniform", "#6a994e"),
    ("klein_top_loaded", "Klein-anchored top-loaded stress test", "#101828"),
]
FILES = {
    "exposure": "exposure.json",
    "junior": "junior.json",
    "compression": "compression.json",
    "uniform": "uniform.json",
    "klein_top_loaded": "klein_top_loaded.json",
}

data = {k: json.load(open(f"{REPO}/results/incidence/{f}")) for k, f in FILES.items()}
draws = pd.read_csv(f"{REPO}/results/robustness/incidence_draws_five.csv")
means = draws.groupby("family")[["exchequer_cost_bn", "poverty_change_bhc_pp"]].mean()

# ---- Figure 1: two panels, shared bottom legend ----
fig1 = make_subplots(
    rows=1, cols=2, column_widths=[0.58, 0.42], horizontal_spacing=0.09,
    subplot_titles=("Who is displaced, by income decile", "What it costs vs poverty impact"),
)

for key, label, color in FAMILIES:
    dec = data[key]["decile_transition_share_pct"]
    xs = [int(k) for k in dec]
    ys = [dec[str(x)] for x in xs]
    fig1.add_trace(go.Scatter(
        x=xs, y=ys, name=label, legendgroup=label, mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=7, color=color),
        hovertemplate=label + "<br>Decile %{x}: %{y:.2f}%<extra></extra>",
    ), row=1, col=1)
    m = means.loc[key]
    fig1.add_trace(go.Scatter(
        x=[m["exchequer_cost_bn"]], y=[m["poverty_change_bhc_pp"]],
        name=label, legendgroup=label, showlegend=False, mode="markers",
        marker=dict(size=16, color=color),
        hovertemplate=label + "<br>Exchequer cost: £%{x:.1f}bn<br>Poverty change: +%{y:.2f}pp<extra></extra>",
    ), row=1, col=2)

axis_style = dict(gridcolor="#eee8e0", zerolinecolor="#d8d2c9", linecolor="#d8d2c9",
                  ticks="outside", tickcolor="#d8d2c9")
fig1.update_xaxes(title=None, dtick=1, row=1, col=1, **axis_style)  # no decile axis label
fig1.update_yaxes(title="Share transitioning to unemployment (%)", row=1, col=1, **axis_style)
fig1.update_xaxes(title="Exchequer cost (£ billion per year)", row=1, col=2, **axis_style)
fig1.update_yaxes(title="Change in BHC poverty rate (pp)", row=1, col=2, **axis_style)
fig1.update_layout(
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family=FONT, color=INK, size=13),
    legend=dict(orientation="h", yanchor="top", y=-0.16, x=0.5, xanchor="center",
                font=dict(size=12), itemwidth=30, tracegroupgap=4,
                entrywidth=0.32, entrywidthmode="fraction"),
    margin=dict(l=60, r=20, t=40, b=90), height=540,
)
fig1.update_annotations(font=dict(size=16, color=INK, family=FONT))

# ---- Figure 2: Gini change heatmap over the 66-cell grid ----
g = pd.read_csv(f"{REPO}/results/jr16/grid.csv")
piv = g.pivot(index="unemployment_pct", columns="wage_pct", values="gini_change_pp")

fig2 = px.imshow(
    piv.values,
    x=[f"{c}%" for c in piv.columns],
    y=[f"{r}%" for r in piv.index],
    color_continuous_scale=["#fdeee7", "#7a2410"],
    text_auto=".2f",
    aspect="auto",
    origin="lower",
)
fig2.update_xaxes(title="Survivor wage uplift", side="bottom")
fig2.update_yaxes(title="Displacement rate")
fig2.update_traces(
    hovertemplate="Displacement %{y}, wage uplift %{x}<br>Gini change: +%{z:.2f}pp<extra></extra>",
    textfont=dict(size=11),
)
fig2.update_layout(
    coloraxis_colorbar=dict(title="Gini change (pp)"),
    plot_bgcolor=BG, paper_bgcolor=BG,
    font=dict(family=FONT, color=INK, size=13),
    margin=dict(l=40, r=20, t=20, b=40), height=460,
)

with open(f"{OUT}/fig1.div.html", "w") as f:
    f.write(fig1.to_html(full_html=False, include_plotlyjs="cdn", div_id="fig1"))
with open(f"{OUT}/fig2.div.html", "w") as f:
    f.write(fig2.to_html(full_html=False, include_plotlyjs=False, div_id="fig2"))
print("done")
