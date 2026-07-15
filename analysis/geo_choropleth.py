"""Real geographic choropleths of the constituency shock (replaces the hex maps).

Reads the cached per-constituency results (results/geo/constituency_impacts.csv)
and 2024 Westminster boundary polygons, joins on GSS code, and draws two
choropleths in the PolicyEngine house style: the diverging income-change map
(grey losses, blue gains, scale clipped at the 95th percentile so the English
gradient is not crushed by extreme seats) and the sequential displacement map.

No microsimulation is run here -- this is presentation only, off the cached CSV.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import figstyle  # noqa: E402

GEO = ROOT / "results" / "geo"
IMPACTS = GEO / "constituency_impacts.csv"
# 2024 Westminster constituency boundaries (GSS-coded), sourced locally.
BOUNDARIES = Path(
    "/Users/janansadeqian/energy-price-shock/dashboard/public/data/uk_constituencies_2024.geojson"
)


def load():
    df = pd.read_csv(IMPACTS)
    gdf = gpd.read_file(BOUNDARIES)[["GSScode", "geometry"]]
    merged = gdf.merge(df, left_on="GSScode", right_on="code", how="inner")
    missing = len(df) - len(merged)
    if missing:
        print(f"warning: {missing} constituencies did not join to a boundary")
    # The GeoJSON is mislabelled EPSG:4326 but its coordinates are already
    # British National Grid eastings/northings; assign the true CRS (no reproject).
    return merged.set_crs(27700, allow_override=True)


def choropleth(gdf, col, title, path, diverging):
    figstyle.apply_style()
    vals = gdf[col].to_numpy()
    if diverging:
        vmax = float(np.nanpercentile(np.abs(vals), 95)) or float(np.abs(vals).max())
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        cmap = figstyle.DIVERGING
        extend = "both"
    else:
        norm = Normalize(vmin=float(np.nanmin(vals)), vmax=float(np.nanmax(vals)))
        cmap = figstyle.SEQUENTIAL
        extend = "neither"

    fig, ax = plt.subplots(figsize=(8.5, 10.5))
    gdf.plot(column=col, cmap=cmap, norm=norm, ax=ax,
             edgecolor="white", linewidth=0.15)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title + "\nCentral scenario (7% displacement, +2.6% wages), 2025, seed 0")

    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                        fraction=0.035, pad=0.01, extend=extend)
    cbar.outline.set_visible(False)
    figstyle.save(fig, path)
    print("wrote", path)


def main():
    gdf = load()
    choropleth(gdf, "income_change_pct",
               "Mean HBAI household income change (%)",
               GEO / "map_income_change.png", diverging=True)
    choropleth(gdf, "displaced_per_1000_workers",
               "Displaced workers per 1,000 workers",
               GEO / "map_displacement.png", diverging=False)


if __name__ == "__main__":
    main()
