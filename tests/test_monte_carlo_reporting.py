import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from monte_carlo_families import convergence_diagnostics, summarise


def test_summary_separates_assignment_spread_quantiles_and_mcse():
    frame = pd.DataFrame({"metric": np.arange(1.0, 51.0)})
    result = summarise(frame, ["metric"])["metric"]
    assert {"mean", "sd", "mcse", "q025", "median", "q975", "min", "max"} <= set(result)
    assert result["mcse"] == result["sd"] / np.sqrt(50)


def test_convergence_diagnostic_compares_20_with_50_draws():
    frame = pd.DataFrame(
        {"seed": np.arange(50), "metric": np.linspace(0.0, 0.1, 50)}
    )
    result = convergence_diagnostics(frame, ["metric"])["metric"]
    assert result["checkpoint_draws"] == 20
    assert result["full_draws"] == 50
    assert isinstance(result["passes_two_mcse"], bool)
