import numpy as np
import pandas as pd

from analysis.monte_carlo_families import summarise


def test_summary_reports_mcse_and_assignment_quantiles():
    frame = pd.DataFrame({"metric": [1.0, 2.0, 3.0, 4.0]})
    result = summarise(frame, ["metric"])["metric"]
    assert result["mean"] == 2.5
    assert result["mcse"] == np.std(frame.metric, ddof=1) / 2
    assert result["q025"] < result["mean"] < result["q975"]
