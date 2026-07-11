"""UK replication of ESRI JR16: AI exposure -> shock scenarios -> PolicyEngine UK.

Doorley, K., O'Connor, S., O'Shea, R. & Tuda, D. (2026), *Artificial
intelligence and income inequality in Ireland*, ESRI/Dept. of Finance
Jointly-published Report No. 16 — replicated for the UK with PolicyEngine UK
in place of SWITCH, FRS in place of SILC.
"""

from uk_ai_study.exposure import (
    attach_soc_major_group,
    exposure_for_major_group,
    load_major_group_exposure,
)
from uk_ai_study.shocks import PRESETS, ShockScenario, apply_shocks
from uk_ai_study.runner import run_scenario, ScenarioResult

__all__ = [
    "PRESETS",
    "ScenarioResult",
    "ShockScenario",
    "apply_shocks",
    "attach_soc_major_group",
    "exposure_for_major_group",
    "load_major_group_exposure",
    "run_scenario",
]
