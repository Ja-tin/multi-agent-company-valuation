import copy
import statistics
from typing import Any, Dict, List

from pipeline import evaluate_context, prepare_context


def _apply_macro_override(context: Dict[str, Any], override: Dict[str, float]) -> Dict[str, Any]:
    ctx = copy.deepcopy(context)
    for key, value in override.items():
        if key == "regime":
            ctx["macro"][key] = value
        else:
            ctx["macro"][key] = float(value)
    return ctx


def stability_analysis(
    ticker: str = "AAPL",
    years_cashflows: int = 5,
    debate_rounds: int = 2,
    eta: float = 0.3,
) -> Dict[str, Any]:
    """
    Run small macro perturbations and measure how much the final consensus changes.
    Lower coefficient of variation (CV) = more stable valuation.
    """
    base_context = prepare_context(ticker=ticker, years_cashflows=years_cashflows)

    base_result = evaluate_context(
        context=base_context,
        debate_rounds=debate_rounds,
        eta=eta,
        save_episode=False,
    )
    base_final = float(base_result["final_consensus"])

    macro = base_context["macro"]
    inflation = float(macro["inflation_yoy"])
    unemployment = float(macro["unemployment"])
    gdp = float(macro["gdp_yoy"])

    scenarios: List[Dict[str, float]] = [
        {"inflation_yoy": inflation + 0.005},
        {"inflation_yoy": max(0.0, inflation - 0.005)},
        {"unemployment": unemployment + 0.003},
        {"unemployment": max(0.0, unemployment - 0.003)},
        {"gdp_yoy": gdp + 0.005},
        {"gdp_yoy": gdp - 0.005},
    ]

    scenario_outputs = []
    all_values = [base_final]

    for i, override in enumerate(scenarios, start=1):
        scenario_context = _apply_macro_override(base_context, override)
        scenario_result = evaluate_context(
            context=scenario_context,
            debate_rounds=debate_rounds,
            eta=eta,
            save_episode=False,
        )
        final_value = float(scenario_result["final_consensus"])
        all_values.append(final_value)

        scenario_outputs.append(
            {
                "scenario_id": i,
                "override": override,
                "final_consensus": final_value,
            }
        )

    mean_value = statistics.mean(all_values)
    stdev_value = statistics.pstdev(all_values)
    cv = stdev_value / mean_value if mean_value != 0 else float("inf")

    return {
        "ticker": ticker,
        "base_final_consensus": base_final,
        "scenario_outputs": scenario_outputs,
        "mean": mean_value,
        "stdev": stdev_value,
        "cv": cv,
    }