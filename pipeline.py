from typing import Any, Dict, Optional

from agents_llm import run_dcf_agent, run_macro_agent, run_risk_agent
from data_cashflows import get_free_cashflows
from data_fred import get_macro_and_risk_free
from memory_store import append_episode, format_episodes_for_prompt, load_recent_episodes
from valuation_math import compute_dcf, run_debate


def prepare_context(
    ticker: str,
    years_cashflows: int = 5,
    macro_override: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Build the full structured context passed to agents.
    """
    macro = get_macro_and_risk_free()
    cashflows = get_free_cashflows(ticker, years=years_cashflows)

    if macro_override:
        macro.update(macro_override)

    context = {
        "ticker": ticker,
        "cashflows": cashflows,
        "risk_free_10y": macro["risk_free_10y"],
        "macro": {
            "inflation_yoy": macro["inflation_yoy"],
            "unemployment": macro["unemployment"],
            "gdp_yoy": macro["gdp_yoy"],
            "regime": macro["regime"],
        },
    }
    return context


def evaluate_context(
    context: Dict[str, Any],
    debate_rounds: int = 2,
    eta: float = 0.3,
    save_episode: bool = False,
) -> Dict[str, Any]:
    """
    Run the 3-agent valuation pipeline on a prepared context.
    """
    memory_block = format_episodes_for_prompt(load_recent_episodes(limit=5))

    dcf_decision = run_dcf_agent(context, memory_block)
    risk_decision = run_risk_agent(context, memory_block)
    macro_decision = run_macro_agent(context, memory_block)

    cashflows = context["cashflows"]

    baseline_dcf = compute_dcf(cashflows, dcf_decision.discount_rate)
    risk_dcf = compute_dcf(cashflows, risk_decision.discount_rate)
    macro_dcf = baseline_dcf * macro_decision.multiplier

    values = [baseline_dcf, risk_dcf, macro_dcf]
    sigmas = [dcf_decision.sigma, risk_decision.sigma, macro_decision.sigma]

    final_consensus, weights, post_values = run_debate(
        values=values,
        sigmas=sigmas,
        rounds=debate_rounds,
        eta=eta,
    )

    result = {
        "ticker": context["ticker"],
        "inputs": context,
        "agent_outputs": {
            "dcf_agent": {
                "discount_rate": dcf_decision.discount_rate,
                "sigma": dcf_decision.sigma,
                "rationale": dcf_decision.rationale,
                "assumptions": dcf_decision.assumptions,
            },
            "risk_agent": {
                "discount_rate": risk_decision.discount_rate,
                "sigma": risk_decision.sigma,
                "rationale": risk_decision.rationale,
                "assumptions": risk_decision.assumptions,
            },
            "macro_agent": {
                "multiplier": macro_decision.multiplier,
                "regime": macro_decision.regime,
                "sigma": macro_decision.sigma,
                "rationale": macro_decision.rationale,
                "assumptions": macro_decision.assumptions,
            },
        },
        "valuations_pre_debate": {
            "baseline_dcf": baseline_dcf,
            "risk_dcf": risk_dcf,
            "macro_dcf": macro_dcf,
        },
        "debate": {
            "rounds": debate_rounds,
            "eta": eta,
            "post_values": post_values,
            "weights": weights,
        },
        "final_consensus": final_consensus,
    }

    if save_episode:
        append_episode(result)

    return result


def run_pipeline(
    ticker: str = "AAPL",
    years_cashflows: int = 5,
    debate_rounds: int = 2,
    eta: float = 0.3,
    save_episode: bool = True,
) -> Dict[str, Any]:
    """
    Main public entry point for one full valuation run.
    """
    context = prepare_context(
        ticker=ticker,
        years_cashflows=years_cashflows,
        macro_override=None,
    )
    return evaluate_context(
        context=context,
        debate_rounds=debate_rounds,
        eta=eta,
        save_episode=save_episode,
    )