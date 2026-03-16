import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def build_model() -> ChatOpenAI:
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-nano")

    return ChatOpenAI(
        model=model_name,
        temperature=0.0,
        max_tokens=200
    )


def parse_json_strict(text: str) -> Dict[str, Any]:
    """
    Parse JSON strictly, but try to recover if the model wraps JSON in extra text.
    """
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"Model did not return valid JSON:\n{text}")
        return json.loads(text[start:end + 1])


@dataclass
class DcfAgentDecision:
    discount_rate: float
    sigma: float
    rationale: str
    assumptions: List[str]


@dataclass
class RiskAgentDecision:
    discount_rate: float
    sigma: float
    rationale: str
    assumptions: List[str]


@dataclass
class MacroAgentDecision:
    multiplier: float
    sigma: float
    regime: str
    rationale: str
    assumptions: List[str]


def _invoke_json(system_prompt: str, user_payload: Dict[str, Any]) -> Dict[str, Any]:
    model = build_model()

    messages = [
        ("system", system_prompt),
        ("human", json.dumps(user_payload, indent=2)),
    ]

    response = model.invoke(messages)
    return parse_json_strict(response.content)


def run_dcf_agent(context: Dict[str, Any], memory_block: str) -> DcfAgentDecision:
    system_prompt = f"""
You are the DCF Agent.

Recent shared episodic memory:
{memory_block}

Your job:
- Focus on baseline intrinsic valuation logic.
- Choose a reasonable discount rate for the baseline DCF.
- The discount rate must be a decimal, like 0.09 for 9%.

Return ONLY strict JSON with keys:
discount_rate, sigma, rationale, assumptions

Rules:
- sigma must be > 0
- assumptions must be a list of 3 to 6 short strings
- use the context numbers provided
- if data quality feels weak, increase sigma
- do not include markdown
""".strip()

    data = _invoke_json(system_prompt, context)

    return DcfAgentDecision(
        discount_rate=float(data["discount_rate"]),
        sigma=max(1e-6, float(data["sigma"])),
        rationale=str(data["rationale"]).strip(),
        assumptions=list(data["assumptions"]),
    )


def run_risk_agent(context: Dict[str, Any], memory_block: str) -> RiskAgentDecision:
    system_prompt = f"""
You are the Risk Premium Agent.

Recent shared episodic memory:
{memory_block}

Your job:
- Focus on discount-rate logic.
- Use the risk_free_10y in context.
- Add an equity risk premium implicitly and output the total discount_rate.

Return ONLY strict JSON with keys:
discount_rate, sigma, rationale, assumptions

Rules:
- sigma must be > 0
- assumptions must be a list of 3 to 6 short strings
- if inflation is high or GDP is weak or unemployment is high, discount_rate should generally be more conservative
- do not include markdown
""".strip()

    data = _invoke_json(system_prompt, context)

    return RiskAgentDecision(
        discount_rate=float(data["discount_rate"]),
        sigma=max(1e-6, float(data["sigma"])),
        rationale=str(data["rationale"]).strip(),
        assumptions=list(data["assumptions"]),
    )


def run_macro_agent(context: Dict[str, Any], memory_block: str) -> MacroAgentDecision:
    system_prompt = f"""
You are the Macro Signal Agent.

Recent shared episodic memory:
{memory_block}

Your job:
- Interpret the macro regime using inflation_yoy, unemployment, and gdp_yoy.
- Output a valuation multiplier.
- Example: 0.95 means apply a 5% haircut to baseline DCF.

Return ONLY strict JSON with keys:
multiplier, sigma, regime, rationale, assumptions

Rules:
- regime must be one of: stable, moderate_stress, high_stress
- multiplier should usually be in [0.85, 1.05]
- sigma must be > 0
- assumptions must be a list of 3 to 6 short strings
- do not include markdown
""".strip()

    data = _invoke_json(system_prompt, context)

    regime = str(data["regime"]).strip()
    if regime not in {"stable", "moderate_stress", "high_stress"}:
        regime = "moderate_stress"

    multiplier = float(data["multiplier"])
    multiplier = max(0.70, min(1.10, multiplier))

    return MacroAgentDecision(
        multiplier=multiplier,
        sigma=max(1e-6, float(data["sigma"])),
        regime=regime,
        rationale=str(data["rationale"]).strip(),
        assumptions=list(data["assumptions"]),
    )