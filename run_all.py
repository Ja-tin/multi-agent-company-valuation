import json

from pipeline import run_pipeline
from stability import stability_analysis


def main() -> None:
    ticker = "A"
    years_cashflows = 5
    debate_rounds = 2
    eta = 0.3

    print("=" * 80)
    print("RUNNING STABLE VALUATION PIPELINE")
    print("=" * 80)

    result = run_pipeline(
        ticker=ticker,
        years_cashflows=years_cashflows,
        debate_rounds=debate_rounds,
        eta=eta,
        save_episode=True,
    )

    print("\n--- Inputs ---")
    print(json.dumps(result["inputs"], indent=2, default=str))

    print("\n--- Agent Outputs ---")
    print(json.dumps(result["agent_outputs"], indent=2, default=str))

    print("\n--- Pre-Debate Valuations ---")
    print(json.dumps(result["valuations_pre_debate"], indent=2, default=str))

    print("\n--- Debate ---")
    print(json.dumps(result["debate"], indent=2, default=str))

    print("\n--- Final Consensus ---")
    print(result["final_consensus"])

    print("\n" + "=" * 80)
    print("RUNNING STABILITY ANALYSIS")
    print("=" * 80)

    stability = stability_analysis(
        ticker=ticker,
        years_cashflows=years_cashflows,
        debate_rounds=debate_rounds,
        eta=eta,
    )

    print(json.dumps(stability, indent=2, default=str))

    print("\nDone.")
    print("A new memory episode should now exist in memory.jsonl")


if __name__ == "__main__":
    main()