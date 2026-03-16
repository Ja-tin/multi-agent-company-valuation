from typing import List, Tuple


def compute_dcf(cashflows: List[float], discount_rate: float) -> float:
    """
    Compute basic DCF:
      PV = sum(CF_t / (1 + r)^t)
    Assumes cashflows are annual and provided in order:
    [year1, year2, year3, ...]
    """
    if not cashflows:
        raise ValueError("cashflows cannot be empty")

    if discount_rate <= -0.99:
        raise ValueError("discount_rate is invalid")

    total_value = 0.0
    for i, cf in enumerate(cashflows):
        t = i + 1
        total_value += cf / ((1.0 + discount_rate) ** t)

    return float(total_value)


def inverse_uncertainty_weights(sigmas: List[float]) -> List[float]:
    """
    Weight_k = (1 / sigma_k) / sum_j (1 / sigma_j)
    """
    if not sigmas:
        raise ValueError("sigmas cannot be empty")

    if any(s <= 0 for s in sigmas):
        raise ValueError("all sigmas must be > 0")

    inv = [1.0 / s for s in sigmas]
    total = sum(inv)
    return [x / total for x in inv]


def weighted_consensus(values: List[float], sigmas: List[float]) -> Tuple[float, List[float]]:
    """
    Returns:
      final_value, weights
    """
    if len(values) != len(sigmas):
        raise ValueError("values and sigmas must have the same length")

    weights = inverse_uncertainty_weights(sigmas)
    final_value = sum(v * w for v, w in zip(values, weights))
    return float(final_value), weights


def debate_round(values: List[float], sigmas: List[float], eta: float) -> Tuple[List[float], float, List[float]]:
    """
    One debate round:
      v_k <- v_k + eta * (consensus - v_k)
    """
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must be between 0 and 1")

    consensus, weights = weighted_consensus(values, sigmas)
    updated = [v + eta * (consensus - v) for v in values]
    return updated, consensus, weights


def run_debate(values: List[float], sigmas: List[float], rounds: int = 2, eta: float = 0.3) -> Tuple[float, List[float], List[float]]:
    """
    Run multiple debate rounds and return:
      final_consensus, final_weights, post_debate_values
    """
    if rounds < 0:
        raise ValueError("rounds must be >= 0")

    current = values[:]
    for _ in range(rounds):
        current, _, _ = debate_round(current, sigmas, eta)

    final_consensus, final_weights = weighted_consensus(current, sigmas)
    return float(final_consensus), final_weights, current