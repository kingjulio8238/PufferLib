"""AReaL (Async RL for LLM Reasoning) integration stub for PufferVerify.

AReaL uses modular reward workers. This module provides a reward computation
function compatible with AReaL's reward worker interface.

Repo: https://github.com/inclusionai/areal
"""

from pufferlib.verify.math import check_batch


def compute_rewards(completions: list[str], references: list[str],
                    tolerance: float = 1e-5) -> list[float]:
    """Compute math verification rewards for AReaL reward workers.

    Args:
        completions: Model-generated completions
        references: Ground truth answers
        tolerance: Numerical comparison tolerance

    Returns:
        List of float rewards (1.0 = correct, 0.0 = incorrect)
    """
    return check_batch(completions, references, tolerance)
