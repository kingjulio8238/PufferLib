"""OpenRLHF reward function integration for PufferVerify.

Usage with OpenRLHF:
    python -m openrlhf.cli.train_ppo \
        --reward_func_path pufferlib/verify/openrlhf.py \
        ...

The reward_func below matches the interface OpenRLHF expects:
    reward_func(queries, prompts, labels) -> {"rewards": [...], "scores": [...]}

Where:
    queries: list[str]  — full model outputs (prompt + completion)
    prompts: list[str]  — original prompts
    labels: list[str]   — ground truth answers
"""

from pufferlib.verify.math import check_batch


def reward_func(queries: list[str], prompts: list[str],
                labels: list[str]) -> dict:
    """Compute math verification rewards for OpenRLHF GRPO/PPO training.

    Extracts numerical answers from model outputs and compares to ground truth.
    Handles GSM8K (#### format), LaTeX (\\boxed{}), and plain text numbers.
    """
    # Extract completion only (strip prompt from query)
    completions = []
    for query, prompt in zip(queries, prompts):
        if query.startswith(prompt):
            completions.append(query[len(prompt):])
        else:
            completions.append(query)

    rewards = check_batch(completions, labels)
    return {
        "rewards": rewards,
        "scores": rewards,
    }
