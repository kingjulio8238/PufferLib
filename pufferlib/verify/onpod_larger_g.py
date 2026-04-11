"""Step 5: Test larger G hypothesis — does more completions per prompt improve GRPO?
Run this on a GPU (RunPod RTX 4090) AFTER onpod_profile.py.

Usage:
    python3 -m pufferlib.verify.onpod_larger_g
"""

import re
import time
import os
import json
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
OUTPUT_DIR = "/root/verify_larger_g"
NUM_TRAIN_SAMPLES = 200
MAX_COMPLETION_LENGTH = 512
NUM_TRAIN_EPOCHS = 1

# G values to test (reduce if OOM)
G_VALUES = [8, 32, 64]

# Adjust batch size per G to avoid OOM
BATCH_CONFIG = {
    8:  {"per_device_batch": 1, "grad_accum": 4},
    32: {"per_device_batch": 1, "grad_accum": 2},
    64: {"per_device_batch": 1, "grad_accum": 1},
}

SYSTEM_PROMPT = "Solve the math problem step by step. Put your final numerical answer after ####."

# ============================================================
# DATASET & REWARD
# ============================================================

def extract_gsm8k_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def load_gsm8k(n_samples: int):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if n_samples < len(ds):
        ds = ds.select(range(n_samples))

    def fmt(x):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": x["question"]},
            ],
            "answer": extract_gsm8k_answer(x["answer"]),
        }

    return ds.map(fmt)


def c_reward(completions, answer, **kwargs) -> list[float]:
    """C-accelerated verification via PufferVerify."""
    from pufferlib.verify.math import check_batch

    texts = []
    for completion in completions:
        if isinstance(completion, list):
            texts.append(completion[0]["content"])
        else:
            texts.append(str(completion))

    gt_strs = [str(a) for a in answer]
    return check_batch(texts, gt_strs)


# ============================================================
# TRAINING
# ============================================================

def run_with_g(g_value: int):
    print(f"\n{'='*60}")
    print(f"TRAINING WITH G={g_value}")
    print(f"{'='*60}")

    dataset = load_gsm8k(NUM_TRAIN_SAMPLES)
    output_dir = f"{OUTPUT_DIR}/g{g_value}"
    os.makedirs(output_dir, exist_ok=True)

    batch_cfg = BATCH_CONFIG.get(g_value, {"per_device_batch": 1, "grad_accum": 1})

    config = GRPOConfig(
        output_dir=output_dir,
        run_name=f"larger-g-{g_value}",
        learning_rate=5e-6,
        bf16=True,
        per_device_train_batch_size=batch_cfg["per_device_batch"],
        gradient_accumulation_steps=batch_cfg["grad_accum"],
        num_generations=g_value,
        max_prompt_length=256,
        max_completion_length=MAX_COMPLETION_LENGTH,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        logging_steps=5,
        save_steps=99999,
        report_to="none",
        log_on_each_node=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=c_reward,
        args=config,
        train_dataset=dataset,
    )

    wall_start = time.perf_counter()
    trainer.train()
    wall_total = time.perf_counter() - wall_start

    # Get training logs
    log_history = trainer.state.log_history
    final_reward = None
    for entry in reversed(log_history):
        if "reward" in entry:
            final_reward = entry.get("reward")
            break
        if "train_loss" in entry:
            final_reward = entry.get("train_loss")
            break

    results = {
        "g_value": g_value,
        "wall_clock_sec": round(wall_total, 2),
        "final_reward": final_reward,
        "num_samples": NUM_TRAIN_SAMPLES,
        "log_history": log_history,
    }

    print(f"\n--- G={g_value} Results ---")
    print(f"  Wall clock: {wall_total:.1f}s")
    print(f"  Final reward/loss: {final_reward}")

    with open(f"{output_dir}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    del trainer, model
    torch.cuda.empty_cache()

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PufferVerify: Larger G Hypothesis Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Testing G values: {G_VALUES}")
    print("=" * 60)

    all_results = {}
    for g in G_VALUES:
        try:
            result = run_with_g(g)
            all_results[f"g{g}"] = result
        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM at G={g} — skipping. Try reducing MAX_COMPLETION_LENGTH or NUM_TRAIN_SAMPLES.")
            all_results[f"g{g}"] = {"error": "OOM", "g_value": g}
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("LARGER G COMPARISON")
    print("=" * 60)
    print(f"{'G':<8} {'Wall Clock':>12} {'Final Metric':>14} {'Status':>10}")
    print("-" * 44)
    for g in G_VALUES:
        key = f"g{g}"
        r = all_results.get(key, {})
        if "error" in r:
            print(f"{g:<8} {'—':>12} {'—':>14} {'OOM':>10}")
        else:
            print(f"{g:<8} {r['wall_clock_sec']:>11.1f}s {str(r.get('final_reward', '?')):>14} {'OK':>10}")

    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
