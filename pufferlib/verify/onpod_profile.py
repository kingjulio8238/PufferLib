"""Step 4: Profile GRPO training with Python vs C verifier.
Run this on a GPU (RunPod RTX 4090).

Usage:
    python3 -m pufferlib.verify.onpod_profile
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
OUTPUT_DIR = "/root/verify_profile"
NUM_TRAIN_SAMPLES = 200
NUM_GENERATIONS = 8     # G in GRPO
MAX_COMPLETION_LENGTH = 512
NUM_TRAIN_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4

SYSTEM_PROMPT = "Solve the math problem step by step. Put your final numerical answer after ####."

# ============================================================
# DATASET
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


# ============================================================
# REWARD FUNCTIONS
# ============================================================

# Timing accumulators
_py_time = 0.0
_py_calls = 0
_c_time = 0.0
_c_calls = 0


def python_reward(completions, answer, **kwargs) -> list[float]:
    """Standard Python regex verification."""
    global _py_time, _py_calls
    t0 = time.perf_counter()

    rewards = []
    for completion, gt in zip(completions, answer):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        numbers = re.findall(r"[-+]?\d[\d,]*\.?\d*", text)
        if not numbers:
            rewards.append(0.0)
            continue
        try:
            pred = float(numbers[-1].replace(",", ""))
            expected = float(str(gt).replace(",", ""))
            rewards.append(1.0 if abs(pred - expected) < 1e-5 else 0.0)
        except (ValueError, TypeError):
            rewards.append(0.0)

    elapsed = time.perf_counter() - t0
    _py_time += elapsed
    _py_calls += 1
    return rewards


def c_reward(completions, answer, **kwargs) -> list[float]:
    """C-accelerated verification via PufferVerify."""
    global _c_time, _c_calls
    t0 = time.perf_counter()

    from pufferlib.verify.math import check_batch

    texts = []
    for completion in completions:
        if isinstance(completion, list):
            texts.append(completion[0]["content"])
        else:
            texts.append(str(completion))

    gt_strs = [str(a) for a in answer]
    rewards = check_batch(texts, gt_strs)

    elapsed = time.perf_counter() - t0
    _c_time += elapsed
    _c_calls += 1
    return rewards


# ============================================================
# TRAINING
# ============================================================

def run_profile(reward_fn, label: str):
    global _py_time, _py_calls, _c_time, _c_calls
    _py_time = _c_time = 0.0
    _py_calls = _c_calls = 0

    print(f"\n{'='*60}")
    print(f"PROFILING: {label}")
    print(f"{'='*60}")

    dataset = load_gsm8k(NUM_TRAIN_SAMPLES)
    print(f"Dataset: {len(dataset)} samples")

    output_dir = f"{OUTPUT_DIR}/{label}"
    os.makedirs(output_dir, exist_ok=True)

    config = GRPOConfig(
        output_dir=output_dir,
        run_name=f"verify-profile-{label}",
        learning_rate=5e-6,
        bf16=True,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        num_generations=NUM_GENERATIONS,
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
        reward_funcs=reward_fn,
        args=config,
        train_dataset=dataset,
    )

    wall_start = time.perf_counter()
    trainer.train()
    wall_total = time.perf_counter() - wall_start

    # Collect timing
    verify_time = _py_time if "python" in label else _c_time
    verify_calls = _py_calls if "python" in label else _c_calls

    results = {
        "label": label,
        "wall_clock_sec": round(wall_total, 2),
        "verify_time_sec": round(verify_time, 4),
        "verify_calls": verify_calls,
        "verify_pct": round(verify_time / wall_total * 100, 2) if wall_total > 0 else 0,
        "avg_verify_ms": round(verify_time / verify_calls * 1000, 3) if verify_calls > 0 else 0,
        "num_samples": NUM_TRAIN_SAMPLES,
        "num_generations": NUM_GENERATIONS,
        "model": MODEL_NAME,
    }

    print(f"\n--- {label} Results ---")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open(f"{output_dir}/profile_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Free GPU memory
    del trainer, model
    torch.cuda.empty_cache()

    return results


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("PufferVerify GRPO Profiling")
    print(f"Model: {MODEL_NAME}")
    print(f"Samples: {NUM_TRAIN_SAMPLES}, G={NUM_GENERATIONS}")
    print("=" * 60)

    # Run Python baseline
    py_results = run_profile(python_reward, "python_verifier")

    # Run C verifier
    c_results = run_profile(c_reward, "c_verifier")

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<30} {'Python':>12} {'C':>12}")
    print("-" * 54)
    print(f"{'Wall clock (sec)':<30} {py_results['wall_clock_sec']:>12.1f} {c_results['wall_clock_sec']:>12.1f}")
    print(f"{'Verify time (sec)':<30} {py_results['verify_time_sec']:>12.4f} {c_results['verify_time_sec']:>12.4f}")
    print(f"{'Verify % of wall clock':<30} {py_results['verify_pct']:>11.2f}% {c_results['verify_pct']:>11.2f}%")
    print(f"{'Avg verify (ms/call)':<30} {py_results['avg_verify_ms']:>12.3f} {c_results['avg_verify_ms']:>12.3f}")
    print(f"{'Verify calls':<30} {py_results['verify_calls']:>12} {c_results['verify_calls']:>12}")

    speedup = py_results['wall_clock_sec'] / c_results['wall_clock_sec'] if c_results['wall_clock_sec'] > 0 else 0
    verify_speedup = py_results['verify_time_sec'] / c_results['verify_time_sec'] if c_results['verify_time_sec'] > 0 else 0
    print(f"\nWall-clock speedup: {speedup:.2f}x")
    print(f"Verify-only speedup: {verify_speedup:.2f}x")
    print(f"Verification is {py_results['verify_pct']:.1f}% of total time (Python baseline)")

    # Save combined summary
    summary = {
        "python": py_results,
        "c": c_results,
        "wall_clock_speedup": round(speedup, 2),
        "verify_speedup": round(verify_speedup, 2),
    }
    with open(f"{OUTPUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/summary.json")

    # Decision gate
    print("\n" + "=" * 60)
    print("DECISION GATE")
    print("=" * 60)
    if py_results['verify_pct'] > 10:
        print("Verification is >10% of wall clock → STRONG signal for C verifier value")
    elif py_results['verify_pct'] > 5:
        print("Verification is 5-10% of wall clock → MODERATE signal, test larger G next")
    else:
        print("Verification is <5% of wall clock → WEAK signal for simple numerical comparison")
        print("Value may come from: complex verification (SymPy replacement), larger G, or reward server mode")


if __name__ == "__main__":
    main()
