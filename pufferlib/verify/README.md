# PufferVerify — Fast Math Verification for LLM RL Training

C-accelerated reward verification for GRPO/RLHF training pipelines.

## Current Status

**Step 1-3 complete (local). Steps 4-5 require GPU (RunPod).**

### What's Built
- C math verifier with OpenMP batch mode (30M verifications/sec pure C)
- Python bindings via ctypes (bulk buffer API)
- OpenRLHF `reward_func` drop-in integration
- AReaL reward worker stub
- 62 C tests, all passing
- Benchmark suite (C vs Python)

### Benchmark Results (local, Apple M-series)

| Mode | Throughput | vs Python |
|---|---|---|
| Python regex baseline | ~750K/sec | 1x |
| C via ctypes (bulk) | ~2.3M/sec | 3x |
| Pure C (no Python) | 30M/sec | 40x |

100% accuracy agreement with Python baseline on all test cases.

## Quick Start

```bash
# Build the C library
cd pufferlib/verify && make all

# Run C tests
make test

# Run benchmark
cd ../.. && python3 -m pufferlib.verify.benchmark
```

## Python API

```python
from pufferlib.verify import check, check_batch, extract_answer

# Single verification
check("The answer is 42", "42")  # 1.0
check("The answer is 41", "42")  # 0.0

# Batch verification
check_batch(
    ["\\boxed{42}", "#### 7", "I got 10"],
    ["42", "7", "11"]
)  # [1.0, 1.0, 0.0]

# Answer extraction
extract_answer("Step 1: 10\\nStep 2: 20\\n#### 30")  # "30"
```

## OpenRLHF Integration

```bash
python -m openrlhf.cli.train_ppo \
  --reward_func_path pufferlib/verify/openrlhf.py \
  ...
```

## Supported Answer Formats

- Plain numbers: `The answer is 42`
- GSM8K: `#### 42`
- LaTeX: `\boxed{42}`
- Fractions: `3/4`
- Commas: `42,000`
- Negatives/decimals: `-3.14`
- Percentages: `42%`

Priority: `\boxed{}` > `####` > last number in text.

## On-Pod Steps (Steps 4-5)

See below for GPU validation instructions.
