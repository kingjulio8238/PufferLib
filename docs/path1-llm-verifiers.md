# Path 1: LLM RL Verifiers — Fast C/CUDA Reward Evaluation for GRPO/RLHF

**Status:** EXPLORE FIRST
**Priority:** Highest ROI — every frontier AI lab needs this
**Target customers:** AI labs doing reasoning model training

---

## The Problem

GRPO/RLVR is how every frontier lab trains reasoning LLMs. The loop:

```
LLM generates N solutions → Verifier scores each → Reward → Update LLM
```

### Where Time Is Actually Spent

The RL training pipeline has THREE bottlenecks, not one:

1. **Generation (the biggest):** LLM produces long chain-of-thought sequences. For reasoning models (o1-style), CoT can be thousands of tokens. This dominates wall-clock time.

2. **Verification:** Scoring each completion. Ranges from microseconds (numerical comparison) to seconds (code sandbox execution) to minutes (LLM-as-judge).

3. **Training:** Gradient updates. Well-optimized by existing frameworks (DeepSpeed, FSDP).

**Key insight from PipelineRL (ServiceNow, 2025):** The solution isn't to speed up one stage — it's to PIPELINE them. PipelineRL runs generation and training concurrently with in-flight weight updates, achieving 2x faster learning on 128 H100s. The bottleneck shifts to whatever can't be pipelined or parallelized.

**Key insight from AReaL (Tsinghua/Ant Group):** Fully async RL (generation, reward, training all running independently) achieves 2.77x speedup. They support GRPO, PPO, DAPO, REINFORCE++, RLOO.

**Key insight from Jack Morris (scaling RL to 10^26 FLOPS):** "CPU bottlenecks in verification slow GPU utilization on expensive hardware ($500K servers)." Verification becomes the bottleneck precisely when you scale generation with more GPUs — the verification can't keep up.

### The Verification Gap (Karpathy's Framing)

Creation has two modes: generation (making) and discrimination (checking). LLMs have made generation near-instant but have done almost nothing to speed up discrimination. For code: "you could say coding LLMs have collapsed (1) to instant, but have done very little to address (2)."

This is the exact gap PufferLib can fill: make discrimination/verification as fast as generation.

## The Opportunity — Where Speed Actually Matters

### Scenario A: Verification IS the bottleneck (high value)

When labs scale to many GPUs for generation, verification becomes the constraint:
- 128 H100s generating completions → millions of completions per hour
- Each needs scoring → verification throughput must match generation throughput
- Python sandboxes at 1 eval/sec can't keep up with 128 GPUs generating
- C/CUDA verification at millions/sec eliminates this bottleneck entirely

### Scenario B: Verification is NOT the bottleneck (still valuable)

Even if generation dominates, fast verification enables:
- **More completions per prompt (larger G in GRPO):** Currently G=8-64 because verification is expensive. With instant verification, G=1000+ becomes practical → better reward signal → better training.
- **Richer reward signals:** Instead of binary correct/incorrect, compute partial credit, multi-dimensional scores — things too expensive to compute at scale in Python.
- **Real-time filtering:** Drop clearly wrong completions before they waste training compute.

## Integration Landscape

### Existing RL Training Frameworks

| Framework | Org | Async? | Reward Interface | License |
|---|---|---|---|---|
| **OpenRLHF** | Community | Yes (Ray) | `reward_func.py` → `{"rewards": [...]}` | MIT |
| **AReaL** | Tsinghua/Ant | Fully async | Reward workers (modular) | Apache 2.0 |
| **PipelineRL** | ServiceNow | Pipelined | Binary reward + Redis streaming | Open source |
| **Verifiers** | Jan.ai fork / PrimeIntellect | Sync | Rubric class with reward functions | MIT |
| **prime-rl** | PrimeIntellect | Async | Environment-based rewards | — |
| **TRL** | Hugging Face | Sync | Callable reward function | Apache 2.0 |

### Integration Priority

1. **OpenRLHF** — largest community, easiest integration (Python reward function calling C library)
2. **AReaL** — modular reward workers, natural fit for a fast C reward server
3. **Verifiers (Jan.ai fork)** — GRPO trainer with pluggable rubrics, good for multi-turn

---

## Steps

### Step 1: Build C math verifier locally (no GPU needed)

Build a standalone C library that parses and verifies numerical answers.

**Location:**
```
pufferlib/verify/
  math_verify.c    — Core: extract number from string, compare to ground truth
  math_verify.h    — Public API
  test_verify.c    — Test harness with known inputs/outputs
  Makefile         — Build standalone .so
```

**Core API:**
```c
typedef struct {
    const char* response;     // Full LLM response text
    const char* ground_truth; // Expected answer string
    float reward;             // Output: 1.0 correct, 0.0 incorrect
    float parsed_value;       // Output: what number was extracted
    int parse_success;        // Output: 1 if a number was found
} VerifyResult;

// Verify a single response
VerifyResult verify_math(const char* response, const char* ground_truth, float tolerance);

// Verify a batch (OpenMP parallel)
void verify_math_batch(VerifyResult* results, int n, float tolerance);
```

**Parser must handle real LLM output formats:**
- `The answer is 42.` → 42
- `\boxed{42}` → 42
- `#### 42` → 42 (GSM8K format)
- `$42` or `42%` or `42,000` → 42, 42 (as percent), 42000
- `3/4` → 0.75
- `-3.14` → -3.14
- `The answer is forty-two` → 42 (stretch goal, not required for Phase 1)

**Test data:** Download GSM8K test set from Hugging Face (1,319 problems with ground truth answers). Write a Python script to generate synthetic LLM responses with known correct/incorrect answers for testing.

**Deliverables:**
- [ ] `libpuffer_verify.so` builds on Linux and macOS
- [ ] Passes test suite of 100+ edge cases
- [ ] Python bindings via ctypes (simple, no pybind11 dependency)

### Step 2: Benchmark C verifier vs Python baseline locally (no GPU needed)

**Python baseline** (what GRPO pipelines use):
```python
import re
def verify_math_python(response, ground_truth):
    # Extract last number from response
    numbers = re.findall(r'[-+]?\d*\.?\d+', response)
    if not numbers:
        return 0.0
    pred = float(numbers[-1])
    gt = float(ground_truth)
    return 1.0 if abs(pred - gt) < 1e-5 else 0.0
```

**Benchmark script:** `benchmark_verify.py`
- Load GSM8K test set
- Generate N synthetic responses per problem (varying formats, correct and incorrect)
- Time Python verifier on full batch
- Time C verifier on full batch (via ctypes)
- Report: throughput (verifications/sec), accuracy agreement, latency distribution

**Measure at multiple batch sizes:** 1K, 10K, 100K, 1M

**Success criteria:**
- [ ] >99% agreement with Python baseline on GSM8K
- [ ] >100x single-threaded speedup over Python
- [ ] >1000x batched speedup with OpenMP
- [ ] Results documented with exact numbers

### Step 3: Build Python package with OpenRLHF-compatible interface (no GPU needed)

```
pufferlib/verify/
  __init__.py
  _binding.py      — ctypes wrapper around libpuffer_verify.so
  math.py          — High-level Python API
  openrlhf.py      — Drop-in reward_func for OpenRLHF
```

**Usage:**
```python
from pufferlib.verify import math

# Batch verify
rewards = math.check_batch(responses, ground_truths)

# OpenRLHF integration
from pufferlib.verify.openrlhf import reward_func
# Pass to OpenRLHF: --reward_func_path pufferlib/verify/openrlhf.py
```

**Deliverables:**
- [ ] `pip install -e .` works
- [ ] `reward_func` matches OpenRLHF's expected signature
- [ ] Unit tests pass

### Step 4: Profile GRPO training on GPU (requires RunPod)

Rent an RTX 4090 or A100 on RunPod. Run OpenRLHF GRPO on GSM8K:

```bash
# Install OpenRLHF
pip install openrlhf

# Run GRPO with Python verifier, instrument timing
python -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path python_verifier.py \
  --dataset gsm8k \
  --num_episodes 500

# Same run with C verifier
python -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path pufferlib/verify/openrlhf.py \
  --dataset gsm8k \
  --num_episodes 500
```

**Instrument and measure:**
- Total wall-clock time
- Time in generation (vLLM inference)
- Time in verification (reward computation)
- Time in training (gradient updates)
- Breakdown as percentages

**This answers the critical question:** What % of GRPO wall-clock is verification?

**Deliverables:**
- [ ] Timing breakdown documented (generation % / verification % / training %)
- [ ] Wall-clock comparison: Python verifier vs C verifier
- [ ] Training curves compared (should be identical — same rewards, just faster)

### Step 5: Test larger G hypothesis (on same GPU session)

If verification is cheap with C verifier, test whether more completions per prompt helps:

- G=8 (standard) → measure final accuracy on MATH
- G=32 → measure
- G=128 → measure
- G=512 → measure (if memory allows)

**This answers:** Does making verification instant unlock better training by enabling larger G?

**Deliverables:**
- [ ] Accuracy vs G curve
- [ ] Training speed vs G curve
- [ ] Clear conclusion: does larger G help?

### Step 6: Decision gate

Review all data from Steps 1-5 and decide:

| Outcome | Decision |
|---|---|
| C verifier is 100x+ faster AND verification is >10% of wall-clock | Strong proceed — direct speedup value |
| C verifier is 100x+ faster BUT verification is <5% of wall-clock, AND larger G helps | Proceed — value is enabling larger G |
| C verifier is 100x+ faster BUT verification is <5% AND larger G doesn't help | Weak — still useful as open-source tool but not a business. Consider pivoting to Path 2 or 3 |
| C verifier can't match Python accuracy (parsing failures) | Fix parser first, then re-evaluate |

---

## Phase 2: Expand Verifier Types (if Step 6 is "proceed")

### Math expression equivalence
- Parse mathematical expressions (text and LaTeX)
- Canonical form comparison (sort terms, normalize coefficients)
- Limited symbolic simplification in C
- Benchmark against SymPy

### Structured output validation
- JSON schema validation in C (fast jsonschema alternative)
- Format checking for function calling outputs
- XML/tag extraction and validation

### Code output matching
- Pre-computed expected outputs → fast string/numerical comparison
- NOT sandboxed execution (different approach needed)
- Useful for: HumanEval post-execution checking, competitive programming

### Multi-dimensional rewards
- Partial credit scoring (not just binary)
- Multiple reward dimensions (correctness, format, reasoning quality)
- Weighted composite scores — configurable per task

## Phase 3: Production Package (if Phase 2 validates)

### "PufferVerify" Python package

```python
import puffer_verify

# Batch verification
results = puffer_verify.math.check_batch(
    answers=["42", "forty-two", "41", "42.0"],
    ground_truths=["42", "42", "42", "42"]
)  # [1.0, 1.0, 0.0, 1.0]

# OpenRLHF drop-in
from puffer_verify.integrations import openrlhf_reward_func

# AReaL reward worker
from puffer_verify.integrations import areal_reward_worker
```

### Async reward server mode

For AReaL/PipelineRL integration — run as a standalone reward server:
```bash
puffer-verify serve --port 8080 --verifier math --workers 16
```
Accepts HTTP requests with batches of (completion, reference) pairs, returns rewards.
Slots into any async RL pipeline without code changes on their side.

### Performance targets

| Verifier type | Python baseline | PufferVerify target |
|---|---|---|
| Numerical comparison | ~10K/sec | >10M/sec |
| Expression equivalence | ~1K/sec | >1M/sec |
| JSON validation | ~50K/sec | >5M/sec |
| Output string matching | ~100K/sec | >10M/sec |

## Key Risks

1. **Verification may not be the bottleneck at current scale.** At G=8 with a single GPU, generation dominates. But at G=64 on 128 GPUs, verification becomes the constraint. Step 4 measures which regime matters.

2. **Deterministic verifiers are the easy case.** Many important tasks need LLM-as-judge or sandboxed code execution. The addressable market for C-speed verifiers is math, structured output, and code output matching — not arbitrary reasoning.

3. **Parsing LLM output is messy.** Real model outputs have varied formatting. The C parser needs battle-testing on actual model outputs, not clean test data.

4. **The "next-token prediction is verifiable" thesis (Jack Morris) could make specialized verifiers obsolete.** If RL scales to web data with next-token prediction as the reward, specialized math/code verifiers become niche. But this is speculative and not imminent.

5. **Adoption friction.** Labs have existing pipelines. Integration must be zero-friction: `pip install puffer-verify`, one line change in reward function.

## Data Requirements

All freely available on Hugging Face:
- **GSM8K:** 8.5K train, 1.3K test (grade school math, numerical answers)
- **MATH:** 12.5K problems (competition math, LaTeX answers)
- **HumanEval/MBPP:** Code generation with expected outputs
- **ARC/HellaSwag:** Multiple choice (trivial verification)

No proprietary data needed.

## Revenue Model

- **Open-source core:** PufferVerify library (MIT, drives adoption)
- **Enterprise:** Custom verifier development, dedicated reward server deployment, SLA
- **Cloud:** Hosted PufferVerify API (pay per million verifications)
- **Integration consulting:** Help AI labs integrate into their training pipelines
