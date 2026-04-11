# Path 1: On-Pod Validation Steps (Steps 4-5)

**Prerequisites:** RTX 4090 or A100 RunPod instance, 50GB disk

---

## Step 4: Profile GRPO Training — Measure Verification % of Wall-Clock

### Setup

```bash
# Clone repo (use your own PAT or SSH key)
git clone https://github.com/kingjulio8238/PufferLib.git
cd PufferLib
git checkout 4.0

# Install PufferLib (for verify module)
pip install -e .

# Build the C verifier
cd pufferlib/verify && make all && make test && cd ../..

# Install OpenRLHF
pip install openrlhf vllm

# Download GSM8K
python3 -c "
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main')
ds.save_to_disk('/root/gsm8k')
print(f'Train: {len(ds[\"train\"])}, Test: {len(ds[\"test\"])}')
"
```

### Run GRPO with Python verifier (baseline)

```bash
# Create baseline Python reward function
cat > /root/python_reward_func.py << 'PYEOF'
import re

def reward_func(queries, prompts, labels):
    rewards = []
    for query, prompt, label in zip(queries, prompts, labels):
        completion = query[len(prompt):] if query.startswith(prompt) else query
        numbers = re.findall(r'[-+]?\d[\d,]*\.?\d*', completion)
        if not numbers:
            rewards.append(0.0)
            continue
        try:
            pred = float(numbers[-1].replace(',', ''))
            gt = float(label.replace(',', ''))
            rewards.append(1.0 if abs(pred - gt) < 1e-5 else 0.0)
        except ValueError:
            rewards.append(0.0)
    return {"rewards": rewards, "scores": rewards}
PYEOF

# Run GRPO training with timing instrumentation
# Adjust flags based on available GPU memory
python3 -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path /root/python_reward_func.py \
  --dataset openai/gsm8k \
  --input_key question \
  --label_key answer \
  --max_samples 500 \
  --num_episodes 200 \
  --logging_steps 10 \
  2>&1 | tee /root/grpo_python_baseline.log
```

**Note:** The exact OpenRLHF CLI flags may differ by version. Check `python3 -m openrlhf.cli.train_ppo --help` and adjust. The key is to run the same config twice — once with Python verifier, once with C verifier.

### Run GRPO with C verifier

```bash
python3 -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path pufferlib/verify/openrlhf.py \
  --dataset openai/gsm8k \
  --input_key question \
  --label_key answer \
  --max_samples 500 \
  --num_episodes 200 \
  --logging_steps 10 \
  2>&1 | tee /root/grpo_c_verifier.log
```

### Measure

After both runs, compare:

```bash
# Extract timing from logs
echo "=== Python verifier ==="
grep -E "time|step|reward|throughput" /root/grpo_python_baseline.log | tail -20

echo "=== C verifier ==="
grep -E "time|step|reward|throughput" /root/grpo_c_verifier.log | tail -20
```

**What to record:**
- Total wall-clock time for both runs
- Per-step timing breakdown (if OpenRLHF reports it)
- Final reward/accuracy (should be identical)
- Observation: what % of time is generation vs verification vs training?

---

## Step 5: Test Larger G Hypothesis

If C verification is effectively free, test whether more completions per prompt improves training:

```bash
# G=8 (standard)
python3 -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path pufferlib/verify/openrlhf.py \
  --dataset openai/gsm8k \
  --num_generations 8 \
  --max_samples 500 \
  2>&1 | tee /root/grpo_g8.log

# G=64
python3 -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path pufferlib/verify/openrlhf.py \
  --dataset openai/gsm8k \
  --num_generations 64 \
  --max_samples 500 \
  2>&1 | tee /root/grpo_g64.log

# G=256 (if memory allows)
python3 -m openrlhf.cli.train_ppo \
  --pretrain Qwen/Qwen2.5-1.5B \
  --reward_func_path pufferlib/verify/openrlhf.py \
  --dataset openai/gsm8k \
  --num_generations 256 \
  --max_samples 500 \
  2>&1 | tee /root/grpo_g256.log
```

**What to record:**
- Final accuracy on held-out test set for each G
- Training wall-clock time for each G
- Whether larger G gives better or equivalent accuracy in fewer optimizer steps

---

## Step 6: Decision Gate

After Steps 4-5, evaluate:

| Result | Decision |
|---|---|
| Verification > 10% of wall-clock AND C is faster end-to-end | Strong proceed — direct speedup value |
| Verification < 5% BUT larger G improves training | Proceed — value is enabling larger G |
| Verification < 5% AND larger G doesn't help | Weak — pivot toward complex verification (SymPy replacement) or reward server mode |

### Upload results

```bash
# Save all logs
tar czf /root/verify_results.tar.gz /root/grpo_*.log

# Upload to HF
python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='YOUR_HF_TOKEN')
api.create_repo('kingjulio/pufferlib-verify-results', exist_ok=True)
api.upload_file(
    path_or_fileobj='/root/verify_results.tar.gz',
    path_in_repo='verify_results.tar.gz',
    repo_id='kingjulio/pufferlib-verify-results'
)
print('Uploaded!')
"
```

---

## Troubleshooting

### OpenRLHF version mismatch
CLI flags change between versions. Run `pip show openrlhf` to check version, then consult their docs for exact flag names.

### Out of memory
Reduce `--max_samples`, `--num_generations`, or use a smaller model. The 1.5B model should fit on a 24GB GPU with G=8.

### C library not found
```bash
cd pufferlib/verify && CC=clang make clean && CC=clang make all
```

### vLLM issues
OpenRLHF uses vLLM for generation. If vLLM has CUDA issues, try: `pip install vllm --upgrade`
