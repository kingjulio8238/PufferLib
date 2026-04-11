# Path 1: On-Pod Steps (Steps 4-5)

**Pod:** RunPod, RTX 4090, `runpod/pytorch:2.9.0-py3.12-cuda12.8.1-cudnn-devel-ubuntu22.04`, 50GB container disk, 0 volume disk

---

## Setup (copy-paste this block)

```bash
# Clone and install
git clone https://github.com/kingjulio8238/PufferLib.git
cd PufferLib && git checkout 4.0
pip install -e .
pip install trl datasets accelerate

# Build C verifier
cd pufferlib/verify && CC=gcc make clean && CC=gcc make all && CC=gcc make test && cd ../..

# Verify
python3 -c "from pufferlib.verify import check; print('OK:', check('#### 42', '42'))"
python3 -c "from trl import GRPOTrainer; print('TRL OK')"
```

## Step 4: Profile GRPO (Python vs C verifier)

```bash
python3 -m pufferlib.verify.onpod_profile
```

This runs GRPO training twice (Python verifier, then C verifier) and prints:
- Wall-clock time for each
- Verification time as % of total
- Verify-only speedup

## Step 5: Test larger G

```bash
python3 -m pufferlib.verify.onpod_larger_g
```

Tests G=8, G=32, G=64. Reports wall-clock and final reward for each.

If G=64 OOMs, edit the script: reduce `NUM_TRAIN_SAMPLES` to 100 or `MAX_COMPLETION_LENGTH` to 256.

## Upload results

```bash
pip install huggingface_hub
python3 -c "
from huggingface_hub import HfApi
import os, tarfile
with tarfile.open('/root/results.tar.gz', 'w:gz') as tar:
    for d in ['/root/verify_profile', '/root/verify_larger_g']:
        if os.path.exists(d): tar.add(d)
api = HfApi(token='YOUR_HF_TOKEN')
api.create_repo('kingjulio/pufferlib-verify-results', exist_ok=True)
api.upload_file('/root/results.tar.gz', 'results.tar.gz', 'kingjulio/pufferlib-verify-results')
print('Done!')
"
```

## Troubleshooting

- **OOM:** Reduce `NUM_TRAIN_SAMPLES`, `MAX_COMPLETION_LENGTH`, or `num_generations` in the scripts
- **TRL version mismatch:** `pip install trl --upgrade` — the scripts use GRPOConfig/GRPOTrainer
- **flash_attention_2 error:** `pip install flash-attn --no-build-isolation`
- **C library not found:** `cd pufferlib/verify && CC=gcc make clean && CC=gcc make all`
