# Path 2: Drug Discovery — GPU-Accelerated Molecular Scoring for RL-Driven Design

**Status:** EXPLORE AFTER PATH 1
**Priority:** Highest revenue potential per customer
**Target customers:** Pharma (AstraZeneca, Novartis, Roche), biotech (Insilico Medicine, Recursion), CROs

---

## The Problem

De novo molecular design with RL (REINVENT4, ACEGEN) generates molecules and scores them in a loop:

```
Generator proposes molecule (SMILES string)
  → Scoring function evaluates properties (QED, SA, similarity, docking)
  → Reward signal updates generator
  → Repeat 10⁵ - 10⁶ times
```

The scoring function is the bottleneck:
- Runs on CPU in Python (RDKit, custom scorers)
- Simple property calculations (QED, Lipinski) take ~1ms each in Python
- Docking simulations take seconds-minutes each
- At 10⁵ molecules per run, total scoring time dominates

## The Opportunity

Implement the common "fast" scoring functions in C/CUDA:
- **QED (Quantitative Estimate of Drug-likeness):** weighted geometric mean of 8 molecular descriptors
- **SA Score (Synthetic Accessibility):** fragment-based complexity estimate
- **Lipinski's Rule of 5:** MW, LogP, HBD, HBA thresholds
- **Tanimoto Similarity:** fingerprint comparison to reference molecule
- **Custom property filters:** molecular weight, rotatable bonds, TPSA, etc.

These are all pure math on molecular graph/fingerprint representations. No quantum chemistry needed.

**Note:** Docking (the expensive part) cannot be replaced by simple C code — it requires physics simulation. But the fast filters gate what goes to docking, so speeding them up reduces the total pipeline time.

## Prerequisites / Risks

### Chemistry Domain Knowledge Required
- Molecular fingerprint computation (ECFP, MACCS) requires understanding of molecular graphs
- QED calculation involves 8 specific descriptor distributions from Bickerton et al. 2012
- SA score requires a fragment frequency database (1M+ fragments from PubChem)
- **Mitigation:** Use RDKit's open-source C++ implementation as reference. Port specific functions, not the whole library.

### Data Requirements
- **Fragment database for SA score:** Available from RDKit open source (fpscores.pkl.gz)
- **QED descriptor distributions:** Published in the original paper, hardcoded constants
- **Test molecules:** ChEMBL and ZINC databases (free, millions of molecules)
- **REINVENT4 benchmarks:** Available in their repo for comparison
- All data is publicly available. No proprietary datasets needed.

## Validation Plan — Phase 1: Proof of Concept
### Step 1: Implement Tanimoto similarity in C

Start with the simplest scorer — fingerprint-based Tanimoto similarity:
- Input: two molecular fingerprints (bit vectors, 1024 or 2048 bits)
- Output: similarity score [0, 1]
- Pure bitwise operations: popcount(A & B) / popcount(A | B)
- CUDA-trivial: one thread per molecule pair

This requires NO chemistry knowledge — just bit manipulation.

### Step 2: Benchmark against RDKit

```python
from rdkit import Chem, DataStructs
# Python/RDKit baseline: compute Tanimoto for 100K molecule pairs
# C/CUDA version: same 100K pairs
# Measure: throughput (pairs/sec), accuracy (exact match)
```

**Success criteria:**
- [ ] Exact numerical match with RDKit on 100K pairs
- [ ] >1000x throughput improvement over RDKit Python

### Step 3: Implement Lipinski Rule of 5

Next simplest — 4 threshold checks on pre-computed descriptors:
- MW < 500, LogP < 5, HBD < 5, HBA < 10
- Input: descriptor vector (4 floats per molecule)
- Output: pass/fail + individual scores

### Step 4: Integrate with REINVENT4 as custom scorer

REINVENT4 supports plugin scoring components. Build a Python wrapper:
```python
class PufferScorer(ScoringComponent):
    def __call__(self, smiles_list):
        fingerprints = [compute_fp(s) for s in smiles_list]  # RDKit (keep)
        scores = puffer_score.tanimoto_batch(fingerprints, self.reference)  # C (fast)
        return scores
```

**Success criteria:**
- [ ] Drop-in replacement for REINVENT4's Tanimoto scorer
- [ ] Measurable speedup in a full REINVENT4 optimization run
- [ ] Correct results on standard benchmarks (GuacaMol)

## Validation Plan — Phase 2: Full Scoring Suite
If Phase 1 succeeds, implement:

### QED Score
- 8 molecular descriptors weighted by published distributions
- Requires: MW, ALOGP, HBA, HBD, PSA, ROTB, AROM, ALERTS
- Reference: Bickerton et al., Nature Chemistry 2012
- Complexity: Medium (need descriptor calculation, not just thresholds)

### SA Score  
- Fragment-based: decompose molecule into fragments, score by frequency
- Requires: pre-computed fragment frequency table (~1M entries)
- Reference: Ertl & Schuffenhauer, J. Cheminf. 2009
- Complexity: Medium-High (fragment enumeration is graph algorithm)

### Multi-objective scoring
- Combine multiple scorers with weights
- Desirability functions (sigmoid transforms)
- Pareto-optimal molecule selection

## Validation Plan — Phase 3: Standalone Product
### Build "PufferChem" package
```python
import puffer_chem

# Batch scoring — the core value proposition
scores = puffer_chem.score_batch(
    smiles=["CCO", "c1ccccc1", ...],  # 100K molecules
    scorers=["qed", "sa", "lipinski", "tanimoto"],
    reference_mol="CC(=O)Oc1ccccc1C(=O)O"  # aspirin
)
# Returns in milliseconds, not minutes
```

### Integration targets
1. REINVENT4 plugin scorer
2. ACEGEN custom scoring function
3. Standalone Python package for any molecular optimization pipeline

## Key Risks

1. **Fingerprint computation is the hard part.** Tanimoto on pre-computed fingerprints is trivial. But computing ECFP fingerprints from SMILES requires a full molecular graph library. This is where RDKit's complexity lives. We may need to keep RDKit for fingerprint generation and only accelerate the scoring math.

2. **Docking is the real bottleneck.** For serious drug discovery, docking scores matter more than QED/SA. Docking cannot be trivially implemented in C — it's a physics simulation. Our fast scorers would be filters that reduce what goes to docking, not replacements.

3. **Chemistry expertise needed.** Unlike game environments, mistakes in molecular scoring can produce scientifically invalid results. Need domain expert review.

4. **Market access.** Pharma companies are slow to adopt open-source tools. Sales cycle is 6-18 months. Biotech startups are faster adopters.

## Revenue Model

- **Open-source core:** PufferChem scoring library (drives adoption, community contributions)
- **Enterprise license:** Priority support, custom scorer development, validation against internal datasets
- **SaaS:** Hosted scoring API (pay per million molecules scored)
- **Consulting:** Integration with customer's molecular design pipeline

**Pricing benchmark:** Schrödinger (molecular modeling) has $200M+ ARR. Chemical computing is a real market.

## Decision Point

After Phase 1 (3-5 days), we will have data on:
1. Can we match RDKit's accuracy on Tanimoto similarity?
2. Is the speedup actually significant (>100x)?
3. Can we plug into REINVENT4 without friction?

If YES → proceed to Phase 2 (QED, SA implementation).
If fingerprint computation is the real bottleneck (not scoring) → reconsider scope.
If pharma adoption looks too slow → deprioritize vs Path 1 or Path 3.
