// g1gpu: GPU-NATIVE G1 environment for PufferLib (mujoco-ultra-fast Phase 3).
//
// The Phase-2 staged engine (GATE-D: full env 5.68M physics steps/s on one
// H100 @131k envs, validated vs the MuJoCo C engine + the CPU g1 env) wired
// to PufferLib's vecenv device buffers. NO libmujoco, no runtime data files:
// topology + solver constants are baked at codegen (g1phys/ headers, synced
// from mujoco-ultra-fast by scripts/sync_g1phys_to_fork.sh).
//
// Hooks consumed by vecenv.h under MY_GPU_NATIVE (see ocean/g1gpu/binding.c):
//   my_gpu_config  — reward weights etc. from env kwargs (cudaMemcpyToSymbol)
//   my_gpu_init    — allocate per-env SoA state for total_agents
//   my_gpu_reset   — reset all envs + write initial obs into vec buffer
//   my_gpu_step_range — one control step for a buffer's agent range, on the
//                       buffer's stream: K_act(vec actions) -> 10x staged
//                       physics -> K_epi(obs/rew/done into vec buffers)
//   my_gpu_log_into — episode-log aggregation (device accumulators -> Log)
//   my_gpu_close

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "g1phys/g1_topology.cuh"
#include "g1phys/common.cuh"
#include "g1phys/traj_format.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("G1GPU CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

#define WARPS_PER_BLOCK 2
// Promoted to DEFAULT-ON (2026-06-16): K5_PARALLEL (k5 Jacobian-walk lane recovery)
// + K78_FUSE (k7+k8 fused, H in smem) = +3.4% bit-exact (1.296M->1.340M, 89.3s->86.4s),
// the standing best at user-approved quality. Opt out for A/B with -DNO_K5_PARALLEL /
// -DNO_K78_FUSE. (K_NEWTON_FUSE stays opt-IN — it's marginal/shelved.)
#ifndef NO_K5_PARALLEL
#define K5_PARALLEL 1
#endif
#ifndef NO_K78_FUSE
#define K78_FUSE 1
#endif
#include "g1phys/g1_step.cuh"
#include "g1phys/g1_full_step.cuh"
#include "g1phys/g1_staged_kernels.cuh"
#include "g1phys/g1_solver_consts.cuh"

#ifndef ENV_DECIMATION
#define ENV_DECIMATION 10
#endif
#define ENV_CTRL_DT 0.02f
#define ENV_RESET_NOISE 0.05f
#define ENV_TERM_HEIGHT 0.35f
#define ENV_TERM_GRAVITY_Z (-0.6f)
#define ENV_CMD_RESAMPLE 500
#ifdef G1_TASK_V3
#define ENV_OBS 98
// --- task v3 gait-shaping constants (KEEP IN SYNC: g1.h / stagedenv.cu /
// g1_gpu.cu) — from unitree_rl_gym's proven G1 recipe ---
#define G1_V3_PERIOD 40
#define G1_V3_STANCE 0.55f
#define G1_V3_W_CONTACT 0.5f
#define G1_V3_W_SWING (-20.0f)
#define G1_V3_W_HIP (-4.0f)
#define G1_V3_FOOT_Z0 0.08f
#define G1_V3_LFOOT_BODY 7
#define G1_V3_RFOOT_BODY 13
#define G1_V3_NUM_ACT 12
#define G1_V3_W_BASE_HEIGHT (-10.0f)   // unitree base_height penalty
#define G1_V3_BASE_Z0 0.78f            // target pelvis height (m)
#else
#define ENV_OBS 96
#endif

// runtime-configurable env parameters (the Protein sweep moves these)
__device__ float g1e_action_scale, g1e_w_track_lin, g1e_w_track_ang;
__device__ float g1e_w_lin_vel_z, g1e_w_ang_vel_xy, g1e_w_orientation;
__device__ float g1e_w_torque, g1e_w_action_rate, g1e_w_alive, g1e_w_termination;
__device__ int g1e_max_ep_len;
// command-curriculum scale (1.0 = full range; ramped 0->1 by the host under
// G1_CURRICULUM). Multiplies the sampled velocity command so early training
// sees slow/easy commands, widening to the full distribution as it learns.
__device__ float g1e_cmd_scale = 1.0f;

// ---------------------------------------------------------------------------
// Domain randomization / sim2real (Phase 1). MASTER SWITCH g1e_dr_enable: when
// 0 (default), EVERY DR path below is skipped entirely — no extra RNG draws, no
// added noise — so the env is BIT-IDENTICAL to the pre-DR baseline (gate G1a).
// Ranges are set from env kwargs via my_gpu_config_dr (see binding.c). See
// docs/sim2real.md (D4) for the reference ranges.
// ---------------------------------------------------------------------------
__device__ int   g1e_dr_enable = 0;        // 0 = OFF (bit-exact baseline)
// DR curriculum scale: multiplies ALL DR magnitudes (noise/push/init/friction/mass
// deviation). 1.0 = full DR (default, non-curriculum). Under -DG1_DR_CURRICULUM the
// host ramps it 0->1 (warmup at 0 so the forward gait forms clean, then DR fades in)
// — avoids the full-DR-from-step-0 degenerate basin. friction/mass interpolate toward
// 1.0: scale = 1 + dr_scale*(sampled-1).
__device__ float g1e_dr_scale = 1.0f;
// observation noise (actor-only; uniform U[-1,1]*scale added per channel group).
// These are OBS-SPACE scales — added AFTER the obs scaling, i.e. the EFFECTIVE
// per-element noise, NOT physical units. They equal (unitree physical noise x the
// obs scale). D4 reference (docs/sim2real.md): angvel 0.05, gravity 0.05, dofpos
// 0.01, dofvel 0.075. Do NOT pass physical values here (e.g. dofvel 1.5) — that
// would be 20x too large; pass 0.075 (= 1.5 rad/s x the 0.05 obs scale).
__device__ float g1e_noise_angvel  = 0.0f; // obs[0:3]  (0.05 = 0.2 rad/s x 0.25)
__device__ float g1e_noise_gravity = 0.0f; // obs[3:6]  (0.05, obs scale 1.0)
__device__ float g1e_noise_dofpos  = 0.0f; // obs[9:38] (0.01 = 0.01 rad x 1.0)
__device__ float g1e_noise_dofvel  = 0.0f; // obs[38:67] (0.075 = 1.5 rad/s x 0.05)
// external push: every g1e_push_interval control steps, base xy lin-vel is set
// to U[-push_vel, push_vel] (matches unitree_rl_gym root-state push).
__device__ int   g1e_push_interval = 0;    // 0 = no push
__device__ float g1e_push_vel      = 0.0f; // m/s
// init-state randomization: base lin+ang vel set to U[-init_basevel, +] at reset
__device__ float g1e_init_basevel  = 0.0f; // m/s & rad/s; 0 = keep qvel=0
// per-env friction & mass scale ranges. STAGED: the kwarg interface is wired, but
// these are NOT yet read by the physics kernels — per-env de-baking of
// g1c_pair_friction (k5_assemble, 4 sites) and body inertia (k1, COM+subtree+inertia)
// is the Phase-1 GPU-co-dev remainder (needs a compile+validate-vs-MuJoCo-C loop on
// real hardware). my_gpu_config_dr warns if these are set so it's never a silent no-op.
__device__ float g1e_fric_lo = 1.0f, g1e_fric_hi = 1.0f;
__device__ float g1e_mass_lo = 1.0f, g1e_mass_hi = 1.0f;

// Stateless per-(env,tick,channel) noise in [-1,1] — a hash, NOT the dynamics
// RNG, so observation noise never perturbs the reset/cmd/push RNG stream (keeps
// dynamics reproducible) and needs no extra per-env state.
__device__ __forceinline__ float obs_noise_pm1(int e, int tick, int ch) {
    unsigned int h = (unsigned int)e * 0x9e3779b9u
                   ^ (unsigned int)(tick + 1) * 0x85ebca6bu
                   ^ (unsigned int)(ch + 1) * 0xc2b2ae35u;
    h ^= h >> 15; h *= 0x2c1b3c6du; h ^= h >> 12; h *= 0x297a2d39u; h ^= h >> 15;
    return 2.0f * ((h >> 8) * (1.0f / 16777216.0f)) - 1.0f;
}

// ---------------------------------------------------------------------------
// device helpers (validated semantics from stagedenv.cu)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned int xorshift32(unsigned int* s) {
    unsigned int x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return x;
}
__device__ __forceinline__ float urand_pm1(unsigned int* s) {
    return 2.0f * ((xorshift32(s) >> 8) * (1.0f / 16777216.0f)) - 1.0f;
}
__device__ __forceinline__ float urand_01(unsigned int* s) {
    return (xorshift32(s) >> 8) * (1.0f / 16777216.0f);
}
__device__ __forceinline__ void world_to_base(const float q[4], const float v[3],
                                              float out[3]) {
    float qinv[4] = {q[0], -q[1], -q[2], -q[3]};
    rot_vec_quat(out, v, qinv);
}

// load baked solver constants into the runtime symbols (one block, once)
__global__ void k_init_consts(void) {
    int i = threadIdx.x;
    if (i == 0) g1c_meaninertia = G1_MEANINERTIA;
    for (int k = i; k < G1_NV; k += blockDim.x)
        g1c_dof_invweight0[k] = g1cs_dof_invweight0[k];
    for (int k = i; k < G1_NBODY; k += blockDim.x)
        g1c_body_invweight0_t[k] = g1cs_body_invweight0_t[k];
    for (int k = i; k < G1_NV * 2; k += blockDim.x)
        g1c_dof_solref[k] = g1cs_dof_solref[k];
    for (int k = i; k < G1_NV * 5; k += blockDim.x)
        g1c_dof_solimp[k] = g1cs_dof_solimp[k];
    for (int k = i; k < G1_NJNT * 2; k += blockDim.x) {
        g1c_jnt_solref[k] = g1cs_jnt_solref[k];
        g1c_jnt_range[k] = g1cs_jnt_range[k];
    }
    for (int k = i; k < G1_NJNT * 5; k += blockDim.x)
        g1c_jnt_solimp[k] = g1cs_jnt_solimp[k];
    for (int k = i; k < G1_NJNT; k += blockDim.x)
        g1c_jnt_limited[k] = g1cs_jnt_limited[k];
}

// ---------------------------------------------------------------------------
// per-env episode state + log accumulators (global, full-batch absolute index)
// ---------------------------------------------------------------------------
// eplog layout per env (10 floats):
//   [0] ep_return  [1] ep_len  [2] ep_track_sum  [3] ep_vel_err_sum   (live)
//   [4] ret_total  [5] len_total  [6] perf_total  [7] vel_total
//   [8] falls_total  [9] n_episodes                                   (done)
#define EPLOG_N 10

__device__ void env_reset_g(int e, int lane, unsigned int* g_rng,
                            float* g_qpos, float* g_qvel, float* g_ws,
                            float* g_prev, float* g_cmd, int* g_tick,
                            float* g_eplog, float* g_fric, float* g_mass) {
    for (int k = lane; k < S_NQ; k += 32) g_qpos[(size_t)e * S_NQ + k] = g1c_key_qpos[k];
    for (int k = lane; k < S_NV; k += 32) {
        g_qvel[(size_t)e * S_NV + k] = 0.0f;
        g_ws[(size_t)e * S_NV + k] = 0.0f;
    }
    for (int j = lane; j < S_NU; j += 32) g_prev[(size_t)e * S_NU + j] = 0.0f;
    __syncwarp();
    if (lane == 0) {
        unsigned int rng = g_rng[e];
        for (int j = 0; j < S_NU; j++)
            g_qpos[(size_t)e * S_NQ + 7 + j] += ENV_RESET_NOISE * urand_pm1(&rng);
        if (urand_01(&rng) < 0.1f) {
            g_cmd[3 * e] = g_cmd[3 * e + 1] = g_cmd[3 * e + 2] = 0.0f;
        } else {
            g_cmd[3 * e + 0] = g1e_cmd_scale * 1.0f * urand_pm1(&rng);
            g_cmd[3 * e + 1] = g1e_cmd_scale * 0.6f * urand_pm1(&rng);
            g_cmd[3 * e + 2] = g1e_cmd_scale * 1.0f * urand_pm1(&rng);
        }
        // DR init-state: randomize base lin+ang velocity at reset (gated; skipped
        // when DR off -> no extra RNG draws -> bit-exact baseline).
        if (g1e_dr_enable && g1e_init_basevel > 0.0f) {
            for (int k = 0; k < 6; k++)
                g_qvel[(size_t)e * S_NV + k] = g1e_dr_scale * g1e_init_basevel * urand_pm1(&rng);
        }
        // DR per-env friction & mass scale (resampled each reset; read by k5/k1).
        // Always written 1.0 when off (no RNG draw) so the kernels multiply
        // unconditionally and stay bit-exact (x1.0).
        g_fric[e] = 1.0f; g_mass[e] = 1.0f;
        if (g1e_dr_enable) {
            // sample in [lo,hi], then interpolate toward 1.0 by the curriculum ramp
            // (g1e_dr_scale=0 -> 1.0 = no DR; =1 -> full sampled range).
            float fsamp = g1e_fric_lo + (g1e_fric_hi - g1e_fric_lo) * urand_01(&rng);
            float msamp = g1e_mass_lo + (g1e_mass_hi - g1e_mass_lo) * urand_01(&rng);
            g_fric[e] = 1.0f + g1e_dr_scale * (fsamp - 1.0f);
            g_mass[e] = 1.0f + g1e_dr_scale * (msamp - 1.0f);
        }
        g_rng[e] = rng;
        g_tick[e] = 0;
        g_eplog[(size_t)e * EPLOG_N + 0] = 0.0f;
        g_eplog[(size_t)e * EPLOG_N + 1] = 0.0f;
        g_eplog[(size_t)e * EPLOG_N + 2] = 0.0f;
        g_eplog[(size_t)e * EPLOG_N + 3] = 0.0f;
    }
    __syncwarp();
}

__global__ void k_reset_all(int n, unsigned int seed, unsigned int* g_rng,
                            float* g_qpos, float* g_qvel, float* g_ws,
                            float* g_prev, float* g_cmd, int* g_tick,
                            float* g_eplog, float* g_fric, float* g_mass) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    if (lane == 0) {
        g_rng[e] = seed ^ (0x9e3779b9u * (unsigned int)(e + 1));
        for (int k = 0; k < EPLOG_N; k++) g_eplog[(size_t)e * EPLOG_N + k] = 0.0f;
    }
    __syncwarp();
    env_reset_g(e, lane, g_rng, g_qpos, g_qvel, g_ws, g_prev, g_cmd, g_tick, g_eplog,
                g_fric, g_mass);
}

// write the 96-float obs of the current state (used post-reset and in K_epi)
__device__ void write_obs(int e, int lane, const float* g_qpos, const float* g_qvel,
                          const float* g_prev, const float* g_cmd,
                          const int* g_tick_obs, float* obs) {
    if (lane == 0) {
        float gw[3] = {0.0f, 0.0f, -1.0f}, gb[3];
        world_to_base(g_qpos + (size_t)e * S_NQ + 3, gw, gb);
        obs[0] = 0.25f * g_qvel[(size_t)e * S_NV + 3];
        obs[1] = 0.25f * g_qvel[(size_t)e * S_NV + 4];
        obs[2] = 0.25f * g_qvel[(size_t)e * S_NV + 5];
        obs[3] = gb[0]; obs[4] = gb[1]; obs[5] = gb[2];
        obs[6] = g_cmd[3*e]; obs[7] = g_cmd[3*e+1]; obs[8] = g_cmd[3*e+2];
        // DR observation noise (actor-only; cmd/phase un-noised). Stateless hash
        // keyed on (env,tick,channel) -> independent of the dynamics RNG. Gated:
        // skipped when DR off -> obs bit-identical to baseline.
        if (g1e_dr_enable) {
            int tk = g_tick_obs[e];
            float ds = g1e_dr_scale;   // DR-curriculum ramp (1.0 if no curriculum)
            obs[0] += ds * g1e_noise_angvel  * obs_noise_pm1(e, tk, 0);
            obs[1] += ds * g1e_noise_angvel  * obs_noise_pm1(e, tk, 1);
            obs[2] += ds * g1e_noise_angvel  * obs_noise_pm1(e, tk, 2);
            obs[3] += ds * g1e_noise_gravity * obs_noise_pm1(e, tk, 3);
            obs[4] += ds * g1e_noise_gravity * obs_noise_pm1(e, tk, 4);
            obs[5] += ds * g1e_noise_gravity * obs_noise_pm1(e, tk, 5);
        }
    }
    for (int j = lane; j < S_NU; j += 32) {
        obs[9 + j]  = g_qpos[(size_t)e * S_NQ + 7 + j] - g1c_key_qpos[7 + j];
        obs[38 + j] = 0.05f * g_qvel[(size_t)e * S_NV + 6 + j];
        obs[67 + j] = g_prev[(size_t)e * S_NU + j];
        if (g1e_dr_enable) {
            int tk = g_tick_obs[e];
            float ds = g1e_dr_scale;
            obs[9 + j]  += ds * g1e_noise_dofpos * obs_noise_pm1(e, tk, 9 + j);
            obs[38 + j] += ds * g1e_noise_dofvel * obs_noise_pm1(e, tk, 38 + j);
        }
    }
#ifdef G1_TASK_V3
    if (lane == 0) {
        float phi = (float)(g_tick_obs[e] % G1_V3_PERIOD) / (float)G1_V3_PERIOD;
        obs[96] = sinf(6.2831853f * phi);
        obs[97] = cosf(6.2831853f * phi);
    }
#endif
}

__global__ void k_obs_all(int n, const float* g_qpos, const float* g_qvel,
                          const float* g_prev, const float* g_cmd,
                          const int* g_tick,
                          float* vec_obs) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    write_obs(e, lane, g_qpos, g_qvel, g_prev, g_cmd, g_tick, vec_obs + (size_t)e * ENV_OBS);
}

// actions (vec gpu_actions) -> clamped a + PD ctrl
__global__ void k_act(int n, const float* __restrict__ vec_actions,
                      float* __restrict__ g_act, float* __restrict__ g_ctrl) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    for (int j = lane; j < S_NU; j += 32) {
        float a = fminf(fmaxf(vec_actions[(size_t)e * S_NU + j], -1.0f), 1.0f);
#ifdef G1_TASK_V3
        if (j >= G1_V3_NUM_ACT) a = 0.0f;   // legs-only actions
#endif
        g_act[(size_t)e * S_NU + j] = a;
        float target = g1c_key_ctrl[j] + g1e_action_scale * a;
        float lo = g1c_act_ctrlrange[2 * j], hi = g1c_act_ctrlrange[2 * j + 1];
        g_ctrl[(size_t)e * S_NU + j] = target < lo ? lo : (target > hi ? hi : target);
    }
}

// epilogue: rewards/termination/self-reset/cmd-resample/log + obs into vec
__global__ void k_epi(int n,
                      float* __restrict__ g_qpos, float* __restrict__ g_qvel,
                      float* __restrict__ g_ws,
                      const float* __restrict__ g_xpos,
                      const float* __restrict__ g_footc,
                      const float* __restrict__ g_af,
                      const float* __restrict__ g_act,
                      float* __restrict__ g_prev, float* __restrict__ g_cmd,
                      int* __restrict__ g_tick, unsigned int* __restrict__ g_rng,
                      float* __restrict__ g_eplog,
                      float* __restrict__ g_fric, float* __restrict__ g_mass,
                      float* __restrict__ vec_obs, float* __restrict__ vec_rew,
                      float* __restrict__ vec_term) {
    __shared__ int s_fell[SWARPS], s_to[SWARPS];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    const float* qpos = g_qpos + (size_t)e * S_NQ;
    const float* qvel = g_qvel + (size_t)e * S_NV;
    float* eplog = g_eplog + (size_t)e * EPLOG_N;

    if (lane == 0) g_tick[e] += 1;
    __syncwarp();

    float t2 = 0.0f, ar2 = 0.0f;
    for (int j = lane; j < S_NU; j += 32) {
        float tau = g_af[(size_t)e * S_NU + j];
        t2 += tau * tau;
        float a = g_act[(size_t)e * S_NU + j];
        float da = a - g_prev[(size_t)e * S_NU + j];
        ar2 += da * da;
        g_prev[(size_t)e * S_NU + j] = a;
    }
    for (int o = 16; o > 0; o >>= 1) {
        t2 += __shfl_xor_sync(0xffffffff, t2, o);
        ar2 += __shfl_xor_sync(0xffffffff, ar2, o);
    }
    if (lane == 0) {
        float vb[3], pg[3];
        float gw[3] = {0.0f, 0.0f, -1.0f};
        world_to_base(qpos + 3, qvel, vb);
        world_to_base(qpos + 3, gw, pg);
        float wx = qvel[3], wy = qvel[4], wz = qvel[5];
        float ex = g_cmd[3*e] - vb[0], ey = g_cmd[3*e+1] - vb[1];
        float lin_err2 = ex * ex + ey * ey;
        float track_lin = expf(-lin_err2 / 0.25f);
        float eyaw = g_cmd[3*e+2] - wz;
        float track_ang = expf(-eyaw * eyaw / 0.25f);
        // competence gate: the torso-wobble penalty applies only once upright
        // (pg.z=-1), fading to 0 as the torso tilts toward a fall (pg.z=-0.75).
        // Lets w_ang_vel_xy be STRONG (bites a competent walker) without
        // punishing early flailing or fall-recovery -> no acquisition fight.
        float upr = (-pg[2] - 0.75f) * 4.0f;
        upr = upr < 0.0f ? 0.0f : (upr > 1.0f ? 1.0f : upr);
        float r = g1e_w_alive
                + g1e_w_track_lin * track_lin
                + g1e_w_track_ang * track_ang
                + g1e_w_lin_vel_z * vb[2] * vb[2]
                + g1e_w_ang_vel_xy * (wx * wx + wy * wy) * upr
                + g1e_w_orientation * (pg[0] * pg[0] + pg[1] * pg[1]) * upr
                + g1e_w_torque * t2
#ifdef N3_AR_UPR_GATE
                // upr-gate the action-rate penalty (matches w_ang_vel_xy/orientation
                // above): active when upright (upr~0.9-1 in normal walking) so it
                // smooths the steady gait, but fades to 0 as the torso tilts toward a
                // fall (upr->0 by pg.z=-0.75) so it does NOT suppress the large
                // corrective actions needed for balance/recovery -> no acquisition
                // fight. Lets w_action_rate be bumped hard enough to cut steady-gait
                // jerk without collapsing the walk. Pairs with N3_TIMELIMIT_FIX.
                + g1e_w_action_rate * ar2 * upr;
#else
                + g1e_w_action_rate * ar2;
#endif
#ifdef G1_TASK_V3
        {
            int tk = g_tick[e];
            float phi = (float)(tk % G1_V3_PERIOD) / (float)G1_V3_PERIOD;
            float lp[2];
            lp[0] = phi;
            lp[1] = phi + 0.5f >= 1.0f ? phi - 0.5f : phi + 0.5f;
            const int fbody[2] = {G1_V3_LFOOT_BODY, G1_V3_RFOOT_BODY};
            for (int f = 0; f < 2; f++) {
                int stance = lp[f] < G1_V3_STANCE;
                int contact = g_footc[2 * e + f] > 0.5f;
                r += G1_V3_W_CONTACT * ((stance == contact) ? 1.0f : 0.0f);
                if (!contact) {
                    float dz = g_xpos[(size_t)e * S_X3 + 3 * fbody[f] + 2]
                             - G1_V3_FOOT_Z0;
                    r += G1_V3_W_SWING * dz * dz;
                }
            }
            float hp = 0.0f;
            const int hdof[4] = {1, 2, 7, 8};
            for (int h = 0; h < 4; h++) {
                float dq = qpos[7 + hdof[h]] - g1c_key_qpos[7 + hdof[h]];
                hp += dq * dq;
            }
            r += G1_V3_W_HIP * hp;
            float dzb = qpos[2] - G1_V3_BASE_Z0;
            r += G1_V3_W_BASE_HEIGHT * dzb * dzb;   // unitree base_height
        }
#endif
        float reward = r * ENV_CTRL_DT;
        int fell = (qpos[2] < ENV_TERM_HEIGHT) || (pg[2] > ENV_TERM_GRAVITY_Z) ||
                   !isfinite(qpos[2]);
        if (fell) reward += g1e_w_termination;
        int timeout = g_tick[e] >= g1e_max_ep_len;
        int done = fell || timeout;
        s_fell[warp] = fell;
        s_to[warp] = timeout;
        vec_rew[e] = reward;
#ifdef N3_TIMELIMIT_FIX
        // Pardo (2018) time-limit bootstrap: distinguish termination (fall) from
        // truncation (timeout). 1.0 => true terminal (fall): value bootstrap is
        // zeroed. 2.0 => truncation (timeout): the advantage must STILL bootstrap
        // V(s_timeout) instead of zeroing it, while cutting the GAE trace. The env
        // resets the physics on either (internal `done` at line ~358); this only
        // changes the terminal signal the trainer reads. REQUIRES the matching
        // decode in src/pufferlib.cu (compile BOTH with -DN3_TIMELIMIT_FIX).
        vec_term[e] = fell ? 1.0f : (timeout ? 2.0f : 0.0f);
#else
        vec_term[e] = done ? 1.0f : 0.0f;
#endif
        // live episode accumulators (c_step bookkeeping)
        eplog[0] += reward;
        eplog[1] += 1.0f;
        eplog[2] += track_lin;
        eplog[3] += sqrtf(lin_err2);
        if (done) {  // add_log
            float len = eplog[1];
            eplog[4] += eplog[0];
            eplog[5] += len;
            eplog[6] += len > 0.0f ? eplog[2] / len : 0.0f;
            eplog[7] += len > 0.0f ? eplog[3] / len : 0.0f;
            eplog[8] += fell ? 1.0f : 0.0f;
            eplog[9] += 1.0f;
        }
    }
    __syncwarp();
    int done = s_fell[warp] || s_to[warp];

    if (done) {
        env_reset_g(e, lane, g_rng, g_qpos, g_qvel, g_ws, g_prev, g_cmd, g_tick,
                    g_eplog, g_fric, g_mass);
    } else if (lane == 0 && g_tick[e] % ENV_CMD_RESAMPLE == 0) {
        unsigned int rng = g_rng[e];
        if (urand_01(&rng) < 0.1f) {
            g_cmd[3*e] = g_cmd[3*e+1] = g_cmd[3*e+2] = 0.0f;
        } else {
            g_cmd[3*e+0] = g1e_cmd_scale * 1.0f * urand_pm1(&rng);
            g_cmd[3*e+1] = g1e_cmd_scale * 0.6f * urand_pm1(&rng);
            g_cmd[3*e+2] = g1e_cmd_scale * 1.0f * urand_pm1(&rng);
        }
        g_rng[e] = rng;
    }
    __syncwarp();
    // DR external push: periodically set base xy lin-vel to U[+-push_vel] (perturbs
    // the NEXT step; the policy feels it via the following obs). Gated -> no write and
    // no RNG draw when DR off, so the baseline is bit-identical.
    if (!done && lane == 0 && g1e_dr_enable && g1e_push_interval > 0 &&
        g1e_dr_scale > 0.0f && g_tick[e] % g1e_push_interval == 0) {
        unsigned int rng = g_rng[e];
        float pv = g1e_dr_scale * g1e_push_vel;   // curriculum-ramped push magnitude
        g_qvel[(size_t)e * S_NV + 0] = pv * urand_pm1(&rng);
        g_qvel[(size_t)e * S_NV + 1] = pv * urand_pm1(&rng);
        g_rng[e] = rng;
    }
    __syncwarp();
    write_obs(e, lane, g_qpos, g_qvel, g_prev, g_cmd, g_tick, vec_obs + (size_t)e * ENV_OBS);
}

// ---------------------------------------------------------------------------
// host state + hooks (extern "C" for the C binding/vecenv)
// ---------------------------------------------------------------------------
typedef struct {
    float perf, score, episode_return, episode_length, vel_err, falls, n;
} G1GpuLog;  // mirrors the binding's Log layout exactly

static int g_total = 0;
// SoA state (full batch, absolute indexing)
static float *p_qpos, *p_qvel, *p_ctrl, *p_xpos, *p_xquat, *p_com, *p_cinert,
    *p_cdof, *p_qM, *p_qLD, *p_qLDiagInv, *p_qfs, *p_qas, *p_af;
static float *p_ws, *p_qaccF, *p_Ma, *p_qfc, *p_search, *p_Mv;
static float *p_jaref, *p_force, *p_jv, *p_rpos, *p_D, *p_R, *p_aref, *p_rowsign;
static float *p_scal, *p_H, *p_cJ, *p_condist;
static int* p_hvalid;
static float* p_footc;  // per-env {L,R} foot contact flags (k5 -> k_epi)  // H-memo cache flags; nullptr = memo off (large batch)
static int *p_ncon, *p_nefc, *p_rowtype, *p_rowdof, *p_rowstash, *p_state;
static float *p_act, *p_prev, *p_cmd, *p_eplog;
static float *p_fric, *p_mass;   // per-env DR friction & mass scale (1.0 = baseline)
static int* p_tick;
static unsigned int* p_rng;
static float* h_eplog = NULL;

#define G1GPU_MALLOC(p, n) CUDA_CHECK(cudaMalloc(&(p), (size_t)(n) * 4))

extern "C" void my_gpu_config(float action_scale, float w_track_lin,
        float w_track_ang, float w_lin_vel_z, float w_ang_vel_xy,
        float w_orientation, float w_torque, float w_action_rate,
        float w_alive, float w_termination, int max_episode_len) {
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_action_scale, &action_scale, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_track_lin, &w_track_lin, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_track_ang, &w_track_ang, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_lin_vel_z, &w_lin_vel_z, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_ang_vel_xy, &w_ang_vel_xy, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_orientation, &w_orientation, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_torque, &w_torque, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_action_rate, &w_action_rate, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_alive, &w_alive, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_w_termination, &w_termination, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_max_ep_len, &max_episode_len, 4));
}

// Domain-randomization config (Phase 1). dr_enable=0 -> full bit-exact baseline.
// obs-noise/push/init-basevel are ACTIVE; friction/mass ranges are accepted but
// STAGED (not yet read by kernels) and warned about so they're never a silent no-op.
extern "C" void my_gpu_config_dr(int dr_enable,
        float noise_angvel, float noise_gravity, float noise_dofpos, float noise_dofvel,
        int push_interval, float push_vel, float init_basevel,
        float fric_lo, float fric_hi, float mass_lo, float mass_hi) {
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_dr_enable, &dr_enable, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_noise_angvel, &noise_angvel, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_noise_gravity, &noise_gravity, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_noise_dofpos, &noise_dofpos, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_noise_dofvel, &noise_dofvel, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_push_interval, &push_interval, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_push_vel, &push_vel, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_init_basevel, &init_basevel, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_fric_lo, &fric_lo, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_fric_hi, &fric_hi, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_mass_lo, &mass_lo, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1e_mass_hi, &mass_hi, 4));
    if (!dr_enable) return;
    printf("g1gpu DR: ON  obs-noise[ang %.3f grav %.3f dofpos %.3f dofvel %.3f]  "
           "push[every %d @ %.2f m/s]  init_basevel %.2f\n",
           noise_angvel, noise_gravity, noise_dofpos, noise_dofvel,
           push_interval, push_vel, init_basevel);
    if (fric_lo != 1.0f || fric_hi != 1.0f || mass_lo != 1.0f || mass_hi != 1.0f)
        printf("g1gpu DR: friction[%.2f,%.2f] mass[%.2f,%.2f] (per-env scale, resampled each reset)\n",
               fric_lo, fric_hi, mass_lo, mass_hi);
}

extern "C" void my_gpu_init(int total_agents, unsigned int seed) {
    g_total = total_agents;
    int n = total_agents;
    G1GPU_MALLOC(p_qpos, (size_t)n * S_NQ);
    G1GPU_MALLOC(p_qvel, (size_t)n * S_NV);
    G1GPU_MALLOC(p_ctrl, (size_t)n * S_NU);
    G1GPU_MALLOC(p_xpos, (size_t)n * S_X3);
    G1GPU_MALLOC(p_xquat, (size_t)n * S_X4);
    G1GPU_MALLOC(p_com, (size_t)n * 3);
    G1GPU_MALLOC(p_cinert, (size_t)n * S_CI);
    G1GPU_MALLOC(p_cdof, (size_t)n * S_CD);
    G1GPU_MALLOC(p_qM, (size_t)n * G1_NM);
    G1GPU_MALLOC(p_qLD, (size_t)n * G1_NM);
    G1GPU_MALLOC(p_qLDiagInv, (size_t)n * S_NV);
    G1GPU_MALLOC(p_qfs, (size_t)n * S_NV);
    G1GPU_MALLOC(p_qas, (size_t)n * S_NV);
    G1GPU_MALLOC(p_af, (size_t)n * S_NU);
    G1GPU_MALLOC(p_ws, (size_t)n * S_NV);
    G1GPU_MALLOC(p_qaccF, (size_t)n * S_NV);
    G1GPU_MALLOC(p_Ma, (size_t)n * S_NV);
    G1GPU_MALLOC(p_qfc, (size_t)n * S_NV);
    G1GPU_MALLOC(p_search, (size_t)n * S_NV);
    G1GPU_MALLOC(p_Mv, (size_t)n * S_NV);
    G1GPU_MALLOC(p_jaref, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_force, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_jv, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_rpos, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_D, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_R, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_aref, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_rowsign, (size_t)n * NEFC_MAX);
    G1GPU_MALLOC(p_scal, (size_t)n * NSCAL);
    G1GPU_MALLOC(p_H, (size_t)n * G1_TRI);  // packed lower-tri factor
    // H-memoization wins when the cached factors stay L2-resident; above
    // ~16k envs the k10 active-set read-back costs more than skipped
    // rebuilds save (measured on GB202). Both paths bit-exact.
    CUDA_CHECK(cudaMalloc(&p_footc, (size_t)n * 2 * 4));
    CUDA_CHECK(cudaMemset(p_footc, 0, (size_t)n * 2 * 4));
    if (n <= 16384) {
        CUDA_CHECK(cudaMalloc(&p_hvalid, (size_t)n * 4));
        CUDA_CHECK(cudaMemset(p_hvalid, 0, (size_t)n * 4));
    } else {
        p_hvalid = nullptr;
    }
    G1GPU_MALLOC(p_cJ, (size_t)n * NCROW_MAX * S_NV);
    G1GPU_MALLOC(p_condist, (size_t)n * NCON_MAX);
    CUDA_CHECK(cudaMalloc(&p_ncon, (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&p_nefc, (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&p_rowtype, (size_t)n * NEFC_MAX * 4));
    CUDA_CHECK(cudaMalloc(&p_rowdof, (size_t)n * NEFC_MAX * 4));
    CUDA_CHECK(cudaMalloc(&p_rowstash, (size_t)n * NEFC_MAX * 4));
    CUDA_CHECK(cudaMalloc(&p_state, (size_t)n * NEFC_MAX * 4));
    G1GPU_MALLOC(p_act, (size_t)n * S_NU);
    G1GPU_MALLOC(p_prev, (size_t)n * S_NU);
    G1GPU_MALLOC(p_cmd, (size_t)n * 3);
    G1GPU_MALLOC(p_eplog, (size_t)n * EPLOG_N);
    G1GPU_MALLOC(p_fric, (size_t)n);   // set to 1.0 by the reset below (env_reset_g)
    G1GPU_MALLOC(p_mass, (size_t)n);
    CUDA_CHECK(cudaMalloc(&p_tick, (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&p_rng, (size_t)n * 4));
    h_eplog = (float*)malloc((size_t)n * EPLOG_N * 4);

    k_init_consts<<<1, 256>>>();
    int blocks = (n + SWARPS - 1) / SWARPS;
    k_reset_all<<<blocks, 32 * SWARPS>>>(n, seed, p_rng, p_qpos, p_qvel, p_ws,
                                         p_prev, p_cmd, p_tick, p_eplog, p_fric, p_mass);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("g1gpu: initialized %d GPU-native envs (no libmujoco)\n", n);
}

extern "C" void my_gpu_reset(void* vec_gpu_obs) {
    int n = g_total;
    int blocks = (n + SWARPS - 1) / SWARPS;
    k_reset_all<<<blocks, 32 * SWARPS>>>(n, 12345u, p_rng, p_qpos, p_qvel, p_ws,
                                         p_prev, p_cmd, p_tick, p_eplog, p_fric, p_mass);
    k_obs_all<<<blocks, 32 * SWARPS>>>(n, p_qpos, p_qvel, p_prev, p_cmd,
                                       p_tick, (float*)vec_gpu_obs);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

extern "C" void my_gpu_step_range(void* stream_v, int start, int count,
        const float* vec_actions, void* vec_obs, float* vec_rew, float* vec_term) {
    cudaStream_t st = (cudaStream_t)stream_v;
    int n = count;
    int blocks = (n + SWARPS - 1) / SWARPS;
    dim3 tpb(32 * SWARPS);
#if defined(K_NEWTON_FUSE) && !defined(SPARSE_SOLVER)
    int nf_blocks = (n + SF_WARPS - 1) / SF_WARPS;   // fused Newton kernel: SF_WARPS/block
    dim3 nf_tpb(32 * SF_WARPS);
#endif
    size_t s = (size_t)start;

#ifdef G1_CURRICULUM
    {   // command curriculum: ramp the velocity-command range from CS_START to
        // 1.0 over CS_RAMP samples, so early training learns the gait on slow
        // commands and widens to the full distribution as it gets competent.
        static long g_curric_steps = 0;
        const float  CS_START = 0.4f;
        const double CS_RAMP  = 40.0e6;   // samples to reach the full range
        g_curric_steps += n;
        double prog = (double)g_curric_steps / CS_RAMP;
        float cs = CS_START + (1.0f - CS_START) * (float)(prog < 1.0 ? prog : 1.0);
        cudaMemcpyToSymbolAsync(g1e_cmd_scale, &cs, sizeof(float), 0,
                                cudaMemcpyHostToDevice, st);
    }
#endif
#ifdef G1_DR_CURRICULUM
    {   // DR curriculum: hold DR at 0 for DR_WARMUP samples (the forward gait forms
        // CLEAN), then ramp dr_scale 0->1 over DR_RAMP samples. The env multiplies all
        // DR magnitudes by dr_scale, so DR fades in instead of shocking a from-scratch
        // (or warm-started) policy into the degenerate "survive, don't track" basin.
        static long g_drc_steps = 0;
        const double DR_WARMUP = 50.0e6;
        const double DR_RAMP   = 100.0e6;   // full DR by DR_WARMUP+DR_RAMP (~150M)
        g_drc_steps += n;
        double prog = ((double)g_drc_steps - DR_WARMUP) / DR_RAMP;
        float ds = (float)(prog < 0.0 ? 0.0 : (prog > 1.0 ? 1.0 : prog));
        cudaMemcpyToSymbolAsync(g1e_dr_scale, &ds, sizeof(float), 0,
                                cudaMemcpyHostToDevice, st);
    }
#endif

    // offset views: every kernel indexes envs 0..count-1 relative to start
    float* qpos = p_qpos + s * S_NQ;
    float* qvel = p_qvel + s * S_NV;
    float* ctrl = p_ctrl + s * S_NU;
    float* xpos = p_xpos + s * S_X3;
    float* xquat = p_xquat + s * S_X4;
    float* com = p_com + s * 3;
    float* cinert = p_cinert + s * S_CI;
    float* cdof = p_cdof + s * S_CD;
    float* qM = p_qM + s * G1_NM;
    float* qLD = p_qLD + s * G1_NM;
    float* qLDiagInv = p_qLDiagInv + s * S_NV;
    float* qfs = p_qfs + s * S_NV;
    float* qas = p_qas + s * S_NV;
    float* af = p_af + s * S_NU;
    float* ws = p_ws + s * S_NV;
    float* qaccF = p_qaccF + s * S_NV;
    float* Ma = p_Ma + s * S_NV;
    float* qfc = p_qfc + s * S_NV;
    float* search = p_search + s * S_NV;
    float* Mv = p_Mv + s * S_NV;
    float* jaref = p_jaref + s * NEFC_MAX;
    float* force = p_force + s * NEFC_MAX;
    float* jv = p_jv + s * NEFC_MAX;
    float* rpos = p_rpos + s * NEFC_MAX;
    float* D = p_D + s * NEFC_MAX;
    float* R = p_R + s * NEFC_MAX;
    float* aref = p_aref + s * NEFC_MAX;
    float* rowsign = p_rowsign + s * NEFC_MAX;
    float* scal = p_scal + s * NSCAL;
    float* H = p_H + (size_t)s * G1_TRI;
    int* hvalid = p_hvalid ? p_hvalid + s : nullptr;
    float* footc = p_footc + (size_t)s * 2;
    float* cJ = p_cJ + s * NCROW_MAX * S_NV;
    float* condist = p_condist + s * NCON_MAX;
    int* ncon = p_ncon + s;
    int* nefc = p_nefc + s;
    int* rowtype = p_rowtype + s * NEFC_MAX;
    int* rowdof = p_rowdof + s * NEFC_MAX;
    int* rowstash = p_rowstash + s * NEFC_MAX;
    int* state = p_state + s * NEFC_MAX;
    float* act = p_act + s * S_NU;
    float* prev = p_prev + s * S_NU;
    float* cmd = p_cmd + s * 3;
    float* eplog = p_eplog + s * EPLOG_N;
    float* fric = p_fric + s;
    float* mass = p_mass + s;
    int* tick = p_tick + s;
    unsigned int* rng = p_rng + s;
    const float* va = vec_actions;             // caller passes offset pointers
    float* vo = (float*)vec_obs;
    float* vr = vec_rew;
    float* vt = vec_term;

    // The full control step as a single callable: k_act + decimation x
    // [k1..k4 (+k3b)] + k_epi. ~70 async kernel launches + D2D memcpy on stream
    // st, fixed sequence (decimation/SOL_ITER compile-time), per-buffer-stable
    // pointers -> safe to CUDA-graph-capture.
    auto step_body = [&]() {
    k_act<<<blocks, tpb, 0, st>>>(n, va, act, ctrl);
#ifndef SKIP_PHYSICS   // ceiling probe: skip the entire physics decimation loop
                       // (k1..k4). Only k_act + k_epi run -> env GPU work ~0.
                       // If training SPS is unchanged vs baseline, the physics is
                       // NOT on the critical path (learner/inference-bound); if it
                       // jumps, physics IS the wall. Garbage dynamics, SPS-only.
    for (int k = 0; k < ENV_DECIMATION; k++) {
        k1_fk_compos<<<blocks, tpb, 0, st>>>(n, qpos, xpos, xquat, com, cinert, cdof, mass);
        k2_crb_factor<<<blocks, tpb, 0, st>>>(n, cinert, cdof, qM, qLD, qLDiagInv);
        k3_rne_act_solve<<<blocks, tpb, 0, st>>>(n, qpos, qvel, ctrl, S_NU, cinert,
                                                 cdof, qLD, qLDiagInv, qfs, qas, af);
#ifdef K3_SPLIT
        // k3b: thread-per-env LDL solve (recovers the 31 idle lanes of k3's old
        // lane-0 solve; bit-identical). Thread-per-env grid, not warp-per-env.
        k3b_ldlsolve<<<(n + 255) / 256, 256, 0, st>>>(n, qfs, qLD, qLDiagInv, qas);
#endif
        k5_assemble<<<blocks, tpb, 0, st>>>(n, qpos, qvel, xpos, xquat, com, ncon,
                                            nefc, condist, rowtype, rowdof, rowsign,
                                            rowstash, rpos, D, R, aref, cJ, cdof,
                                            footc, fric);
#ifndef SKIP_SOLVER   // ceiling probe: skip the whole Newton solver (k6 + SOL_ITER
                      // x [k7..k10]) and integrate the unconstrained smooth accel
                      // (qas) instead of the constrained qaccF. Garbage contacts,
                      // SPS-only. Measures the solver's TRUE end-to-end share: if SPS
                      // rises ~proportionally, a sparse-factor solver would translate.
        k6_wsinit<<<blocks, tpb, 0, st>>>(n, qM, qfs, qas, ws, nefc, rowtype, rowdof,
                                          rowsign, D, R, aref, cJ, qaccF, Ma, jaref,
                                          force, state, qfc, scal, hvalid);
#if defined(K_NEWTON_FUSE) && !defined(SPARSE_SOLVER)   // full Newton loop in one kernel,
        // H + search/Mv/jv/jaref kept in smem across iters -> no solver round-trips,
        // 3 fewer launches/iter, and in-register H-reuse (active-set-stable skip).
        k_newton_iter<<<nf_blocks, nf_tpb, 0, st>>>(n, qM, qfs, qas, nefc, rowtype,
                                                    rowdof, rowsign, D, R, cJ, qaccF, Ma,
                                                    jaref, force, state, qfc, scal);
#else
        for (int it = 0; it < SOL_ITER; it++) {
#if defined(K78_FUSE) && !defined(SPARSE_SOLVER)   // fused build+factor+solve, H in smem
            k78_solve<<<blocks, tpb, 0, st>>>(n, qM, nefc, rowtype, rowdof, D, state, cJ,
                                              qfs, Ma, qfc, rowsign, search, Mv, jv, scal);
#else
            k7_hessian<<<blocks, tpb, 0, st>>>(n, qM, nefc, rowtype, rowdof, D,
                                               state, cJ, scal, H, hvalid);
            k8_solvesearch<<<blocks, tpb, 0, st>>>(n, qM, qfs, Ma, qfc, H, nefc,
                                                   rowtype, rowdof, rowsign, cJ,
                                                   search, Mv, jv, scal);
#endif
            k9_linesearch<<<blocks, tpb, 0, st>>>(n, nefc, rowtype, rowdof, D, R,
                                                  jaref, jv, scal);
            k10_update<<<blocks, tpb, 0, st>>>(n, qfs, qas, nefc, rowtype, rowdof,
                                               rowsign, D, R, cJ, search, Mv, jv,
                                               qaccF, Ma, jaref, force, state, qfc,
                                               scal, hvalid);
        }
#endif
        k4_euler<<<blocks, tpb, 0, st>>>(n, qpos, qvel, qaccF);
        CUDA_CHECK(cudaMemcpyAsync(ws, qaccF, (size_t)n * S_NV * 4,
                                   cudaMemcpyDeviceToDevice, st));
#else
        k4_euler<<<blocks, tpb, 0, st>>>(n, qpos, qvel, qas);
        CUDA_CHECK(cudaMemcpyAsync(ws, qas, (size_t)n * S_NV * 4,
                                   cudaMemcpyDeviceToDevice, st));
#endif
    }
#endif  // SKIP_PHYSICS
    k_epi<<<blocks, tpb, 0, st>>>(n, qpos, qvel, ws, xpos, footc, af, act, prev,
                                  cmd, tick, rng, eplog, fric, mass, vo, vr, vt);
    };

#ifdef GRAPH_ENV_STEP
    // Capture the whole control-step sequence into a CUDA graph ONCE per worker
    // thread (each thread owns one fixed buffer/stream/start -> thread_local cache,
    // no locking) and replay it every step. Collapses ~70 eager launch+gap
    // latencies into ONE cudaGraphLaunch. Targets the env-step WALL, which k3b
    // showed is gap-bound (gpu_busy -12% moved the wall ~0%), not gpu_busy-bound.
    // Bit-exact: replays identical kernels on identical device pointers.
    static thread_local cudaGraphExec_t g_exec = nullptr;
    static thread_local int g_key = -1;
    if (g_exec == nullptr || g_key != start) {
        if (g_exec) { cudaGraphExecDestroy(g_exec); g_exec = nullptr; }
        CUDA_CHECK(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal));
        step_body();
        cudaGraph_t graph;
        CUDA_CHECK(cudaStreamEndCapture(st, &graph));
        CUDA_CHECK(cudaGraphInstantiate(&g_exec, graph, 0));
        cudaGraphDestroy(graph);
        g_key = start;
    }
    CUDA_CHECK(cudaGraphLaunch(g_exec, st));
#else
    step_body();
#endif
}

extern "C" float my_gpu_log_into(void* log_out) {
    G1GpuLog* out = (G1GpuLog*)log_out;
    memset(out, 0, sizeof(G1GpuLog));
    int n = g_total;
    CUDA_CHECK(cudaMemcpy(h_eplog, p_eplog, (size_t)n * EPLOG_N * 4,
                          cudaMemcpyDeviceToHost));
    for (int e = 0; e < n; e++) {
        float* l = h_eplog + (size_t)e * EPLOG_N;
        out->episode_return += l[4];
        out->score += l[4];
        out->episode_length += l[5];
        out->perf += l[6];
        out->vel_err += l[7];
        out->falls += l[8];
        out->n += l[9];
        // zero the completed-episode totals (live accumulators stay)
        l[4] = l[5] = l[6] = l[7] = l[8] = l[9] = 0.0f;
    }
    CUDA_CHECK(cudaMemcpy(p_eplog, h_eplog, (size_t)n * EPLOG_N * 4,
                          cudaMemcpyHostToDevice));
    return out->n;
}

extern "C" void my_gpu_close(void) { /* process teardown frees the device */ }
