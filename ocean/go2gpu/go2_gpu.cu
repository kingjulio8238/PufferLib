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

#include "g1phys/robot_topology.cuh"   // -DROBOT_TOPO_H='"go2_topology.cuh"' selects Go2
#include "g1phys/common.cuh"
#include "g1phys/traj_format.h"

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("G1GPU CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); } } while (0)

#define WARPS_PER_BLOCK 2
#include "g1phys/g1_step.cuh"
#include "g1phys/g1_full_step.cuh"
#include "g1phys/g1_staged_kernels.cuh"
#include "g1phys/g1_solver_consts.cuh"

#ifndef ENV_DECIMATION
#define ENV_DECIMATION 10
#endif
#define ENV_CTRL_DT 0.02f               // dt 0.005 x decimation 4 = 50 Hz control
#define ENV_RESET_NOISE 0.05f
// Go2 sits at trunk z=0.27; at kp20 it sags to ~0.18 just holding the default
// pose, so terminate only on a real collapse (well below the sag) or a flipped
// base. The +alive reward pushes the policy to stand tall.
#define ENV_TERM_HEIGHT 0.15f
#define ENV_TERM_GRAVITY_Z (-0.6f)
#define ENV_CMD_RESAMPLE 500
// obs layout: [0:3] base linear velocity (base frame, x2.0), [3:6] base ang vel,
// [6:9] projected gravity, [9:12] velocity command, [12:12+nu] joint pos-default,
// [..] joint vel, [..] previous action, then PRIVILEGED state appended (clean,
// no noise): [.. +0:4] foot contacts (FL,FR,RL,RR), [.. +4] trunk height. The
// native trainer has no separate critic obs, so we feed this richer state to
// the shared net to make value estimation tractable. Go2: 12 + 3*12 + 5 = 53.
#define ENV_PRIV 5
#define ENV_OBS (12 + 3 * S_NU + ENV_PRIV)

// legged_gym Go2 gait/safety reward terms (fixed reference values, not swept):
//   feet_air_time +1.0 (reward foot airborne >0.5s at touchdown, cmd-gated),
//   collision -1.0 (non-foot sphere on the floor), dof_pos_limits -10.0 (soft
//   0.9 of range), dof_acc -2.5e-7. Foot/collision flags: g_footc stride 5
//   [FL,FR,RL,RR,n_collisions] from k5_assemble.
// Champion (go2_rl_gym) reward terms. The standing bootstrap is correct_base_height
// (command-independent), NOT a curriculum or pose crutch; no feet_air/gait term.
#define W_FEET_AIR    0.0f          // champion has no feet_air_time term
#define W_COLLISION  (-1.0f)
#define W_DOF_LIMIT  (-2.0f)        // champion -2.0 (was a too-harsh -10)
#define W_DOF_ACC    (-2.5e-7f)
#define AIR_TARGET    0.5f
#define SOFT_LIMIT    0.9f
#define N_FEET        4
#define W_BASE_HEIGHT (-20.0f)      // correct_base_height: -(z-target)^2; the
#define BASE_HEIGHT_TARGET 0.33f    //   command-independent stand-up gradient
#define W_HIP_DEFAULT (-0.05f)      // hip_to_default: mild hip-neutral anchor
#define W_ACTION_SMOOTH (-0.01f)    // action_smoothness: 2nd-order action diff
#define TRACK_SIGMA   0.5f          // velocity-tracking exp width (was 0.25 — too
                                    //   narrow: no move-gradient at full commands)
#define W_VEL_PROJ    2.0f          // linear (non-saturating) reward for base speed
                                    //   along the command dir — breaks stand-still
// domain randomization (legged_gym subset): observation noise + random pushes.
// (friction/mass randomization needs per-env contact params = an engine change;
// deferred and documented.) Noise levels are on the SCALED obs entries.
#define OBS_NOISE 1                 // 0 disables
#define NZ_LINVEL   0.20f
#define NZ_ANGVEL   0.05f
#define NZ_GRAVITY  0.05f
#define NZ_JPOS     0.01f
#define NZ_JVEL     0.075f
#define PUSH_INTERVAL 200           // control steps (~4 s @ 50 Hz, champion)
#define PUSH_VEL      0.4f          // m/s, applied to base xy velocity

// runtime-configurable env parameters (the Protein sweep moves these)
__device__ float g1e_action_scale, g1e_w_track_lin, g1e_w_track_ang;
__device__ float g1e_w_lin_vel_z, g1e_w_ang_vel_xy, g1e_w_orientation;
__device__ float g1e_w_torque, g1e_w_action_rate, g1e_w_alive, g1e_w_termination;
__device__ int g1e_max_ep_len;
// command ranges, raised by a SELF-PACED curriculum (host side). Start small so
// the robot can satisfy commands by barely moving (tracking gradient is alive),
// then graduate: once standing no longer satisfies the larger command, it is
// forced to actually locomote. vx/vy ramp 0.25 -> 1.0; wz stays full.
__device__ float g2e_cmd_vx = 0.25f, g2e_cmd_vy = 0.25f, g2e_cmd_wz = 1.0f;

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
                            float* g_eplog) {
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
            g_cmd[3 * e + 0] = g2e_cmd_vx * urand_pm1(&rng);
            g_cmd[3 * e + 1] = g2e_cmd_vy * urand_pm1(&rng);
            g_cmd[3 * e + 2] = g2e_cmd_wz * urand_pm1(&rng);
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
                            float* g_eplog) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    if (lane == 0) {
        g_rng[e] = seed ^ (0x9e3779b9u * (unsigned int)(e + 1));
        for (int k = 0; k < EPLOG_N; k++) g_eplog[(size_t)e * EPLOG_N + k] = 0.0f;
    }
    __syncwarp();
    env_reset_g(e, lane, g_rng, g_qpos, g_qvel, g_ws, g_prev, g_cmd, g_tick, g_eplog);
}

// write the 96-float obs of the current state (used post-reset and in K_epi)
__device__ void write_obs(int e, int lane, const float* g_qpos, const float* g_qvel,
                          const float* g_prev, const float* g_cmd,
                          const int* g_tick_obs, float* obs,
                          unsigned int* g_rng_noise = nullptr,
                          const float* g_footc = nullptr) {
    if (lane == 0) {
        const float* q = g_qpos + (size_t)e * S_NQ;
        const float* v = g_qvel + (size_t)e * S_NV;
        float vw[3] = {v[0], v[1], v[2]}, vb[3];   // base-frame linear velocity
        world_to_base(q + 3, vw, vb);
        float gw[3] = {0.0f, 0.0f, -1.0f}, gb[3];
        world_to_base(q + 3, gw, gb);
        obs[0] = 2.0f * vb[0]; obs[1] = 2.0f * vb[1]; obs[2] = 2.0f * vb[2];
        obs[3] = 0.25f * v[3]; obs[4] = 0.25f * v[4]; obs[5] = 0.25f * v[5];
        obs[6] = gb[0]; obs[7] = gb[1]; obs[8] = gb[2];
        obs[9] = g_cmd[3*e]; obs[10] = g_cmd[3*e+1]; obs[11] = g_cmd[3*e+2];
    }
    for (int j = lane; j < S_NU; j += 32) {
        obs[12 + j]               = g_qpos[(size_t)e * S_NQ + 7 + j] - g1c_key_qpos[7 + j];
        obs[(12 + S_NU) + j]      = 0.05f * g_qvel[(size_t)e * S_NV + 6 + j];
        obs[(12 + 2 * S_NU) + j]  = g_prev[(size_t)e * S_NU + j];
    }
#if OBS_NOISE
    // additive observation noise (legged_gym add_noise) — applied serially on
    // lane 0 after all entries are written (single per-env RNG, no race).
    __syncwarp();
    if (g_rng_noise && lane == 0) {
        unsigned int rng = g_rng_noise[e];
        obs[0] += NZ_LINVEL*urand_pm1(&rng); obs[1] += NZ_LINVEL*urand_pm1(&rng); obs[2] += NZ_LINVEL*urand_pm1(&rng);
        obs[3] += NZ_ANGVEL*urand_pm1(&rng); obs[4] += NZ_ANGVEL*urand_pm1(&rng); obs[5] += NZ_ANGVEL*urand_pm1(&rng);
        obs[6] += NZ_GRAVITY*urand_pm1(&rng); obs[7] += NZ_GRAVITY*urand_pm1(&rng); obs[8] += NZ_GRAVITY*urand_pm1(&rng);
        for (int j = 0; j < S_NU; j++) obs[12 + j]          += NZ_JPOS*urand_pm1(&rng);
        for (int j = 0; j < S_NU; j++) obs[12 + S_NU + j]   += NZ_JVEL*urand_pm1(&rng);
        g_rng_noise[e] = rng;
    }
#endif
    // privileged state (clean, no noise): foot contacts + trunk height
    if (lane == 0) {
        int pb = 12 + 3 * S_NU;
        obs[pb+0] = g_footc ? g_footc[5*e+0] : 0.0f;
        obs[pb+1] = g_footc ? g_footc[5*e+1] : 0.0f;
        obs[pb+2] = g_footc ? g_footc[5*e+2] : 0.0f;
        obs[pb+3] = g_footc ? g_footc[5*e+3] : 0.0f;
        obs[pb+4] = g_qpos[(size_t)e * S_NQ + 2];   // trunk height z
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
                      float* __restrict__ vec_obs, float* __restrict__ vec_rew,
                      float* __restrict__ vec_term,
                      float* __restrict__ g_air, float* __restrict__ g_dofvel_prev,
                      float* __restrict__ g_prev2) {
    __shared__ int s_fell[SWARPS], s_to[SWARPS];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    const float* qpos = g_qpos + (size_t)e * S_NQ;
    const float* qvel = g_qvel + (size_t)e * S_NV;
    float* eplog = g_eplog + (size_t)e * EPLOG_N;

    if (lane == 0) g_tick[e] += 1;
    __syncwarp();

    float t2 = 0.0f, ar2 = 0.0f, sm2 = 0.0f;
    for (int j = lane; j < S_NU; j += 32) {
        float tau = g_af[(size_t)e * S_NU + j];
        t2 += tau * tau;
        float a = g_act[(size_t)e * S_NU + j];
        float prev = g_prev[(size_t)e * S_NU + j];
        float prev2 = g_prev2[(size_t)e * S_NU + j];
        float da = a - prev;
        ar2 += da * da;
        float ds = a - 2.0f * prev + prev2;   // 2nd-order (action_smoothness)
        sm2 += ds * ds;
        g_prev2[(size_t)e * S_NU + j] = prev;
        g_prev[(size_t)e * S_NU + j] = a;
    }
    for (int o = 16; o > 0; o >>= 1) {
        t2 += __shfl_xor_sync(0xffffffff, t2, o);
        ar2 += __shfl_xor_sync(0xffffffff, ar2, o);
        sm2 += __shfl_xor_sync(0xffffffff, sm2, o);
    }
    if (lane == 0) {
        float vb[3], pg[3];
        float gw[3] = {0.0f, 0.0f, -1.0f};
        world_to_base(qpos + 3, qvel, vb);
        world_to_base(qpos + 3, gw, pg);
        float wx = qvel[3], wy = qvel[4], wz = qvel[5];
        float ex = g_cmd[3*e] - vb[0], ey = g_cmd[3*e+1] - vb[1];
        float lin_err2 = ex * ex + ey * ey;
        // WIDER tracking sigma (0.25 -> 0.5): at full +-1 commands the original
        // sigma puts the velocity error on the flat tail of exp(), giving ~no
        // gradient to start moving (-> stand-still optimum). Widening restores a
        // real pull toward the commanded velocity without a curriculum.
        float track_lin = expf(-lin_err2 / TRACK_SIGMA);
        float eyaw = g_cmd[3*e+2] - wz;
        float track_ang = expf(-eyaw * eyaw / TRACK_SIGMA);
        // velocity-projection: base speed along the command direction, capped at
        // the commanded speed. LINEAR (non-saturating) so there is always a pull
        // to move faster toward the command — standing (vproj=0) earns nothing
        // here, moving earns up to W_VEL_PROJ*|cmd|. This is what the saturating
        // exp-tracking cannot do at the stand-still optimum.
        float cmdmag = sqrtf(g_cmd[3*e]*g_cmd[3*e] + g_cmd[3*e+1]*g_cmd[3*e+1]);
        float vproj = 0.0f;
        if (cmdmag > 0.05f) {
            vproj = (vb[0]*g_cmd[3*e] + vb[1]*g_cmd[3*e+1]) / cmdmag;
            vproj = fminf(vproj, cmdmag);   // no bonus beyond the commanded speed
        }
        float r = g1e_w_alive
                + g1e_w_track_lin * track_lin
                + g1e_w_track_ang * track_ang
                + W_VEL_PROJ * vproj
                + g1e_w_lin_vel_z * vb[2] * vb[2]
                + g1e_w_ang_vel_xy * (wx * wx + wy * wy)
                + g1e_w_orientation * (pg[0] * pg[0] + pg[1] * pg[1])
                + g1e_w_torque * t2
                + g1e_w_action_rate * ar2;
        // --- champion (go2_rl_gym) safety/regularization terms (lane 0) ---
        // correct_base_height: maintain the standing trunk height. THIS is the
        // command-independent stand-up gradient (no curriculum / no only_positive
        // needed): falling drops z -> penalty, standing tall -> 0.
        float zerr = qpos[2] - BASE_HEIGHT_TARGET;
        r += W_BASE_HEIGHT * zerr * zerr;
        r += W_ACTION_SMOOTH * sm2;                    // action_smoothness (2nd order)
        r += W_COLLISION * g_footc[5 * e + 4];         // non-foot sphere on the floor
        // dof_pos_limits (soft 0.9) + dof_acc + hip_to_default (mild hip anchor).
        float lim_pen = 0.0f, acc2 = 0.0f, hip_pen = 0.0f;
        for (int j = 0; j < S_NU; j++) {
            int jnt = j + 1;  // actuated joints are 1..S_NU (joint 0 = free base)
            float lo = g1c_jnt_range[2*jnt], hi = g1c_jnt_range[2*jnt+1];
            float mid = 0.5f*(lo+hi), half = 0.5f*(hi-lo)*SOFT_LIMIT;
            float q = qpos[7 + j];
            lim_pen += fmaxf(0.0f, (mid - half) - q) + fmaxf(0.0f, q - (mid + half));
            float dv = qvel[6 + j];
            float acc = (g_dofvel_prev[(size_t)e * S_NU + j] - dv) / ENV_CTRL_DT;
            acc2 += acc * acc;
            g_dofvel_prev[(size_t)e * S_NU + j] = dv;
            if (j % 3 == 0) hip_pen += fabsf(q - g1c_key_qpos[7 + j]);  // hip joints
        }
        r += W_DOF_LIMIT * lim_pen + W_DOF_ACC * acc2 + W_HIP_DEFAULT * hip_pen;
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
        }
#endif
        // NO only_positive clip (champion): penalties flow as live gradient.
        // Suicide-proof via PERMISSIVE termination — we terminate ONLY on a real
        // turn-over (base tilted past ~72 deg, pg[2] > -0.3) or non-finite state,
        // NOT on low height. So sprawling/lying-low is not a fast death-escape to
        // stop the negative reward; the robot must accrue penalties until it
        // stands (base_height + lin_vel_z drive that). Episode timeout bounds it.
        float reward = r * ENV_CTRL_DT;
        int fell = (pg[2] > -0.3f) || !isfinite(qpos[2]);
        if (fell) reward += g1e_w_termination;
        int timeout = g_tick[e] >= g1e_max_ep_len;
        int done = fell || timeout;
        s_fell[warp] = fell;
        s_to[warp] = timeout;
        vec_rew[e] = reward;
        vec_term[e] = done ? 1.0f : 0.0f;
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
                    g_eplog);
        for (int f = lane; f < N_FEET; f += 32) g_air[(size_t)e * N_FEET + f] = 0.0f;
        for (int j = lane; j < S_NU; j += 32) {
            g_dofvel_prev[(size_t)e * S_NU + j] = 0.0f;
            g_prev2[(size_t)e * S_NU + j] = 0.0f;
        }
    } else if (lane == 0 && g_tick[e] % ENV_CMD_RESAMPLE == 0) {
        unsigned int rng = g_rng[e];
        if (urand_01(&rng) < 0.1f) {
            g_cmd[3*e] = g_cmd[3*e+1] = g_cmd[3*e+2] = 0.0f;
        } else {
            g_cmd[3*e+0] = g2e_cmd_vx * urand_pm1(&rng);
            g_cmd[3*e+1] = g2e_cmd_vy * urand_pm1(&rng);
            g_cmd[3*e+2] = g2e_cmd_wz * urand_pm1(&rng);
        }
        g_rng[e] = rng;
    }
    // random push: kick the base xy velocity periodically (champion: 0.4 m/s, 4 s)
    if (lane == 0 && g_tick[e] > 0 && g_tick[e] % PUSH_INTERVAL == 0) {
        unsigned int rng = g_rng[e];
        g_qvel[(size_t)e * S_NV + 0] += PUSH_VEL * urand_pm1(&rng);
        g_qvel[(size_t)e * S_NV + 1] += PUSH_VEL * urand_pm1(&rng);
        g_rng[e] = rng;
    }
    __syncwarp();
    write_obs(e, lane, g_qpos, g_qvel, g_prev, g_cmd, g_tick,
              vec_obs + (size_t)e * ENV_OBS, g_rng, g_footc);
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
static float* p_footc;  // per-env [FL,FR,RL,RR,n_collisions] (stride 5, k5 -> k_epi)
static float* p_air;        // per-env per-foot air time (n*4) for feet_air_time
static float* p_dofvel_prev; // per-env last joint qvel (n*S_NU) for dof_acc
static float* p_prev2;       // per-env action two steps ago (n*S_NU) for action_smoothness
static int *p_ncon, *p_nefc, *p_rowtype, *p_rowdof, *p_rowstash, *p_state;
static float *p_act, *p_prev, *p_cmd, *p_eplog;
static int* p_tick;
static unsigned int* p_rng;
static float* h_eplog = NULL;
// optional render dump: env-0 qpos per control step -> GO2_RENDER_OUT file
static FILE* g_render_fp = NULL;
static int g_render_frames = 0, g_render_max = 1500;

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
    CUDA_CHECK(cudaMalloc(&p_footc, (size_t)n * 5 * 4));
    CUDA_CHECK(cudaMemset(p_footc, 0, (size_t)n * 5 * 4));
    CUDA_CHECK(cudaMalloc(&p_air, (size_t)n * N_FEET * 4));
    CUDA_CHECK(cudaMemset(p_air, 0, (size_t)n * N_FEET * 4));
    CUDA_CHECK(cudaMalloc(&p_dofvel_prev, (size_t)n * S_NU * 4));
    CUDA_CHECK(cudaMemset(p_dofvel_prev, 0, (size_t)n * S_NU * 4));
    CUDA_CHECK(cudaMalloc(&p_prev2, (size_t)n * S_NU * 4));
    CUDA_CHECK(cudaMemset(p_prev2, 0, (size_t)n * S_NU * 4));
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
    CUDA_CHECK(cudaMalloc(&p_tick, (size_t)n * 4));
    CUDA_CHECK(cudaMalloc(&p_rng, (size_t)n * 4));
    h_eplog = (float*)malloc((size_t)n * EPLOG_N * 4);

    const char* rpath = getenv("GO2_RENDER_OUT");
    if (rpath) {
        g_render_fp = fopen(rpath, "wb");
        if (g_render_fp) {
            int hdr[2] = {S_NQ, g_render_max};   // small header: nq, max frames
            fwrite(hdr, 4, 2, g_render_fp);
            printf("go2gpu: render dump -> %s (env 0 qpos, up to %d frames)\n",
                   rpath, g_render_max);
        }
    }

    k_init_consts<<<1, 256>>>();
    int blocks = (n + SWARPS - 1) / SWARPS;
    k_reset_all<<<blocks, 32 * SWARPS>>>(n, seed, p_rng, p_qpos, p_qvel, p_ws,
                                         p_prev, p_cmd, p_tick, p_eplog);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("g1gpu: initialized %d GPU-native envs (no libmujoco)\n", n);
}

extern "C" void my_gpu_reset(void* vec_gpu_obs) {
    int n = g_total;
    int blocks = (n + SWARPS - 1) / SWARPS;
    k_reset_all<<<blocks, 32 * SWARPS>>>(n, 12345u, p_rng, p_qpos, p_qvel, p_ws,
                                         p_prev, p_cmd, p_tick, p_eplog);
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
    size_t s = (size_t)start;

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
    float* footc = p_footc + (size_t)s * 5;
    float* air = p_air + (size_t)s * N_FEET;
    float* dofvel_prev = p_dofvel_prev + (size_t)s * S_NU;
    float* prev2 = p_prev2 + (size_t)s * S_NU;
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
    int* tick = p_tick + s;
    unsigned int* rng = p_rng + s;
    const float* va = vec_actions;             // caller passes offset pointers
    float* vo = (float*)vec_obs;
    float* vr = vec_rew;
    float* vt = vec_term;

    k_act<<<blocks, tpb, 0, st>>>(n, va, act, ctrl);
    for (int k = 0; k < ENV_DECIMATION; k++) {
        k1_fk_compos<<<blocks, tpb, 0, st>>>(n, qpos, xpos, xquat, com, cinert, cdof);
        k2_crb_factor<<<blocks, tpb, 0, st>>>(n, cinert, cdof, qM, qLD, qLDiagInv);
        k3_rne_act_solve<<<blocks, tpb, 0, st>>>(n, qpos, qvel, ctrl, S_NU, cinert,
                                                 cdof, qLD, qLDiagInv, qfs, qas, af);
        k5_assemble<<<blocks, tpb, 0, st>>>(n, qpos, qvel, xpos, xquat, com, ncon,
                                            nefc, condist, rowtype, rowdof, rowsign,
                                            rowstash, rpos, D, R, aref, cJ, cdof,
                                            footc);
        k6_wsinit<<<blocks, tpb, 0, st>>>(n, qM, qfs, qas, ws, nefc, rowtype, rowdof,
                                          rowsign, D, R, aref, cJ, qaccF, Ma, jaref,
                                          force, state, qfc, scal, hvalid);
        for (int it = 0; it < SOL_ITER; it++) {
            k7_hessian<<<blocks, tpb, 0, st>>>(n, qM, nefc, rowtype, rowdof, D,
                                               state, cJ, scal, H, hvalid);
            k8_solvesearch<<<blocks, tpb, 0, st>>>(n, qM, qfs, Ma, qfc, H, nefc,
                                                   rowtype, rowdof, rowsign, cJ,
                                                   search, Mv, jv, scal);
            k9_linesearch<<<blocks, tpb, 0, st>>>(n, nefc, rowtype, rowdof, D, R,
                                                  jaref, jv, scal);
            k10_update<<<blocks, tpb, 0, st>>>(n, qfs, qas, nefc, rowtype, rowdof,
                                               rowsign, D, R, cJ, search, Mv, jv,
                                               qaccF, Ma, jaref, force, state, qfc,
                                               scal, hvalid);
        }
        k4_euler<<<blocks, tpb, 0, st>>>(n, qpos, qvel, qaccF);
        CUDA_CHECK(cudaMemcpyAsync(ws, qaccF, (size_t)n * S_NV * 4,
                                   cudaMemcpyDeviceToDevice, st));
    }
    k_epi<<<blocks, tpb, 0, st>>>(n, qpos, qvel, ws, xpos, footc, af, act, prev,
                                  cmd, tick, rng, eplog, vo, vr, vt, air, dofvel_prev,
                                  prev2);
}

// render hook: called by c_render (binding) once per eval iteration, OUTSIDE
// the captured rollout graph -> a plain cudaMemcpy here is graph-safe. Dumps
// env-0 qpos to the GO2_RENDER_OUT file for offline MuJoCo rendering.
extern "C" void my_gpu_render(void) {
    if (!g_render_fp || g_render_frames >= g_render_max) return;
    // force env-0's command to a steady forward walk (0.8 m/s) so the render
    // shows locomotion, not whatever (possibly ~zero) command it would sample.
    float fcmd[3] = {0.8f, 0.0f, 0.0f};
    cudaMemcpy(p_cmd, fcmd, 12, cudaMemcpyHostToDevice);
    float hq[S_NQ];
    if (cudaMemcpy(hq, p_qpos, S_NQ * 4, cudaMemcpyDeviceToHost) == cudaSuccess) {
        fwrite(hq, 4, S_NQ, g_render_fp);
        fflush(g_render_fp);
        g_render_frames++;
    }
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

    // SELF-PACED command curriculum: raise the velocity-command range only when
    // the policy tracks the CURRENT range well (mean perf > 0.7). After each
    // raise, the harder commands drop perf below 0.7 until the policy re-learns,
    // so advancement is paced by LEARNING, not by log-call frequency (the bug in
    // earlier versions). vx/vy: 0.25 -> 1.0, step 0.03. This is what walks the
    // robot out of the stand-still optimum.
    static float g_vmax = 0.25f;
    if (out->n > 0.0f) {
        float mean_perf = out->perf / out->n;
        if (mean_perf > 0.7f && g_vmax < 1.0f)
            g_vmax = fminf(1.0f, g_vmax + 0.03f);
        else if (mean_perf < 0.4f && g_vmax > 0.25f)
            g_vmax = fmaxf(0.25f, g_vmax - 0.03f);
        CUDA_CHECK(cudaMemcpyToSymbol(g2e_cmd_vx, &g_vmax, 4));
        CUDA_CHECK(cudaMemcpyToSymbol(g2e_cmd_vy, &g_vmax, 4));
    }
    return out->n;
}

extern "C" void my_gpu_close(void) { /* process teardown frees the device */ }
