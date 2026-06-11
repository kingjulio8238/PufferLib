/* G1 velocity-command locomotion (Phase 1 of mujoco-ultra-fast).
 *
 * Unitree G1 (29 DOF, playground G1JoystickFlatTerrain model, frozen at the
 * Phase-0 wall: dt=0.002, Newton 3 iter, pyramidal cone) wrapped as a
 * PufferLib ocean env. CPU MuJoCo stepping; one mjData per env; one shared
 * read-only mjModel (safe across OMP threads).
 *
 * Control: 50 Hz (G1_DECIMATION=10 physics substeps). Actions are 29
 * continuous PD-target offsets around the 'home' keyframe pose; the model's
 * actuators are position servos (kp=75 baked in MJCF), so ctrl = target angle.
 *
 * Obs v1-minimal (96 floats) — richer playground-style obs is a planned
 * upgrade (docs/g1_task.md):
 *   [0:3)   base angular velocity (base frame, scaled 0.25)
 *   [3:6)   projected gravity (base frame)
 *   [6:9)   velocity command (vx, vy, wyaw)
 *   [9:38)  joint pos - default (29)
 *   [38:67) joint vel (29, scaled 0.05)
 *   [67:96) previous action (29)
 *
 * Rewards: legged-gym style, scaled by ctrl dt (0.02) to stay well within
 * PufferLib's reward clamp. Env self-resets on termination (ocean contract).
 */

#pragma once
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mujoco/mujoco.h>

#define G1_NUM_JOINTS 29
#define G1_OBS_SIZE 96
#define G1_DECIMATION 10
#define G1_CTRL_DT 0.02f

// Frozen wall spec (docs/baselines.md) — guarded at model load.
#define G1_WALL_TIMESTEP 0.002
#define G1_WALL_SOLVER mjSOL_NEWTON
#define G1_WALL_ITERATIONS 3
#define G1_WALL_LS_ITERATIONS 5
#define G1_WALL_CONE mjCONE_PYRAMIDAL

typedef struct {
    float perf;            // mean lin-vel tracking kernel over episode (0-1)
    float score;           // episode return
    float episode_return;
    float episode_length;
    float vel_err;         // mean |cmd - v| at episode end
    float falls;           // 1 if episode ended in a fall
    float n;               // required last field
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
    unsigned int rng;
    // --- mujoco ---
    mjData* d;
    // --- episode state ---
    float cmd[3];                       // vx, vy, wyaw command
    float prev_action[G1_NUM_JOINTS];
    int tick;                           // control steps this episode
    float ep_return;
    float ep_track_sum;                 // sum of lin-vel tracking kernel
    float ep_vel_err_sum;
    // --- config (set via g1_set_default_config / binding kwargs) ---
    int max_episode_len;                // control steps (1000 = 20 s)
    int cmd_resample_interval;          // control steps
    float action_scale;                 // rad per unit action
    float reset_noise;                  // rad joint noise at reset
    float w_track_lin, w_track_ang;     // positive weights
    float w_lin_vel_z, w_ang_vel_xy;    // negative weights (penalties)
    float w_orientation, w_torque, w_action_rate;
    float w_alive;          // per-step survival bonus (dt-scaled)
    float w_termination;    // one-time raw penalty on fall (NOT dt-scaled;
                            // trainer clamps rewards to +-1)
    float term_height;                  // fall if pelvis z below
    float term_gravity_z;               // fall if proj-gravity z above (toward 0)
} G1;

// ---------------------------------------------------------------------------
// Shared model (read-only during stepping -> OMP-safe). Loaded once.
// ---------------------------------------------------------------------------
static mjModel* g1_model = NULL;
static int g1_model_refs = 0;
static int g1_key_home = -1;
static mjtNum g1_default_qpos[G1_NUM_JOINTS];
static mjtNum g1_default_ctrl[G1_NUM_JOINTS];

static void g1_load_model(void) {
    if (g1_model != NULL) return;
    const char* candidates[] = {
        getenv("G1_MODEL_PATH"),
        "ocean/g1/model/g1.mjb",      // running from PufferLib root
        "envs/g1/model/g1.mjb",       // running from mujoco-ultra-fast root
        "../../envs/g1/model/g1.mjb", // running from vendor/PufferLib
        "/root/model/g1.mjb",         // Modal image bake location
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); i++) {
        if (candidates[i] == NULL) continue;
        g1_model = mj_loadModel(candidates[i], NULL);
        if (g1_model != NULL) break;
    }
    if (g1_model == NULL) {
        fprintf(stderr, "g1: FAILED to load g1.mjb (set G1_MODEL_PATH)\n");
        exit(1);
    }

    // Physics-parity guard: this env must run the EXACT wall physics.
    if (g1_model->opt.timestep != G1_WALL_TIMESTEP ||
        g1_model->opt.solver != G1_WALL_SOLVER ||
        g1_model->opt.iterations != G1_WALL_ITERATIONS ||
        g1_model->opt.ls_iterations != G1_WALL_LS_ITERATIONS ||
        g1_model->opt.cone != G1_WALL_CONE ||
        g1_model->nq != 36 || g1_model->nv != 35 || g1_model->nu != G1_NUM_JOINTS) {
        fprintf(stderr, "g1: model does not match the frozen wall spec!\n");
        exit(1);
    }

    // Trim per-env arena: default narena is sized generously; this model's
    // constraint state is tiny (~96 efc rows). Validated via maxuse_arena in
    // the smoke test. 1 MB x 4096 envs = 4 GB host RAM.
    g1_model->narena = 1 << 20;

    g1_key_home = mj_name2id(g1_model, mjOBJ_KEY, "home");
    if (g1_key_home < 0) g1_key_home = 0;
    for (int j = 0; j < G1_NUM_JOINTS; j++) {
        g1_default_qpos[j] = g1_model->key_qpos[g1_key_home * g1_model->nq + 7 + j];
        g1_default_ctrl[j] = g1_model->key_ctrl[g1_key_home * g1_model->nu + j];
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static inline float g1_urand(G1* env) {  // uniform [-1, 1]
    return 2.0f * ((float)rand_r(&env->rng) / (float)RAND_MAX) - 1.0f;
}

static inline float g1_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

// rotate world vector into base frame: out = R(q)^T v
static inline void g1_world_to_base(const mjtNum q[4], const mjtNum v[3], mjtNum out[3]) {
    mjtNum qinv[4];
    mju_negQuat(qinv, q);
    mju_rotVecQuat(out, v, qinv);
}

static void g1_sample_command(G1* env) {
    if ((float)rand_r(&env->rng) / (float)RAND_MAX < 0.1f) {  // 10% stand still
        env->cmd[0] = env->cmd[1] = env->cmd[2] = 0.0f;
        return;
    }
    env->cmd[0] = 1.0f * g1_urand(env);   // vx in [-1, 1] m/s
    env->cmd[1] = 0.6f * g1_urand(env);   // vy in [-0.6, 0.6] m/s
    env->cmd[2] = 1.0f * g1_urand(env);   // wyaw in [-1, 1] rad/s
}

void g1_set_default_config(G1* env) {
    env->max_episode_len = 1000;       // 20 s
    env->cmd_resample_interval = 500;  // 10 s
    env->action_scale = 0.5f;
    env->reset_noise = 0.05f;
    env->w_track_lin = 1.0f;
    env->w_track_ang = 0.5f;
    env->w_lin_vel_z = -2.0f;
    env->w_ang_vel_xy = -0.05f;
    env->w_orientation = -5.0f;
    env->w_torque = -2e-5f;
    env->w_action_rate = -0.01f;
    env->w_alive = 0.25f;          // 0.005/step after dt scale — small vs
                                   // tracking's 0.03/step (no camping)
    env->w_termination = -1.0f;    // dying instantly forfeits ~33 steps of
                                   // perfect tracking — survival now pays
    env->term_height = 0.35f;
    env->term_gravity_z = -0.6f;       // upright is -1; fall if above (>~53 deg)
}

void g1_init(G1* env) {
    g1_load_model();
    env->num_agents = 1;
    env->d = mj_makeData(g1_model);
    g1_model_refs++;
    if (env->max_episode_len == 0) g1_set_default_config(env);
}

// ---------------------------------------------------------------------------
// Observations
// ---------------------------------------------------------------------------
void compute_observations(G1* env) {
    mjData* d = env->d;
    float* obs = env->observations;
    const mjtNum* quat = d->qpos + 3;

    mjtNum ang_base[3];                       // free-joint ang vel is already local
    ang_base[0] = d->qvel[3]; ang_base[1] = d->qvel[4]; ang_base[2] = d->qvel[5];

    static const mjtNum GRAV_WORLD[3] = {0.0, 0.0, -1.0};
    mjtNum grav_base[3];
    g1_world_to_base(quat, GRAV_WORLD, grav_base);

    obs[0] = 0.25f * (float)ang_base[0];
    obs[1] = 0.25f * (float)ang_base[1];
    obs[2] = 0.25f * (float)ang_base[2];
    obs[3] = (float)grav_base[0];
    obs[4] = (float)grav_base[1];
    obs[5] = (float)grav_base[2];
    obs[6] = env->cmd[0];
    obs[7] = env->cmd[1];
    obs[8] = env->cmd[2];
    for (int j = 0; j < G1_NUM_JOINTS; j++) {
        obs[9 + j]  = (float)(d->qpos[7 + j] - g1_default_qpos[j]);
        obs[38 + j] = 0.05f * (float)d->qvel[6 + j];
        obs[67 + j] = env->prev_action[j];
    }
}

// ---------------------------------------------------------------------------
// Episode lifecycle
// ---------------------------------------------------------------------------
void add_log(G1* env, int fell) {
    env->log.perf += env->tick > 0 ? env->ep_track_sum / (float)env->tick : 0.0f;
    env->log.score += env->ep_return;
    env->log.episode_return += env->ep_return;
    env->log.episode_length += (float)env->tick;
    env->log.vel_err += env->tick > 0 ? env->ep_vel_err_sum / (float)env->tick : 0.0f;
    env->log.falls += fell ? 1.0f : 0.0f;
    env->log.n++;
}

void c_reset(G1* env) {
    mjData* d = env->d;
    mj_resetDataKeyframe(g1_model, d, g1_key_home);
    for (int j = 0; j < G1_NUM_JOINTS; j++) {
        d->qpos[7 + j] += env->reset_noise * g1_urand(env);
        env->prev_action[j] = 0.0f;
    }
    mj_forward(g1_model, d);
    g1_sample_command(env);
    env->tick = 0;
    env->ep_return = 0.0f;
    env->ep_track_sum = 0.0f;
    env->ep_vel_err_sum = 0.0f;
    compute_observations(env);
}

void c_step(G1* env) {
    mjData* d = env->d;
    env->rewards[0] = 0.0f;
    env->terminals[0] = 0.0f;

    // --- apply action: PD targets around default pose (in-model servos) ---
    float a[G1_NUM_JOINTS];
    for (int j = 0; j < G1_NUM_JOINTS; j++) {
        a[j] = g1_clampf(env->actions[j], -1.0f, 1.0f);
        mjtNum target = g1_default_ctrl[j] + (mjtNum)(env->action_scale * a[j]);
        mjtNum lo = g1_model->actuator_ctrlrange[2 * j];
        mjtNum hi = g1_model->actuator_ctrlrange[2 * j + 1];
        d->ctrl[j] = target < lo ? lo : (target > hi ? hi : target);
    }

    for (int k = 0; k < G1_DECIMATION; k++) mj_step(g1_model, d);
    env->tick++;

    // --- base-frame quantities ---
    const mjtNum* quat = d->qpos + 3;
    mjtNum vel_base[3];
    g1_world_to_base(quat, d->qvel, vel_base);   // world lin vel -> base frame
    mjtNum wx = d->qvel[3], wy = d->qvel[4], wz = d->qvel[5];  // local ang vel
    static const mjtNum GRAV_WORLD[3] = {0.0, 0.0, -1.0};
    mjtNum pg[3];
    g1_world_to_base(quat, GRAV_WORLD, pg);

    // --- rewards (legged-gym style) ---
    float ex = env->cmd[0] - (float)vel_base[0];
    float ey = env->cmd[1] - (float)vel_base[1];
    float lin_err2 = ex * ex + ey * ey;
    float track_lin = expf(-lin_err2 / 0.25f);
    float eyaw = env->cmd[2] - (float)wz;
    float track_ang = expf(-eyaw * eyaw / 0.25f);

    float torque2 = 0.0f, act_rate2 = 0.0f;
    for (int j = 0; j < G1_NUM_JOINTS; j++) {
        float tau = (float)d->actuator_force[j];
        torque2 += tau * tau;
        float da = a[j] - env->prev_action[j];
        act_rate2 += da * da;
        env->prev_action[j] = a[j];
    }

    float r = env->w_alive
            + env->w_track_lin * track_lin
            + env->w_track_ang * track_ang
            + env->w_lin_vel_z * (float)(vel_base[2] * vel_base[2])
            + env->w_ang_vel_xy * (float)(wx * wx + wy * wy)
            + env->w_orientation * (float)(pg[0] * pg[0] + pg[1] * pg[1])
            + env->w_torque * torque2
            + env->w_action_rate * act_rate2;
    float reward = r * G1_CTRL_DT;

    env->rewards[0] = reward;
    env->ep_return += reward;
    env->ep_track_sum += track_lin;
    env->ep_vel_err_sum += sqrtf(lin_err2);

    // --- termination / truncation / command resample ---
    int fell = (d->qpos[2] < (mjtNum)env->term_height) ||
               (pg[2] > (mjtNum)env->term_gravity_z) ||
               !isfinite((float)d->qpos[2]);
    int timeout = env->tick >= env->max_episode_len;

    if (fell || timeout) {
        if (fell) {  // one-time penalty: make falling expensive (raw units)
            env->rewards[0] += env->w_termination;
            env->ep_return += env->w_termination;
        }
        env->terminals[0] = 1.0f;
        add_log(env, fell);
        c_reset(env);
        return;
    }
    if (env->cmd_resample_interval > 0 &&
        env->tick % env->cmd_resample_interval == 0) {
        g1_sample_command(env);
    }
    compute_observations(env);
}

void c_close(G1* env) {
    if (env->d != NULL) {
        mj_deleteData(env->d);
        env->d = NULL;
    }
    if (--g1_model_refs == 0 && g1_model != NULL) {
        mj_deleteModel(g1_model);
        g1_model = NULL;
    }
}

// Rendering lives in g1.c (raylib); vecenv needs the symbol to exist.
#ifndef G1_HAS_RENDER
void c_render(G1* env) { (void)env; }
#endif
