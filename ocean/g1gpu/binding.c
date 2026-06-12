/* g1gpu: GPU-NATIVE G1 velocity-command locomotion (mujoco-ultra-fast Phase 3).
 *
 * Same task contract as ocean/g1 (96 obs / 29 continuous actions / frozen
 * task v1.1), but the envs live ENTIRELY on the GPU: the Phase-2 staged
 * CUDA engine (validated vs the MuJoCo C engine + the CPU g1 env; GATE-D
 * 5.68M physics steps/s standalone) steps all agents in-place on the vecenv
 * device buffers. No libmujoco, no per-env mjData, no D2H/H2D in the loop.
 * Kernels + hooks live in g1_gpu.cu (compiled by build.sh into the env lib).
 */
#include <string.h>

typedef struct {
    float perf;            /* mean lin-vel tracking kernel over episode */
    float score;           /* episode return */
    float episode_return;
    float episode_length;
    float vel_err;
    float falls;
    float n;               /* required last field */
} Log;

typedef struct {
    Log log;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int num_agents;
} G1Gpu;

#define Env G1Gpu
#define OBS_SIZE 96
#define NUM_ATNS 29
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
#define OBS_TENSOR_T FloatTensor
#define MY_GPU_NATIVE 1

#include "vecenv.h"

/* config push implemented in g1_gpu.cu */
void my_gpu_config(float action_scale, float w_track_lin, float w_track_ang,
                   float w_lin_vel_z, float w_ang_vel_xy, float w_orientation,
                   float w_torque, float w_action_rate, float w_alive,
                   float w_termination, int max_episode_len);

void my_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
    static int configured = 0;
    if (!configured) {
        my_gpu_config(
            (float)dict_get(kwargs, "action_scale")->value,
            (float)dict_get(kwargs, "w_track_lin")->value,
            (float)dict_get(kwargs, "w_track_ang")->value,
            (float)dict_get(kwargs, "w_lin_vel_z")->value,
            (float)dict_get(kwargs, "w_ang_vel_xy")->value,
            (float)dict_get(kwargs, "w_orientation")->value,
            (float)dict_get(kwargs, "w_torque")->value,
            (float)dict_get(kwargs, "w_action_rate")->value,
            (float)dict_get(kwargs, "w_alive")->value,
            (float)dict_get(kwargs, "w_termination")->value,
            (int)dict_get(kwargs, "max_episode_len")->value);
        configured = 1;
    }
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "vel_err", log->vel_err);
    dict_set(out, "falls", log->falls);
}

/* CPU-path stubs: never invoked under MY_GPU_NATIVE, but vecenv references
 * the symbols. */
void c_reset(Env* env) { (void)env; }
void c_step(Env* env) { (void)env; }
void c_close(Env* env) { (void)env; }
void c_render(Env* env) { (void)env; }
