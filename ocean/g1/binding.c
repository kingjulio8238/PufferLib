#include "g1.h"
#define OBS_SIZE 96
#define NUM_ATNS 29
#define ACT_SIZES {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, \
                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}  // 29 continuous dims
#define OBS_TENSOR_T FloatTensor

#define Env G1
#include "vecenv.h"

void my_init(Env* env, Dict* kwargs) {
    g1_set_default_config(env);
    env->action_scale = (float)dict_get(kwargs, "action_scale")->value;
    env->max_episode_len = (int)dict_get(kwargs, "max_episode_len")->value;
    env->w_track_lin = (float)dict_get(kwargs, "w_track_lin")->value;
    env->w_track_ang = (float)dict_get(kwargs, "w_track_ang")->value;
    env->w_lin_vel_z = (float)dict_get(kwargs, "w_lin_vel_z")->value;
    env->w_ang_vel_xy = (float)dict_get(kwargs, "w_ang_vel_xy")->value;
    env->w_orientation = (float)dict_get(kwargs, "w_orientation")->value;
    env->w_torque = (float)dict_get(kwargs, "w_torque")->value;
    env->w_action_rate = (float)dict_get(kwargs, "w_action_rate")->value;
    env->w_alive = (float)dict_get(kwargs, "w_alive")->value;
    env->w_termination = (float)dict_get(kwargs, "w_termination")->value;
    g1_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "perf", log->perf);
    dict_set(out, "score", log->score);
    dict_set(out, "episode_return", log->episode_return);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "vel_err", log->vel_err);
    dict_set(out, "falls", log->falls);
}
