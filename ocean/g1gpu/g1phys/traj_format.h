// Reference-trajectory binary format for GPU-kernel validation (Phase 2 A1).
//
// One file = header + nsteps fixed-stride TrajStep records. All doubles are
// the MuJoCo C engine's native fp64. The GPU side loads, converts to fp32,
// replays stages from each step's PRE-state, and compares per stage.
//
// Stage pairing: xpos/xquat/qM/... are the forward() outputs for the PRE-step
// (qpos, qvel, ctrl); qpos_next/qvel_next are the post-Euler state.
#pragma once
#include <stdint.h>

#define TRAJ_MAGIC   0x47315452u  // "G1TR"
#define TRAJ_VERSION 2
#define TRAJ_NQ      36
#define TRAJ_NV      35
#define TRAJ_NU      29
#define TRAJ_NBODY   31
#define TRAJ_NCONMAX 64

enum { TRAJ_SCEN_AIR = 0, TRAJ_SCEN_STAND = 1, TRAJ_SCEN_RANDOM = 2 };

typedef struct {
    uint32_t magic, version;
    int32_t nsteps, scenario;
    uint32_t seed;
    int32_t nq, nv, nu, nbody, nconmax;
    double dt;
    char model_md5[36];
} TrajHeader;

typedef struct {
    int32_t geom1, geom2;
    double dist;
    double pos[3];
    double frame[9];
} TrajContact;

typedef struct {
    // pre-step state + action
    double qpos[TRAJ_NQ], qvel[TRAJ_NV], ctrl[TRAJ_NU];
    // solver warm-start (REQUIRED for reproducibility: Newton at the
    // wall's fixed 3 iterations does not converge, so its output depends
    // on the start point — the GPU solver must warm-start identically)
    double qacc_warmstart[TRAJ_NV];
    // forward() stage outputs (for the pre-step state)
    double xpos[TRAJ_NBODY * 3], xquat[TRAJ_NBODY * 4];
    double qM_dense[TRAJ_NV * TRAJ_NV];     // via mj_fullM
    double qfrc_bias[TRAJ_NV], qfrc_passive[TRAJ_NV];
    double qacc_smooth[TRAJ_NV];
    double qfrc_constraint[TRAJ_NV], qacc[TRAJ_NV];
    int32_t ncon, pad_;
    TrajContact con[TRAJ_NCONMAX];
    // post-Euler state
    double qpos_next[TRAJ_NQ], qvel_next[TRAJ_NV];
} TrajStep;
