// g1_mirror.h — G1 left<->right mirror map for N1 symmetry regularization.
//
// Generated + VALIDATED by scripts/validate_g1_mirror.py against the MuJoCo C
// engine: gravity/kinematic mirror-equivariance is exact (2e-15); obs/action map
// is an exact involution (0e0). (A ~3% velocity-coupling residual is a documented
// inertia-frame quirk in the published menagerie G1 — equal principal moments but
// non-mirror-paired iquat on hip_yaw/ankle_pitch/arms — not a map error, and
// irrelevant to a soft symmetry prior.)
//
// obs layout (V3, 98-d): [0:3] 0.25*base_angvel  [3:6] proj_gravity  [6:9] cmd
//   [9:38] qpos-key (29 joints)  [38:67] 0.05*qvel  [67:96] prev_action
//   [96:98] sin/cos(gait phase).  Mirror = sagittal-plane reflection (y flips):
//   angvel(wx,wz flip), gravity(gy), cmd(vy,yaw), joints L<->R swap + roll/yaw sign
//   flip, gait phase -> antiphase (sin,cos negate).
// action layout (29-d): joint targets, same L<->R swap + roll/yaw sign flip.
#ifndef G1_MIRROR_H
#define G1_MIRROR_H

#define G1_MIRROR_OBS 98
#define G1_MIRROR_ACT 29

// output[i] = sign[i] * input[src[i]]
__device__ __constant__ int   G1_OBS_MIRROR_SRC[98] = {
  0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 18, 19,
  20, 9, 10, 11, 12, 13, 14, 21, 22, 23, 31, 32, 33, 34,
  35, 36, 37, 24, 25, 26, 27, 28, 29, 30, 44, 45, 46, 47,
  48, 49, 38, 39, 40, 41, 42, 43, 50, 51, 52, 60, 61, 62,
  63, 64, 65, 66, 53, 54, 55, 56, 57, 58, 59, 73, 74, 75,
  76, 77, 78, 67, 68, 69, 70, 71, 72, 79, 80, 81, 89, 90,
  91, 92, 93, 94, 95, 82, 83, 84, 85, 86, 87, 88, 96, 97 };
__device__ __constant__ float G1_OBS_MIRROR_SIGN[98] = {
  -1.f, +1.f, -1.f, +1.f, -1.f, +1.f, +1.f, -1.f, -1.f, +1.f, -1.f, -1.f, +1.f, +1.f,
  -1.f, +1.f, -1.f, -1.f, +1.f, +1.f, -1.f, -1.f, -1.f, +1.f, +1.f, -1.f, -1.f, +1.f,
  -1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, -1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f,
  +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, +1.f, -1.f, -1.f, -1.f, +1.f, +1.f, -1.f, -1.f,
  +1.f, -1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, -1.f, +1.f, -1.f, +1.f, -1.f, -1.f,
  +1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, +1.f, -1.f, -1.f, -1.f, +1.f, +1.f, -1.f,
  -1.f, +1.f, -1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, -1.f, +1.f, -1.f, -1.f, -1.f };
__device__ __constant__ int   G1_ACT_MIRROR_SRC[29] = {
  6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13,
  14, 22, 23, 24, 25, 26, 27, 28, 15, 16, 17, 18, 19, 20, 21 };
__device__ __constant__ float G1_ACT_MIRROR_SIGN[29] = {
  +1.f, -1.f, -1.f, +1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, +1.f, -1.f, -1.f, -1.f,
  +1.f, +1.f, -1.f, -1.f, +1.f, -1.f, +1.f, -1.f, +1.f, -1.f, -1.f, +1.f, -1.f, +1.f, -1.f };

// mirror an obs vector (98-d): dst may not alias src.
__device__ __forceinline__ void g1_mirror_obs(const float* __restrict__ src,
                                               float* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < G1_MIRROR_OBS; i++) dst[i] = G1_OBS_MIRROR_SIGN[i] * src[G1_OBS_MIRROR_SRC[i]];
}

// mirror an action/mean vector (29-d): dst may not alias src.
__device__ __forceinline__ void g1_mirror_act(const float* __restrict__ src,
                                               float* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < G1_MIRROR_ACT; i++) dst[i] = G1_ACT_MIRROR_SIGN[i] * src[G1_ACT_MIRROR_SRC[i]];
}

#endif // G1_MIRROR_H
