// g1_staged_kernels.cuh — the per-stage G1 physics kernels (Workstream E).
// E1+E2 validated 2026-06-12: full-physics het wallproto 4.02M steps/s H100
// @32768 (1.20x the warp wall), validation contact-equivalent.
// Requires (before include): CUDA_CHECK, g1_step.cuh, g1_full_step.cuh.
#pragma once

#define G1_TRI (G1_NV * (G1_NV + 1) / 2)
__device__ __forceinline__ int tridx(int i, int j) { return i * (i + 1) / 2 + j; }

#ifndef SWARPS
#define SWARPS 8                   // warps (envs) per stage-kernel block
#endif

// SoA strides
#define S_NQ G1_NQ
#define S_NV G1_NV
#define S_NU G1_NU
#define S_X3 (G1_NBODY * 3)
#define S_X4 (G1_NBODY * 4)
#define S_CI (G1_NBODY * 10)
#define S_CD (G1_NV * 6)

// ---------------------------------------------------------------------------
// K1: FK + comPos. shared/env: qpos 36 + xpos 93 + xquat 124 + xipos 93 +
// com 3 = 349 floats (1.4KB)
// ---------------------------------------------------------------------------
__global__ void k1_fk_compos(int n, const float* __restrict__ g_qpos,
                             float* __restrict__ g_xpos, float* __restrict__ g_xquat,
                             float* __restrict__ g_com, float* __restrict__ g_cinert,
                             float* __restrict__ g_cdof) {
    __shared__ float s_qpos[SWARPS][S_NQ];
    __shared__ float s_xpos[SWARPS][S_X3];
    __shared__ float s_xquat[SWARPS][S_X4];
    __shared__ float s_xipos[SWARPS][S_X3];
    __shared__ float s_com[SWARPS][3];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* qpos = s_qpos[warp];
    float* xpos = s_xpos[warp];
    float* xquat = s_xquat[warp];
    float* xipos = s_xipos[warp];
    float* com = s_com[warp];

    for (int k = lane; k < S_NQ; k += 32) qpos[k] = g_qpos[(size_t)e * S_NQ + k];
    if (lane == 0) {
        xpos[0] = xpos[1] = xpos[2] = 0.0f;
        xquat[0] = 1.0f; xquat[1] = xquat[2] = xquat[3] = 0.0f;
    }
    __syncwarp();
    for (int lv = 0; lv < G1_NLEVELS; lv++) {
        int off = g1c_level_offset[lv], cnt = g1c_level_offset[lv + 1] - off;
        if (lane < cnt) fk_body(g1c_level_bodies[off + lane], qpos, xpos, xquat);
        __syncwarp();
    }

    // xipos + root COM
    float part[3] = {0.0f, 0.0f, 0.0f};
    if (lane >= 1 && lane < G1_NBODY) {
        int i = lane;
        rot_vec_quat(xipos + 3 * i, g1c_body_ipos + 3 * i, xquat + 4 * i);
        xipos[3*i]   += xpos[3*i];
        xipos[3*i+1] += xpos[3*i+1];
        xipos[3*i+2] += xpos[3*i+2];
        float mass = g1c_body_mass[i];
        part[0] = mass * xipos[3*i];
        part[1] = mass * xipos[3*i+1];
        part[2] = mass * xipos[3*i+2];
    }
    for (int o = 16; o > 0; o >>= 1) {
        part[0] += __shfl_down_sync(0xffffffff, part[0], o);
        part[1] += __shfl_down_sync(0xffffffff, part[1], o);
        part[2] += __shfl_down_sync(0xffffffff, part[2], o);
    }
    if (lane == 0) {
        float inv = 1.0f / g1c_body_subtreemass[1];
        com[0] = part[0] * inv; com[1] = part[1] * inv; com[2] = part[2] * inv;
    }
    __syncwarp();

    // cinert (registers -> global)
    if (lane >= 1 && lane < G1_NBODY) {
        int i = lane;
        float iq[4], mat[9], dif[3], ci[10];
        quat_mul(iq, xquat + 4 * i, g1c_body_iquat + 4 * i);
        quat2mat(mat, iq);
        dif[0] = xipos[3*i]   - com[0];
        dif[1] = xipos[3*i+1] - com[1];
        dif[2] = xipos[3*i+2] - com[2];
        inert_com(ci, g1c_body_inertia + 3 * i, mat, dif, g1c_body_mass[i]);
        for (int k = 0; k < 10; k++) g_cinert[(size_t)e * S_CI + 10 * i + k] = ci[k];
    } else if (lane == 0) {
        for (int k = 0; k < 10; k++) g_cinert[(size_t)e * S_CI + k] = 0.0f;
    }

    // cdof (registers -> global)
    float* gcd = g_cdof + (size_t)e * S_CD;
    if (lane == 0) {
        int da = g1c_jnt_dofadr[0];
        float mat1[9];
        quat2mat(mat1, xquat + 4);
        for (int k = 0; k < 18; k++) gcd[6 * da + k] = 0.0f;
        gcd[6*(da+0) + 3] = 1.0f;
        gcd[6*(da+1) + 4] = 1.0f;
        gcd[6*(da+2) + 5] = 1.0f;
        float off3[3] = {com[0] - xpos[3], com[1] - xpos[4], com[2] - xpos[5]};
        for (int k = 0; k < 3; k++) {
            float axis[3] = {mat1[k], mat1[k + 3], mat1[k + 6]};
            float cd[6];
            cd[0] = axis[0]; cd[1] = axis[1]; cd[2] = axis[2];
            cross3(cd + 3, axis, off3);
            for (int c = 0; c < 6; c++) gcd[6 * (da + 3 + k) + c] = cd[c];
        }
    } else if (lane < G1_NJNT) {
        int j = lane;
        int b = g1c_jnt_bodyid[j];
        int da = g1c_jnt_dofadr[j];
        float axis[3], anchor[3], cd[6];
        rot_vec_quat(axis, g1c_jnt_axis + 3 * j, xquat + 4 * b);
        rot_vec_quat(anchor, g1c_jnt_pos + 3 * j, xquat + 4 * b);
        float off3[3] = {com[0] - (anchor[0] + xpos[3*b]),
                         com[1] - (anchor[1] + xpos[3*b+1]),
                         com[2] - (anchor[2] + xpos[3*b+2])};
        cd[0] = axis[0]; cd[1] = axis[1]; cd[2] = axis[2];
        cross3(cd + 3, axis, off3);
        for (int c = 0; c < 6; c++) gcd[6 * da + c] = cd[c];
    }

    // pose out
    for (int k = lane; k < S_X3; k += 32) g_xpos[(size_t)e * S_X3 + k] = xpos[k];
    for (int k = lane; k < S_X4; k += 32) g_xquat[(size_t)e * S_X4 + k] = xquat[k];
    if (lane < 3) g_com[(size_t)e * 3 + lane] = com[lane];
}

// ---------------------------------------------------------------------------
// K2: CRB -> qM -> LDL^T. shared/env: crb 310 + cdof 210 + qM 341 + qLD 341
// = 1202 floats (4.8KB)
// ---------------------------------------------------------------------------
__global__ void k2_crb_factor(int n, const float* __restrict__ g_cinert,
                              const float* __restrict__ g_cdof,
                              float* __restrict__ g_qM, float* __restrict__ g_qLD,
                              float* __restrict__ g_qLDiagInv) {
    // smem diet: qM streams to global as computed (factor happens in qLD);
    // cdof is read-only -> served from L2. 4.8KB -> 2.6KB per env.
    __shared__ float s_crb[SWARPS][S_CI];
    __shared__ float s_qLD[SWARPS][G1_NM];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* crb = s_crb[warp];
    const float* cdof = g_cdof + (size_t)e * S_CD;
    float* qLD = s_qLD[warp];

    for (int k = lane; k < S_CI; k += 32) crb[k] = g_cinert[(size_t)e * S_CI + k];
    __syncwarp();
    if (lane < 10) {
        for (int i = G1_NBODY - 1; i >= 1; i--) {
            int p = g1c_body_parentid[i];
            if (p > 0) crb[10 * p + lane] += crb[10 * i + lane];
        }
    }
    __syncwarp();
    float* gqM = g_qM + (size_t)e * G1_NM;
    for (int i = lane; i < G1_NV; i += 32) {
        float buf[6];
        mul_inert_vec(buf, crb + 10 * g1c_dof_bodyid[i], cdof + 6 * i);
        int adr = g1c_dof_Madr[i];
        float v = g1c_dof_armature[i] + dot6(cdof + 6 * i, buf);
        qLD[adr] = v; gqM[adr] = v; adr++;
        for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j]) {
            v = dot6(cdof + 6 * j, buf);
            qLD[adr] = v; gqM[adr] = v; adr++;
        }
    }
    __syncwarp();
    // LDL^T factor (mj_factorI_legacy; k serial, ancestors lane-parallel)
    for (int k = G1_NV - 1; k >= 0; k--) {
        int Madr_kk = g1c_dof_Madr[k];
        int nanc = (k < G1_NV - 1 ? g1c_dof_Madr[k + 1] : G1_NM) - Madr_kk - 1;
        int i = (lane < nanc) ? g1c_dof_chain[k * G1_MAX_CHAIN + lane] : -1;
        float tmp = 0.0f;
        if (i >= 0) tmp = qLD[Madr_kk + 1 + lane] / qLD[Madr_kk];
        __syncwarp();
        if (i >= 0) {
            int cnt = g1c_dof_Madr[i + 1] - g1c_dof_Madr[i];
            for (int c = 0; c < cnt; c++)
                qLD[g1c_dof_Madr[i] + c] -= qLD[Madr_kk + 1 + lane + c] * tmp;
        }
        __syncwarp();
        if (i >= 0) qLD[Madr_kk + 1 + lane] = tmp;
        __syncwarp();
    }
    for (int k = lane; k < G1_NM; k += 32)
        g_qLD[(size_t)e * G1_NM + k] = qLD[k];
    for (int i = lane; i < G1_NV; i += 32)
        g_qLDiagInv[(size_t)e * G1_NV + i] = 1.0f / qLD[g1c_dof_Madr[i]];
}

// ---------------------------------------------------------------------------
// K3: comVel + RNE + passive + PD actuation + qacc_smooth solve.
// shared/env: cdof 210 + cvel 186 + cacc 186 + cdof_dot 210 + qLD 341 +
// diag 35 + qvel 35 + qpos 36 + smooth/bias/qacc 105 = 1344 floats (5.4KB)
// ---------------------------------------------------------------------------
__global__ void k3_rne_act_solve(int n, const float* __restrict__ g_qpos,
                                 const float* __restrict__ g_qvel,
                                 const float* __restrict__ g_ctrl,
                                 int ctrl_stride,  // S_NU per-env, 0 broadcast
                                 const float* __restrict__ g_cinert,
                                 const float* __restrict__ g_cdof,
                                 const float* __restrict__ g_qLD,
                                 const float* __restrict__ g_qLDiagInv,
                                 float* __restrict__ g_qfrc_smooth,
                                 float* __restrict__ g_qacc_smooth,
                                 float* __restrict__ g_act_force) {
    // smem diet: qLD/diag (lane-0 serial solve) and qpos (one actuation read)
    // served from L2; bias folded into smooth. 5.4KB -> 3.5KB per env.
    __shared__ float s_cdof[SWARPS][S_CD];
    __shared__ float s_cvel[SWARPS][G1_NBODY * 6];
    __shared__ float s_cacc[SWARPS][G1_NBODY * 6];
    __shared__ float s_cdd[SWARPS][S_CD];
    __shared__ float s_qvel[SWARPS][G1_NV];
    __shared__ float s_smooth[SWARPS][G1_NV];
    __shared__ float s_qacc[SWARPS][G1_NV];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* cdof = s_cdof[warp];
    float* cvel = s_cvel[warp];
    float* cacc = s_cacc[warp];
    float* cdd = s_cdd[warp];
    const float* qLD = g_qLD + (size_t)e * G1_NM;
    const float* diag = g_qLDiagInv + (size_t)e * G1_NV;
    float* qvel = s_qvel[warp];
    const float* qpos = g_qpos + (size_t)e * S_NQ;
    float* smoo = s_smooth[warp];
    float* qacc = s_qacc[warp];

    for (int k = lane; k < S_CD; k += 32) cdof[k] = g_cdof[(size_t)e * S_CD + k];
    for (int k = lane; k < G1_NV; k += 32)
        qvel[k] = g_qvel[(size_t)e * G1_NV + k];
    if (lane == 0) for (int k = 0; k < 6; k++) cvel[k] = 0.0f;
    __syncwarp();

    // comVel (level-synchronized; identical math to smooth_forward)
    for (int lv = 0; lv < G1_NLEVELS; lv++) {
        int off = g1c_level_offset[lv], cnt = g1c_level_offset[lv + 1] - off;
        if (lane < cnt) {
            int i = g1c_level_bodies[off + lane];
            int bda = g1c_body_dofadr[i], dofnum = g1c_body_dofnum[i];
            float v[6];
            for (int c = 0; c < 6; c++) v[c] = cvel[6 * g1c_body_parentid[i] + c];
            if (dofnum == 6) {
                for (int k = 0; k < 3; k++)
                    for (int c = 0; c < 6; c++)
                        v[c] += cdof[6 * (bda + k) + c] * qvel[bda + k];
                for (int k = 0; k < 3; k++) {
                    for (int c = 0; c < 6; c++) cdd[6*(bda+k) + c] = 0.0f;
                    cross_motion(cdd + 6 * (bda + 3 + k), v, cdof + 6 * (bda + 3 + k));
                }
                for (int k = 0; k < 3; k++)
                    for (int c = 0; c < 6; c++)
                        v[c] += cdof[6 * (bda + 3 + k) + c] * qvel[bda + 3 + k];
            } else {
                for (int k = 0; k < dofnum; k++) {
                    cross_motion(cdd + 6 * (bda + k), v, cdof + 6 * (bda + k));
                    for (int c = 0; c < 6; c++)
                        v[c] += cdof[6 * (bda + k) + c] * qvel[bda + k];
                }
            }
            for (int c = 0; c < 6; c++) cvel[6 * i + c] = v[c];
        }
        __syncwarp();
    }

    // RNE (flg_acc=0): cacc forward, cfrc into cvel, backward, project
    if (lane == 0) {
        for (int k = 0; k < 5; k++) cacc[k] = 0.0f;
        cacc[5] = -(G1_GRAVITY_Z);
    }
    __syncwarp();
    const float* gci = g_cinert + (size_t)e * S_CI;
    for (int lv = 0; lv < G1_NLEVELS; lv++) {
        int off = g1c_level_offset[lv], cnt = g1c_level_offset[lv + 1] - off;
        if (lane < cnt) {
            int i = g1c_level_bodies[off + lane];
            int bda = g1c_body_dofadr[i], dofnum = g1c_body_dofnum[i];
            float ci[10];
            for (int k = 0; k < 10; k++) ci[k] = gci[10 * i + k];
            float a[6];
            for (int c = 0; c < 6; c++) {
                a[c] = cacc[6 * g1c_body_parentid[i] + c];
                for (int k = 0; k < dofnum; k++)
                    a[c] += cdd[6 * (bda + k) + c] * qvel[bda + k];
                cacc[6 * i + c] = a[c];
            }
            float f1[6], iv[6], f2[6];
            mul_inert_vec(f1, ci, a);
            mul_inert_vec(iv, ci, cvel + 6 * i);
            cross_force(f2, cvel + 6 * i, iv);
            for (int c = 0; c < 6; c++) cvel[6 * i + c] = f1[c] + f2[c];
        }
        __syncwarp();
    }
    if (lane < 6) {
        for (int i = G1_NBODY - 1; i >= 1; i--) {
            int p = g1c_body_parentid[i];
            if (p) cvel[6 * p + lane] += cvel[6 * i + lane];
        }
    }
    __syncwarp();
    // passive + bias fusion + PD actuation (bias folded; identical op order)
    for (int i = lane; i < G1_NV; i += 32) {
        float b = dot6(cdof + 6 * i, cvel + 6 * g1c_dof_bodyid[i]);
        smoo[i] = -g1c_dof_damping[i] * qvel[i] - b;
    }
    __syncwarp();
    if (lane < G1_NU) {
        int a = lane;
        int j = g1c_act_jntid[a];
        int padr = g1c_jnt_qposadr[j], dadr = g1c_jnt_dofadr[j];
        float c = g_ctrl[(size_t)e * ctrl_stride + a];
        float lo = g1c_act_ctrlrange[2 * a], hi = g1c_act_ctrlrange[2 * a + 1];
        c = c < lo ? lo : (c > hi ? hi : c);
        float force = g1c_act_gain0[a] * c + g1c_act_bias1[a] * qpos[padr]
                      + g1c_act_bias2[a] * qvel[dadr];
        if (g1c_jnt_actfrclimited[j]) {
            float flo = g1c_jnt_actfrcrange[2 * j], fhi = g1c_jnt_actfrcrange[2 * j + 1];
            force = force < flo ? flo : (force > fhi ? fhi : force);
        }
        smoo[dadr] += force;
        g_act_force[(size_t)e * S_NU + a] = force;
    }
    __syncwarp();

    // qacc_smooth = LDL solve (lane-0 serial v1; its own kernel later if hot)
    if (lane == 0) {
        for (int i = 0; i < G1_NV; i++) qacc[i] = smoo[i];
        for (int i = G1_NV - 1; i >= 0; i--) {
            if (qacc[i] != 0.0f) {
                int adr = g1c_dof_Madr[i] + 1;
                for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j])
                    qacc[j] -= qLD[adr++] * qacc[i];
            }
        }
        for (int i = 0; i < G1_NV; i++) qacc[i] *= diag[i];
        for (int i = 0; i < G1_NV; i++) {
            int adr = g1c_dof_Madr[i] + 1;
            for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j])
                qacc[i] -= qLD[adr++] * qacc[j];
        }
    }
    __syncwarp();
    for (int k = lane; k < G1_NV; k += 32) {
        g_qfrc_smooth[(size_t)e * G1_NV + k] = smoo[k];
        g_qacc_smooth[(size_t)e * G1_NV + k] = qacc[k];
    }
}

// ---------------------------------------------------------------------------
// K4: explicit Euler (advance qpos/qvel in place with the given qacc)
// ---------------------------------------------------------------------------
__global__ void k4_euler(int n, float* __restrict__ g_qpos,
                         float* __restrict__ g_qvel,
                         const float* __restrict__ g_qacc) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* qpos = g_qpos + (size_t)e * S_NQ;
    float* qvel = g_qvel + (size_t)e * S_NV;
    const float* qacc = g_qacc + (size_t)e * S_NV;
    for (int i = lane; i < G1_NV; i += 32) qvel[i] += G1_DT * qacc[i];
    __syncwarp();
    if (lane == 0) {
        qpos[0] += G1_DT * qvel[0];
        qpos[1] += G1_DT * qvel[1];
        qpos[2] += G1_DT * qvel[2];
        float q[4] = {qpos[3], qpos[4], qpos[5], qpos[6]};
        quat_integrate(q, qvel + 3, G1_DT);
        qpos[3] = q[0]; qpos[4] = q[1]; qpos[5] = q[2]; qpos[6] = q[3];
    } else if (lane < G1_NJNT) {
        int j = lane;
        qpos[g1c_jnt_qposadr[j]] += G1_DT * qvel[g1c_jnt_dofadr[j]];
    }
}


// ===========================================================================
// E2: CONSTRAINT STAGES (full physics). H is rebuilt every Newton iteration
// by a fast uniform kernel — the megakernel's churn-adaptive machinery is
// unnecessary here (full rebuild == MuJoCo's incremental updates exactly).
// Per-env "done" flag replicates the solver's early-termination breaks.
// ===========================================================================
#define SC_COST 0
#define SC_GAUSS 1
#define SC_QG0 2
#define SC_QG1 3
#define SC_QG2 4
#define SC_ALPHA 5
#define SC_DONE 6
#define SC_SNORM 7
#define NSCAL 8

// parallel symmetric M*v from global sparse qM (descendant lists)
__device__ __forceinline__ void mul_M_vec_g(const float* qM, const float* v,
                                            float* out, int lane) {
    for (int j = lane; j < G1_NV; j += 32) {
        int adr = g1c_dof_Madr[j];
        float s = qM[adr++] * v[j];
        for (int a = g1c_dof_parentid[j]; a >= 0; a = g1c_dof_parentid[a])
            s += qM[adr++] * v[a];
        for (int d = g1c_dof_descoff[j]; d < g1c_dof_descoff[j + 1]; d++)
            s += qM[g1c_dof_desc_adr[d]] * v[g1c_dof_desc_i[d]];
        out[j] = s;
    }
    __syncwarp();
}

// J row · v with the hybrid row structure (cJ in global)
__device__ __forceinline__ float row_dot_g(int rt, int rd, float rs,
                                           const float* cJ, const float* v) {
    if (rt == ROW_FRICTION) return v[rd];
    if (rt == ROW_LIMIT) return rs * v[rd];
    const float* J = cJ + rd * G1_NV;
    float s = 0.0f;
    for (int i = 0; i < G1_NV; i++) s += J[i] * v[i];
    return s;
}

// constraint update on global rows: force/state + cost (all lanes return cost)
__device__ float constraint_update_g(int nefc, const int* rt, const int* rd,
                                     const float* D, const float* R,
                                     const float* jar, float* force, int* state,
                                     int lane, int* out_changed = nullptr) {
    float cost = 0.0f;
    int ch = 0;
    for (int r = lane; r < nefc; r += 32) {
        float f = -D[r] * jar[r];
        int st = ST_QUAD;
        if (rt[r] == ROW_FRICTION) {
            float fl = g1c_dof_frictionloss[rd[r]], Rf = R[r] * fl;
            if (jar[r] <= -Rf) { cost += -0.5f*Rf*fl - fl*jar[r]; f = fl; st = ST_LINNEG; }
            else if (jar[r] >= Rf) { cost += -0.5f*Rf*fl + fl*jar[r]; f = -fl; st = ST_LINPOS; }
            else cost += 0.5f * D[r] * jar[r] * jar[r];
        } else {
            if (jar[r] >= 0.0f) { f = 0.0f; st = ST_SAT; }
            else cost += 0.5f * D[r] * jar[r] * jar[r];
        }
        force[r] = f;
        if (out_changed) ch |= (state[r] != st);
        state[r] = st;
    }
    for (int o = 16; o > 0; o >>= 1) cost += __shfl_xor_sync(0xffffffff, cost, o);
    if (out_changed) *out_changed = __any_sync(0xffffffff, ch);
    return cost;
}

// qfrc = J^T force (lanes over dofs, scan rows)
__device__ void jt_force_g(int nefc, const int* rt, const int* rd, const float* rs,
                           const float* cJ, const float* force, float* qfrc, int lane) {
    for (int i = lane; i < G1_NV; i += 32) {
        float s = 0.0f;
        for (int r = 0; r < nefc; r++) {
            float f = force[r];
            if (f == 0.0f) continue;
            if (rt[r] == ROW_FRICTION) { if (rd[r] == i) s += f; }
            else if (rt[r] == ROW_LIMIT) { if (rd[r] == i) s += rs[r] * f; }
            else s += cJ[rd[r] * G1_NV + i] * f;
        }
        qfrc[i] = s;
    }
    __syncwarp();
}

// ---------------------------------------------------------------------------
// K5: narrowphase + efc assembly (the megakernel assemble_constraints math,
// rows written to global). shared/env: ~606 floats (2.4KB)
// ---------------------------------------------------------------------------
__global__ void k5_assemble(int n, const float* __restrict__ g_qpos,
                            const float* __restrict__ g_qvel,
                            const float* __restrict__ g_xpos,
                            const float* __restrict__ g_xquat,
                            const float* __restrict__ g_gcom,
                            int* __restrict__ g_ncon, int* __restrict__ g_nefc,
                            float* __restrict__ g_condist,
                            int* __restrict__ g_rowtype, int* __restrict__ g_rowdof,
                            float* __restrict__ g_rowsign, int* __restrict__ g_rowstash,
                            float* __restrict__ g_rpos, float* __restrict__ g_D,
                            float* __restrict__ g_R, float* __restrict__ g_aref,
                            float* __restrict__ g_cJ,
                            const float* __restrict__ g_cdof) {
    __shared__ float s_xpos[SWARPS][S_X3];
    __shared__ float s_xquat[SWARPS][S_X4];
    __shared__ float s_qpos[SWARPS][S_NQ];
    __shared__ float s_qvel[SWARPS][S_NV];
    __shared__ float s_com[SWARPS][3];
    __shared__ float s_jd[SWARPS][3 * S_NV];
    __shared__ float s_cdist[SWARPS][NCON_MAX];
    __shared__ float s_cpos[SWARPS][NCON_MAX * 3];
    __shared__ float s_cnorm[SWARPS][NCON_MAX * 3];
    __shared__ int s_cpair[SWARPS][NCON_MAX];
    __shared__ int s_cnt[SWARPS][3];   // ncon, nefc, nl
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* xpos = s_xpos[warp];
    float* xquat = s_xquat[warp];
    float* qpos = s_qpos[warp];
    float* qvel = s_qvel[warp];
    float* com = s_com[warp];
    float* jd = s_jd[warp];
    float* cdist = s_cdist[warp];
    float* cpos = s_cpos[warp];
    float* cnorm = s_cnorm[warp];
    int* cpair = s_cpair[warp];
    int* cnt = s_cnt[warp];

    for (int k = lane; k < S_X3; k += 32) xpos[k] = g_xpos[(size_t)e * S_X3 + k];
    for (int k = lane; k < S_X4; k += 32) xquat[k] = g_xquat[(size_t)e * S_X4 + k];
    for (int k = lane; k < S_NQ; k += 32) qpos[k] = g_qpos[(size_t)e * S_NQ + k];
    for (int k = lane; k < S_NV; k += 32) qvel[k] = g_qvel[(size_t)e * S_NV + k];
    if (lane < 3) com[lane] = g_gcom[(size_t)e * 3 + lane];
    __syncwarp();

    // narrowphase (lane 0, all 5 pairs)
    if (lane == 0) {
        int nc = 0;
        for (int p = 0; p < G1_NPAIR; p++) {
            int g1 = g1c_pair_geom1[p], g2 = g1c_pair_geom2[p];
            int t1 = g1c_geom_type[g1], t2 = g1c_geom_type[g2];
            int need = (t1 == 0 && t2 == 6) ? 4 : 2;
            if (nc + need > NCON_MAX) continue;
            int c2 = 0;
            if (t1 == 0 && t2 == 6)
                c2 = plane_box_np(xpos, xquat, g1, g2, cdist + nc, cpos + 3*nc, cnorm + 3*nc);
            else if (t1 == 3 && t2 == 3)
                c2 = capsule_capsule_np(xpos, xquat, g1, g2, cdist + nc, cpos + 3*nc, cnorm + 3*nc);
            for (int c = 0; c < c2; c++) cpair[nc + c] = p;
            nc += c2;
        }
        cnt[0] = nc;
        // friction + limit rows
        int r = 0;
        for (int i = 0; i < G1_NV; i++) {
            if (g1c_dof_frictionloss[i] == 0.0f) continue;
            g_rowtype[(size_t)e * NEFC_MAX + r] = ROW_FRICTION;
            g_rowdof[(size_t)e * NEFC_MAX + r] = i;
            g_rowsign[(size_t)e * NEFC_MAX + r] = 1.0f;
            g_rpos[(size_t)e * NEFC_MAX + r] = 0.0f;
            r++;
        }
        int nf = r;
        for (int j = 0; j < G1_NJNT; j++) {
            if (!g1c_jnt_limited[j]) continue;
            float value = qpos[g1c_jnt_qposadr[j]];
            for (int side = -1; side <= 1; side += 2) {
                float lim = g1c_jnt_range[2*j + (side + 1)/2];
                float dist = side * (lim - value);
                if (dist < 0.0f) {
                    g_rowtype[(size_t)e * NEFC_MAX + r] = ROW_LIMIT;
                    g_rowdof[(size_t)e * NEFC_MAX + r] = g1c_jnt_dofadr[j];
                    g_rowsign[(size_t)e * NEFC_MAX + r] = -(float)side;
                    g_rpos[(size_t)e * NEFC_MAX + r] = dist;
                    g_rowstash[(size_t)e * NEFC_MAX + r] = j;
                    r++;
                }
            }
        }
        cnt[2] = r - nf;
        cnt[1] = r;   // contacts appended below
    }
    __syncwarp();

    // contact Jacobians + rows
    int ncon = cnt[0];
    int crow = 0;
    for (int c = 0; c < ncon; c++) {
        int p = cpair[c];
        int g1 = g1c_pair_geom1[p], g2 = g1c_pair_geom2[p];
        int b1 = g1c_geom_bodyid[g1], b2 = g1c_geom_bodyid[g2];
        int condim = g1c_pair_dim[p];
        for (int k = lane; k < 3 * G1_NV; k += 32) jd[k] = 0.0f;
        __syncwarp();
        if (lane == 0) {
            const float* point = cpos + 3 * c;
            float off[3] = {point[0] - com[0], point[1] - com[1], point[2] - com[2]};
            const float* gcd = g_cdof + (size_t)e * S_CD;
            for (int side = 0; side < 2; side++) {
                int body = side == 0 ? b1 : b2;
                float sgn = side == 0 ? -1.0f : 1.0f;
                if (body == 0) continue;
                int i = g1c_body_dofadr[body] + g1c_body_dofnum[body] - 1;
                while (i >= 0) {
                    const float* cd = gcd + 6 * i;
                    float t[3];
                    cross3(t, cd, off);
                    jd[0 * G1_NV + i] += sgn * (cd[3] + t[0]);
                    jd[1 * G1_NV + i] += sgn * (cd[4] + t[1]);
                    jd[2 * G1_NV + i] += sgn * (cd[5] + t[2]);
                    i = g1c_dof_parentid[i];
                }
            }
        }
        __syncwarp();
        float fr[9];
        make_frame(fr, cnorm + 3 * c);
        for (int i = lane; i < G1_NV; i += 32) {
            float v0 = jd[0 * G1_NV + i], v1 = jd[1 * G1_NV + i], v2 = jd[2 * G1_NV + i];
            jd[0 * G1_NV + i] = fr[0]*v0 + fr[1]*v1 + fr[2]*v2;
            jd[1 * G1_NV + i] = fr[3]*v0 + fr[4]*v1 + fr[5]*v2;
            jd[2 * G1_NV + i] = fr[6]*v0 + fr[7]*v1 + fr[8]*v2;
        }
        __syncwarp();
        float* gcJ = g_cJ + (size_t)e * NCROW_MAX * G1_NV;
        if (condim == 1) {
            for (int i = lane; i < G1_NV; i += 32) gcJ[crow * G1_NV + i] = jd[i];
            if (lane == 0) {
                int r = cnt[1];
                g_rowtype[(size_t)e * NEFC_MAX + r] = ROW_CONTACT;
                g_rowdof[(size_t)e * NEFC_MAX + r] = crow;
                g_rowsign[(size_t)e * NEFC_MAX + r] = 1.0f;
                g_rpos[(size_t)e * NEFC_MAX + r] = cdist[c];
                g_rowstash[(size_t)e * NEFC_MAX + r] = -1 - c;
                cnt[1] = r + 1;
            }
            crow += 1;
        } else {
            float mu0 = g1c_pair_friction[5 * p + 0];
            float mu1 = g1c_pair_friction[5 * p + 1];
            for (int i = lane; i < G1_NV; i += 32) {
                float j0 = jd[0 * G1_NV + i], j1 = jd[1 * G1_NV + i], j2 = jd[2 * G1_NV + i];
                gcJ[(crow + 0) * G1_NV + i] = j0 + mu0 * j1;
                gcJ[(crow + 1) * G1_NV + i] = j0 - mu0 * j1;
                gcJ[(crow + 2) * G1_NV + i] = j0 + mu1 * j2;
                gcJ[(crow + 3) * G1_NV + i] = j0 - mu1 * j2;
            }
            if (lane == 0) {
                for (int k = 0; k < 4; k++) {
                    int r = cnt[1] + k;
                    g_rowtype[(size_t)e * NEFC_MAX + r] = ROW_CONTACT;
                    g_rowdof[(size_t)e * NEFC_MAX + r] = crow + k;
                    g_rowsign[(size_t)e * NEFC_MAX + r] = 1.0f;
                    g_rpos[(size_t)e * NEFC_MAX + r] = cdist[c];
                    g_rowstash[(size_t)e * NEFC_MAX + r] = -1 - c;
                }
                cnt[1] += 4;
            }
            crow += 4;
        }
        __syncwarp();
    }
    __syncwarp();

    // impedance -> R, aref; then pyramidal R adjust; D = 1/R
    int nefc = cnt[1], nf = nefc - cnt[2] - crow;  // friction count
    (void)nf;
    const float* gcJ = g_cJ + (size_t)e * NCROW_MAX * G1_NV;
    for (int r = lane; r < nefc; r += 32) {
        int rt = g_rowtype[(size_t)e * NEFC_MAX + r];
        int rd = g_rowdof[(size_t)e * NEFC_MAX + r];
        float rposv = g_rpos[(size_t)e * NEFC_MAX + r];
        const float* solref;
        const float* solimp;
        float dA;
        int is_fric = 0;
        if (rt == ROW_FRICTION) {
            solref = g1c_dof_solref + 2 * rd;
            solimp = g1c_dof_solimp + 5 * rd;
            dA = g1c_dof_invweight0[rd];
            is_fric = 1;
        } else if (rt == ROW_LIMIT) {
            int j = g_rowstash[(size_t)e * NEFC_MAX + r];
            solref = g1c_jnt_solref + 2 * j;
            solimp = g1c_jnt_solimp + 5 * j;
            dA = g1c_dof_invweight0[rd];
        } else {
            int c = -1 - g_rowstash[(size_t)e * NEFC_MAX + r];
            int p = cpair[c];
            int g1 = g1c_pair_geom1[p], g2 = g1c_pair_geom2[p];
            solref = g1c_pair_solref + 2 * p;
            solimp = g1c_pair_solimp + 5 * p;
            float tran = g1c_body_invweight0_t[g1c_geom_bodyid[g1]]
                       + g1c_body_invweight0_t[g1c_geom_bodyid[g2]];
            if (g1c_pair_dim[p] == 1) {
                dA = tran;
            } else {
                int k_in = (rd - (rd / 4) * 4) / 2;
                float mu = g1c_pair_friction[5 * p + k_in];
                dA = tran + mu * mu * tran;
            }
        }
        float imp = impedance(solimp, rposv);
        float Rr = fmaxf(1e-15f, (1.0f - imp) * dA / imp);
        float K, B;
        kbip_from(solref, solimp, imp, is_fric, &K, &B);
        g_R[(size_t)e * NEFC_MAX + r] = Rr;
        float vel;
        if (rt == ROW_FRICTION) vel = qvel[rd];
        else if (rt == ROW_LIMIT) vel = g_rowsign[(size_t)e * NEFC_MAX + r] * qvel[rd];
        else {
            vel = 0.0f;
            for (int i = 0; i < G1_NV; i++) vel += gcJ[rd * G1_NV + i] * qvel[i];
        }
        g_aref[(size_t)e * NEFC_MAX + r] = -B * vel - K * imp * rposv;
    }
    __syncwarp();
    if (lane == 0) {
        for (int r = nefc - 1; r >= 0; ) {
            if (g_rowtype[(size_t)e * NEFC_MAX + r] != ROW_CONTACT) { r--; continue; }
            int c = -1 - g_rowstash[(size_t)e * NEFC_MAX + r];
            int p = cpair[c];
            if (g1c_pair_dim[p] > 1) {
                int r0 = r - 3;   // rows r0..r0+3 are this contact's pyramid
                float R0 = g_R[(size_t)e * NEFC_MAX + r0];
                float R1 = R0 / SOL_IMPRATIO;
                float mu = g1c_pair_friction[5 * p] * sqrtf(R1 / R0);
                float Rpy = 2.0f * mu * mu * R0;
                for (int k = 0; k < 4; k++) g_R[(size_t)e * NEFC_MAX + r0 + k] = Rpy;
                r = r0 - 1;
            } else {
                r--;
            }
        }
    }
    __syncwarp();
    for (int r = lane; r < nefc; r += 32)
        g_D[(size_t)e * NEFC_MAX + r] = 1.0f / g_R[(size_t)e * NEFC_MAX + r];
    if (lane == 0) {
        g_ncon[e] = cnt[0];
        g_nefc[e] = nefc;
        for (int c = 0; c < NCON_MAX; c++)
            g_condist[(size_t)e * NCON_MAX + c] = c < cnt[0] ? cdist[c] : 1e9f;
    }
}

// ---------------------------------------------------------------------------
// K6: warm-start selection + Newton init (jaref, force/state, qfc, cost)
// ---------------------------------------------------------------------------
__global__ void k6_wsinit(int n, const float* __restrict__ g_qM,
                          const float* __restrict__ g_qfs,   // qfrc_smooth
                          const float* __restrict__ g_qas,   // qacc_smooth
                          const float* __restrict__ g_ws,
                          const int* __restrict__ g_nefc,
                          const int* __restrict__ g_rowtype,
                          const int* __restrict__ g_rowdof,
                          const float* __restrict__ g_rowsign,
                          const float* __restrict__ g_D, const float* __restrict__ g_R,
                          const float* __restrict__ g_aref,
                          const float* __restrict__ g_cJ,
                          float* __restrict__ g_qacc, float* __restrict__ g_Ma,
                          float* __restrict__ g_jaref, float* __restrict__ g_force,
                          int* __restrict__ g_state, float* __restrict__ g_qfc,
                          float* __restrict__ g_scal,
                          int* __restrict__ g_hvalid) {
    __shared__ float s_qacc[SWARPS][G1_NV];
    __shared__ float s_Ma[SWARPS][G1_NV];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* qacc = s_qacc[warp];
    float* Ma = s_Ma[warp];
    int nefc = g_nefc[e];
    const float* qM = g_qM + (size_t)e * G1_NM;
    const float* cJ = g_cJ + (size_t)e * NCROW_MAX * G1_NV;
    const int* rt = g_rowtype + (size_t)e * NEFC_MAX;
    const int* rd = g_rowdof + (size_t)e * NEFC_MAX;
    const float* rs = g_rowsign + (size_t)e * NEFC_MAX;
    const float* D = g_D + (size_t)e * NEFC_MAX;
    const float* R = g_R + (size_t)e * NEFC_MAX;
    const float* aref = g_aref + (size_t)e * NEFC_MAX;
    float* jaref = g_jaref + (size_t)e * NEFC_MAX;
    float* force = g_force + (size_t)e * NEFC_MAX;
    int* state = g_state + (size_t)e * NEFC_MAX;
    float* scal = g_scal + (size_t)e * NSCAL;

    if (nefc == 0) {
        for (int i = lane; i < G1_NV; i += 32) {
            g_qacc[(size_t)e * G1_NV + i] = g_qas[(size_t)e * G1_NV + i];
            g_qfc[(size_t)e * G1_NV + i] = 0.0f;
        }
        if (lane == 0) scal[SC_DONE] = 1.0f;
        return;
    }

    // cost(ws) with Gauss vs cost(qacc_smooth)
    const float* ws = g_ws + (size_t)e * G1_NV;
    const float* qas = g_qas + (size_t)e * G1_NV;
    float cost_ws = 0.0f, cost_sm = 0.0f;
    for (int r = lane; r < nefc; r += 32) {
        float jw = row_dot_g(rt[r], rd[r], rs[r], cJ, ws) - aref[r];
        float js = row_dot_g(rt[r], rd[r], rs[r], cJ, qas) - aref[r];
        // friction Huber / nonneg costs (same as constraint_update_g, cost only)
        if (rt[r] == ROW_FRICTION) {
            float fl = g1c_dof_frictionloss[rd[r]], Rf = R[r] * fl;
            cost_ws += (jw <= -Rf) ? (-0.5f*Rf*fl - fl*jw)
                     : (jw >= Rf) ? (-0.5f*Rf*fl + fl*jw) : 0.5f*D[r]*jw*jw;
            cost_sm += (js <= -Rf) ? (-0.5f*Rf*fl - fl*js)
                     : (js >= Rf) ? (-0.5f*Rf*fl + fl*js) : 0.5f*D[r]*js*js;
        } else {
            cost_ws += (jw < 0.0f) ? 0.5f*D[r]*jw*jw : 0.0f;
            cost_sm += (js < 0.0f) ? 0.5f*D[r]*js*js : 0.0f;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        cost_ws += __shfl_xor_sync(0xffffffff, cost_ws, o);
        cost_sm += __shfl_xor_sync(0xffffffff, cost_sm, o);
    }
    mul_M_vec_g(qM, ws, Ma, lane);
    float g = 0.0f;
    for (int i = lane; i < G1_NV; i += 32)
        g += 0.5f * (Ma[i] - g_qfs[(size_t)e * G1_NV + i]) * (ws[i] - qas[i]);
    for (int o = 16; o > 0; o >>= 1) g += __shfl_xor_sync(0xffffffff, g, o);
    cost_ws += g;
    int use_ws = cost_ws <= cost_sm;
    for (int i = lane; i < G1_NV; i += 32) qacc[i] = use_ws ? ws[i] : qas[i];
    __syncwarp();

    // Newton init at qacc
    mul_M_vec_g(qM, qacc, Ma, lane);
    for (int r = lane; r < nefc; r += 32)
        jaref[r] = row_dot_g(rt[r], rd[r], rs[r], cJ, qacc) - aref[r];
    __syncwarp();
    float cost = constraint_update_g(nefc, rt, rd, D, R, jaref, force, state, lane);
    jt_force_g(nefc, rt, rd, rs, cJ, force, g_qfc + (size_t)e * G1_NV, lane);
    float gauss = 0.0f;
    for (int i = lane; i < G1_NV; i += 32)
        gauss += 0.5f * (Ma[i] - g_qfs[(size_t)e * G1_NV + i]) * (qacc[i] - qas[i]);
    for (int o = 16; o > 0; o >>= 1) gauss += __shfl_xor_sync(0xffffffff, gauss, o);
    for (int i = lane; i < G1_NV; i += 32) {
        g_qacc[(size_t)e * G1_NV + i] = qacc[i];
        g_Ma[(size_t)e * G1_NV + i] = Ma[i];
    }
    if (lane == 0) {
        scal[SC_COST] = cost + gauss;
        scal[SC_GAUSS] = gauss;
        scal[SC_DONE] = 0.0f;
        if (g_hvalid) g_hvalid[e] = 0;  // new substep -> H stale (null = memo off)
    }
}

// ---------------------------------------------------------------------------
// K7: H = M + sum_quad D J'J -> Cholesky -> L (global)
// ---------------------------------------------------------------------------
__global__ void k7_hessian(int n, const float* __restrict__ g_qM,
                           const int* __restrict__ g_nefc,
                           const int* __restrict__ g_rowtype,
                           const int* __restrict__ g_rowdof,
                           const float* __restrict__ g_D,
                           const int* __restrict__ g_state,
                           const float* __restrict__ g_cJ,
                           const float* __restrict__ g_scal,
                           float* __restrict__ g_H,
                           int* __restrict__ g_hvalid) {
    __shared__ float s_H[SWARPS][G1_TRI];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    if (g_scal[(size_t)e * NSCAL + SC_DONE] != 0.0f) return;
    // H depends ONLY on the constraint active-set pattern (M, J, D fixed
    // within a substep). If no row changed state since the last build, the
    // factored H in global memory is bit-identical -> skip rebuild+factor.
    // g_hvalid == nullptr disables memoization (large batches: the active-set
    // read-back in k10 costs more than the skipped rebuilds save).
    if (g_hvalid && g_hvalid[e]) return;
    float* H = s_H[warp];
    const int nv = G1_NV;
    const float* qM = g_qM + (size_t)e * G1_NM;
    int nefc = g_nefc[e];
    const int* rt = g_rowtype + (size_t)e * NEFC_MAX;
    const int* rd = g_rowdof + (size_t)e * NEFC_MAX;
    const float* D = g_D + (size_t)e * NEFC_MAX;
    const int* state = g_state + (size_t)e * NEFC_MAX;
    const float* cJ = g_cJ + (size_t)e * NCROW_MAX * G1_NV;

    for (int k = lane; k < G1_TRI; k += 32) H[k] = 0.0f;
    __syncwarp();
    for (int i = lane; i < nv; i += 32) {
        int adr = g1c_dof_Madr[i];
        H[tridx(i, i)] = qM[adr++];
        for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j])
            H[tridx(i, j)] = qM[adr++];
    }
    __syncwarp();
    for (int r = lane; r < nefc; r += 32) {
        if (state[r] != ST_QUAD || rt[r] == ROW_CONTACT) continue;
        atomicAdd(&H[tridx(rd[r], rd[r])], D[r]);
    }
    __syncwarp();
    for (int r = 0; r < nefc; r++) {
        if (state[r] != ST_QUAD || rt[r] != ROW_CONTACT) continue;
        float Dr = D[r];
        const float* J = cJ + rd[r] * G1_NV;
        for (int i = lane; i < nv; i += 32) {
            float Ji = J[i];
            if (Ji == 0.0f) continue;
            float DJi = Dr * Ji;
            for (int j = 0; j <= i; j++) H[tridx(i, j)] += DJi * J[j];
        }
    }
    __syncwarp();
    for (int k = 0; k < nv; k++) {
        if (lane == 0) H[tridx(k, k)] = sqrtf(H[tridx(k, k)]);
        __syncwarp();
        float invd = 1.0f / H[tridx(k, k)];
        for (int i = k + 1 + lane; i < nv; i += 32) H[tridx(i, k)] *= invd;
        __syncwarp();
        for (int i = k + 1 + lane; i < nv; i += 32) {
            float Lik = H[tridx(i, k)];
            for (int j = k + 1; j <= i; j++) H[tridx(i, j)] -= Lik * H[tridx(j, k)];
        }
        __syncwarp();
    }
    for (int k = lane; k < G1_TRI; k += 32)
        g_H[(size_t)e * (size_t)G1_TRI + k] = H[k];
    if (lane == 0 && g_hvalid) g_hvalid[e] = 1;
}

// ---------------------------------------------------------------------------
// K8: grad -> Mgrad (L solve) -> search, Mv, jv, quadGauss, snorm
// ---------------------------------------------------------------------------
__global__ void k8_solvesearch(int n, const float* __restrict__ g_qM,
                               const float* __restrict__ g_qfs,
                               const float* __restrict__ g_Ma,
                               const float* __restrict__ g_qfc,
                               const float* __restrict__ g_H,
                               const int* __restrict__ g_nefc,
                               const int* __restrict__ g_rowtype,
                               const int* __restrict__ g_rowdof,
                               const float* __restrict__ g_rowsign,
                               const float* __restrict__ g_cJ,
                               float* __restrict__ g_search, float* __restrict__ g_Mv,
                               float* __restrict__ g_jv, float* __restrict__ g_scal) {
    __shared__ float s_L[SWARPS][G1_TRI];
    __shared__ float s_x[SWARPS][G1_NV];
    __shared__ float s_srch[SWARPS][G1_NV];
    __shared__ float s_Mv[SWARPS][G1_NV];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* scal = g_scal + (size_t)e * NSCAL;
    if (scal[SC_DONE] != 0.0f) return;
    float* L = s_L[warp];
    float* x = s_x[warp];
    float* srch = s_srch[warp];
    float* Mv = s_Mv[warp];
    const int nv = G1_NV;

    for (int k = lane; k < G1_TRI; k += 32)
        L[k] = g_H[(size_t)e * (size_t)G1_TRI + k];
    for (int i = lane; i < nv; i += 32)
        x[i] = g_Ma[(size_t)e * nv + i] - g_qfs[(size_t)e * nv + i]
             - g_qfc[(size_t)e * nv + i];
    __syncwarp();
    // x <- (L L')^-1 x, column sweeps
    for (int j = 0; j < nv; j++) {
        if (lane == 0) x[j] /= L[tridx(j, j)];
        __syncwarp();
        float xj = x[j];
        for (int i = j + 1 + lane; i < nv; i += 32) x[i] -= L[tridx(i, j)] * xj;
        __syncwarp();
    }
    for (int j = nv - 1; j >= 0; j--) {
        if (lane == 0) x[j] /= L[tridx(j, j)];
        __syncwarp();
        float xj = x[j];
        for (int i = lane; i < j; i += 32) x[i] -= L[tridx(j, i)] * xj;
        __syncwarp();
    }
    for (int i = lane; i < nv; i += 32) srch[i] = -x[i];
    __syncwarp();
    mul_M_vec_g(g_qM + (size_t)e * G1_NM, srch, Mv, lane);

    float snorm = 0.0f, qg1 = 0.0f, qg2 = 0.0f;
    for (int i = lane; i < nv; i += 32) {
        snorm += srch[i] * srch[i];
        qg1 += srch[i] * g_Ma[(size_t)e * nv + i]
             - g_qfs[(size_t)e * nv + i] * srch[i];
        qg2 += 0.5f * srch[i] * Mv[i];
    }
    for (int o = 16; o > 0; o >>= 1) {
        snorm += __shfl_xor_sync(0xffffffff, snorm, o);
        qg1 += __shfl_xor_sync(0xffffffff, qg1, o);
        qg2 += __shfl_xor_sync(0xffffffff, qg2, o);
    }
    int nefc = g_nefc[e];
    const int* rt = g_rowtype + (size_t)e * NEFC_MAX;
    const int* rd = g_rowdof + (size_t)e * NEFC_MAX;
    const float* rs = g_rowsign + (size_t)e * NEFC_MAX;
    const float* cJ = g_cJ + (size_t)e * NCROW_MAX * G1_NV;
    for (int r = lane; r < nefc; r += 32)
        g_jv[(size_t)e * NEFC_MAX + r] = row_dot_g(rt[r], rd[r], rs[r], cJ, srch);
    for (int i = lane; i < nv; i += 32) {
        g_search[(size_t)e * nv + i] = srch[i];
        g_Mv[(size_t)e * nv + i] = Mv[i];
    }
    if (lane == 0) {
        scal[SC_QG0] = scal[SC_GAUSS];
        scal[SC_QG1] = qg1;
        scal[SC_QG2] = qg2;
        scal[SC_SNORM] = sqrtf(snorm);
    }
}

// ---------------------------------------------------------------------------
// K9: linesearch (exact PrimalSearch; lane-parallel evals, uniform bracketing)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void staged_eval(int nefc, const int* rt, const int* rd,
                                            const float* D, const float* R,
                                            const float* jaref, const float* jv,
                                            float qg0, float qg1, float qg2,
                                            float alpha, float* cost, float* d0,
                                            float* d1) {
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f;
    for (int r = threadIdx.x % 32; r < nefc; r += 32) {
        float ja = jaref[r], jvr = jv[r], Dr = D[r];
        if (rt[r] == ROW_FRICTION) {
            float x = ja + alpha * jvr;
            float fl = g1c_dof_frictionloss[rd[r]], Rf = R[r] * fl;
            if (-Rf < x && x < Rf) {
                float DJ = Dr * ja;
                t0 += 0.5f * ja * DJ; t1 += jvr * DJ; t2 += 0.5f * jvr * Dr * jvr;
            }
            else if (x <= -Rf) { t0 += fl * (-0.5f*Rf - ja); t1 += -fl * jvr; }
            else { t0 += fl * (-0.5f*Rf + ja); t1 += fl * jvr; }
        } else {
            if (ja + alpha * jvr < 0.0f) {
                float DJ = Dr * ja;
                t0 += 0.5f * ja * DJ; t1 += jvr * DJ; t2 += 0.5f * jvr * Dr * jvr;
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        t0 += __shfl_xor_sync(0xffffffff, t0, o);
        t1 += __shfl_xor_sync(0xffffffff, t1, o);
        t2 += __shfl_xor_sync(0xffffffff, t2, o);
    }
    t0 += qg0; t1 += qg1; t2 += qg2;
    *cost = alpha*alpha*t2 + alpha*t1 + t0;
    *d0 = 2.0f*alpha*t2 + t1;
    *d1 = fmaxf(2.0f*t2, 1e-15f);
}

__global__ void k9_linesearch(int n, const int* __restrict__ g_nefc,
                              const int* __restrict__ g_rowtype,
                              const int* __restrict__ g_rowdof,
                              const float* __restrict__ g_D,
                              const float* __restrict__ g_R,
                              const float* __restrict__ g_jaref,
                              const float* __restrict__ g_jv,
                              float* __restrict__ g_scal) {
    __shared__ float s_ja[SWARPS][NEFC_MAX];
    __shared__ float s_jv[SWARPS][NEFC_MAX];
    __shared__ float s_D[SWARPS][NEFC_MAX];
    __shared__ float s_R[SWARPS][NEFC_MAX];
    __shared__ int s_rt[SWARPS][NEFC_MAX];
    __shared__ int s_rd[SWARPS][NEFC_MAX];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* scal = g_scal + (size_t)e * NSCAL;
    if (scal[SC_DONE] != 0.0f) return;
    int nefc = g_nefc[e];
    for (int r = lane; r < nefc; r += 32) {
        s_ja[warp][r] = g_jaref[(size_t)e * NEFC_MAX + r];
        s_jv[warp][r] = g_jv[(size_t)e * NEFC_MAX + r];
        s_D[warp][r] = g_D[(size_t)e * NEFC_MAX + r];
        s_R[warp][r] = g_R[(size_t)e * NEFC_MAX + r];
        s_rt[warp][r] = g_rowtype[(size_t)e * NEFC_MAX + r];
        s_rd[warp][r] = g_rowdof[(size_t)e * NEFC_MAX + r];
    }
    __syncwarp();
    float snorm = scal[SC_SNORM];
    float qg0 = scal[SC_QG0], qg1 = scal[SC_QG1], qg2 = scal[SC_QG2];
    if (snorm < 1e-15f) { if (lane == 0) scal[SC_ALPHA] = 0.0f; return; }
    float scale = 1.0f / (g1c_meaninertia * (float)G1_NV);
    float gtol = SOL_TOL * SOL_LS_TOL * snorm / scale;
    const int* rt = s_rt[warp];
    const int* rd = s_rd[warp];
    const float* D = s_D[warp];
    const float* R = s_R[warp];
    const float* ja = s_ja[warp];
    const float* jv = s_jv[warp];

#define EV(a, c, d0v, d1v) staged_eval(nefc, rt, rd, D, R, ja, jv, qg0, qg1, qg2, a, c, d0v, d1v)
    float a_p0 = 0.0f, c_p0, d0_p0, d1_p0;
    EV(a_p0, &c_p0, &d0_p0, &d1_p0);
    int ls_iter = 1;
    float a_p1 = a_p0 - d0_p0 / d1_p0, c_p1, d0_p1, d1_p1;
    EV(a_p1, &c_p1, &d0_p1, &d1_p1);
    ls_iter++;
    float alpha = 0.0f;
    int done = 0;
    if (fabsf(d0_p1) < gtol) { alpha = a_p1; done = 1; }
    if (!done) {
        int dir = d0_p1 < 0.0f ? 1 : -1;
        float a_p2 = a_p0, c_p2 = c_p0, d0_p2 = d0_p0, d1_p2 = d1_p0;
        while (d0_p1 * dir <= -gtol && ls_iter < SOL_LS_ITER) {
            a_p2 = a_p1; c_p2 = c_p1; d0_p2 = d0_p1; d1_p2 = d1_p1;
            a_p1 = a_p1 - d0_p1 / d1_p1;
            EV(a_p1, &c_p1, &d0_p1, &d1_p1);
            ls_iter++;
            if (fabsf(d0_p1) < gtol) { alpha = a_p1; done = 1; break; }
        }
        if (!done && ls_iter >= SOL_LS_ITER) { alpha = a_p1; done = 1; }
        if (!done) {
            float a_n1, c_n1, d0_n1, d1_n1;
            float a_n2 = a_p1, c_n2 = c_p1, d0_n2 = d0_p1, d1_n2 = d1_p1;
            a_n1 = a_p1 - d0_p1 / d1_p1;
            EV(a_n1, &c_n1, &d0_n1, &d1_n1);
            ls_iter++;
            while (!done && ls_iter < SOL_LS_ITER) {
                float a_m = 0.5f * (a_p1 + a_p2), c_m, d0_m, d1_m;
                EV(a_m, &c_m, &d0_m, &d1_m);
                ls_iter++;
                float ca[3] = {a_n1, a_n2, a_m}, cc[3] = {c_n1, c_n2, c_m};
                float cd0[3] = {d0_n1, d0_n2, d0_m}, cd1[3] = {d1_n1, d1_n2, d1_m};
                int best = -1;
                for (int q = 0; q < 3; q++)
                    if (fabsf(cd0[q]) < gtol && (best == -1 || cc[q] < cc[best])) best = q;
                if (best >= 0) { alpha = ca[best]; done = 1; break; }
                int b1 = 0, b2 = 0;
                for (int q = 0; q < 3; q++) {
                    if (d0_p1 < 0 && cd0[q] < 0 && d0_p1 < cd0[q]) {
                        a_p1 = ca[q]; c_p1 = cc[q]; d0_p1 = cd0[q]; d1_p1 = cd1[q]; b1 = 1;
                    } else if (d0_p1 > 0 && cd0[q] > 0 && d0_p1 > cd0[q]) {
                        a_p1 = ca[q]; c_p1 = cc[q]; d0_p1 = cd0[q]; d1_p1 = cd1[q]; b1 = 2;
                    }
                }
                if (b1) {
                    a_n1 = a_p1 - d0_p1 / d1_p1;
                    EV(a_n1, &c_n1, &d0_n1, &d1_n1);
                    ls_iter++;
                }
                for (int q = 0; q < 3; q++) {
                    if (d0_p2 < 0 && cd0[q] < 0 && d0_p2 < cd0[q]) {
                        a_p2 = ca[q]; c_p2 = cc[q]; d0_p2 = cd0[q]; d1_p2 = cd1[q]; b2 = 1;
                    } else if (d0_p2 > 0 && cd0[q] > 0 && d0_p2 > cd0[q]) {
                        a_p2 = ca[q]; c_p2 = cc[q]; d0_p2 = cd0[q]; d1_p2 = cd1[q]; b2 = 2;
                    }
                }
                if (b2) {
                    a_n2 = a_p2 - d0_p2 / d1_p2;
                    EV(a_n2, &c_n2, &d0_n2, &d1_n2);
                    ls_iter++;
                }
                if (!b1 && !b2) { alpha = a_m; done = 1; break; }
            }
            if (!done) {
                if (c_p1 <= c_p2 && c_p1 < c_p0) alpha = a_p1;
                else if (c_p2 <= c_p1 && c_p2 < c_p0) alpha = a_p2;
                else alpha = 0.0f;
            }
        }
    }
#undef EV
    if (lane == 0) scal[SC_ALPHA] = alpha;
}

// ---------------------------------------------------------------------------
// K10: move + constraint update + qfc + gauss + termination
// ---------------------------------------------------------------------------
__global__ void k10_update(int n, const float* __restrict__ g_qfs,
                           const float* __restrict__ g_qas,
                           const int* __restrict__ g_nefc,
                           const int* __restrict__ g_rowtype,
                           const int* __restrict__ g_rowdof,
                           const float* __restrict__ g_rowsign,
                           const float* __restrict__ g_D, const float* __restrict__ g_R,
                           const float* __restrict__ g_cJ,
                           const float* __restrict__ g_search,
                           const float* __restrict__ g_Mv,
                           const float* __restrict__ g_jv,
                           float* __restrict__ g_qacc, float* __restrict__ g_Ma,
                           float* __restrict__ g_jaref, float* __restrict__ g_force,
                           int* __restrict__ g_state, float* __restrict__ g_qfc,
                           float* __restrict__ g_scal,
                           int* __restrict__ g_hvalid) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    float* scal = g_scal + (size_t)e * NSCAL;
    if (scal[SC_DONE] != 0.0f) return;
    float alpha = scal[SC_ALPHA];
    if (alpha == 0.0f) { if (lane == 0) scal[SC_DONE] = 1.0f; return; }
    const int nv = G1_NV;
    int nefc = g_nefc[e];
    for (int i = lane; i < nv; i += 32) {
        g_qacc[(size_t)e * nv + i] += alpha * g_search[(size_t)e * nv + i];
        g_Ma[(size_t)e * nv + i] += alpha * g_Mv[(size_t)e * nv + i];
    }
    for (int r = lane; r < nefc; r += 32)
        g_jaref[(size_t)e * NEFC_MAX + r] += alpha * g_jv[(size_t)e * NEFC_MAX + r];
    __syncwarp();
    const int* rt = g_rowtype + (size_t)e * NEFC_MAX;
    const int* rd = g_rowdof + (size_t)e * NEFC_MAX;
    const float* rs = g_rowsign + (size_t)e * NEFC_MAX;
    const float* cJ = g_cJ + (size_t)e * NCROW_MAX * G1_NV;
    int hch = 0;
    float cost = constraint_update_g(nefc, rt, rd,
                                     g_D + (size_t)e * NEFC_MAX,
                                     g_R + (size_t)e * NEFC_MAX,
                                     g_jaref + (size_t)e * NEFC_MAX,
                                     g_force + (size_t)e * NEFC_MAX,
                                     g_state + (size_t)e * NEFC_MAX, lane,
                                     g_hvalid ? &hch : nullptr);
    jt_force_g(nefc, rt, rd, rs, cJ, g_force + (size_t)e * NEFC_MAX,
               g_qfc + (size_t)e * nv, lane);
    float gauss = 0.0f, gn = 0.0f;
    for (int i = lane; i < nv; i += 32) {
        float ma = g_Ma[(size_t)e * nv + i];
        float qf = g_qfs[(size_t)e * nv + i];
        gauss += 0.5f * (ma - qf) * (g_qacc[(size_t)e * nv + i] - g_qas[(size_t)e * nv + i]);
        float gi = ma - qf - g_qfc[(size_t)e * nv + i];
        gn += gi * gi;
    }
    for (int o = 16; o > 0; o >>= 1) {
        gauss += __shfl_xor_sync(0xffffffff, gauss, o);
        gn += __shfl_xor_sync(0xffffffff, gn, o);
    }
    if (lane == 0) {
        float newcost = cost + gauss;
        float scale = 1.0f / (g1c_meaninertia * (float)nv);
        float improvement = scale * (scal[SC_COST] - newcost);
        float gradient = scale * sqrtf(gn);
        scal[SC_COST] = newcost;
        scal[SC_GAUSS] = gauss;
        if (improvement < SOL_TOL || gradient < SOL_TOL) scal[SC_DONE] = 1.0f;
        if (g_hvalid && hch) g_hvalid[e] = 0;  // active set flipped -> H stale
    }
}

// expand sparse qM to dense for the host compare
__global__ void k_expand_qM(int n, const float* __restrict__ g_qM,
                            float* __restrict__ g_qMd) {
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int e = blockIdx.x * SWARPS + warp;
    if (e >= n) return;
    const float* qM = g_qM + (size_t)e * G1_NM;
    float* dq = g_qMd + (size_t)e * G1_NV * G1_NV;
    for (int k = lane; k < G1_NV * G1_NV; k += 32) dq[k] = 0.0f;
    __syncwarp();
    for (int i = lane; i < G1_NV; i += 32) {
        int adr = g1c_dof_Madr[i];
        dq[i * G1_NV + i] = qM[adr++];
        for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j]) {
            float v = qM[adr++];
            dq[i * G1_NV + j] = v;
            dq[j * G1_NV + i] = v;
        }
    }
}

// ---------------------------------------------------------------------------
