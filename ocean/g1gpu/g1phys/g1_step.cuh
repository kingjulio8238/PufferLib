// g1_step.cuh — shared device engine for the G1-specialized step.
// EnvShared + FK + full smooth pipeline (validated: GATE-B 2026-06-11,
// 1000-step in-air rollout err 1.6e-5) split as smooth_forward/euler_advance
// so contact.cu can insert the constraint solve between them.
#pragma once

#include "robot_topology.cuh"
#include "common.cuh"

#define WARPS_PER_BLOCK_SMOOTH 4

// ---------------------------------------------------------------------------
// per-env shared-memory state (~10.4 KB)
// ---------------------------------------------------------------------------
struct EnvShared {
    float qpos[G1_NQ], qvel[G1_NV], ctrl[G1_NU];
    float xpos[G1_NBODY * 3], xquat[G1_NBODY * 4];
    float xipos[G1_NBODY * 3];
    float com[3];                       // subtree_com of the root (single tree)
    float cinert[G1_NBODY * 10];
    float crb[G1_NBODY * 10];
    float cdof[G1_NV * 6];
    float cvel[G1_NBODY * 6];           // overwritten by cfrc_body in RNE
    float cacc[G1_NBODY * 6];
    float cdof_dot[G1_NV * 6];
    float qM[G1_NM], qLD[G1_NM], qLDiagInv[G1_NV];
    float qfrc_bias[G1_NV], qfrc_passive[G1_NV], qfrc_smooth[G1_NV];
    float qacc[G1_NV];
    float act_force[G1_NU];     // clamped actuator force (last forward pass;
                                // the env reward's torque penalty source)
};

// ---------------------------------------------------------------------------
// FK core for one body (identical to the Gate-A-validated proto_mapping.cu)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fk_body(int i, const float* q,
                                        float* xpos, float* xquat) {
    int jntadr = g1c_body_jntadr[i];
    int jntnum = g1c_body_jntnum[i];

    float xp[3], xq[4];
    if (jntnum == 1 && g1c_jnt_type[jntadr] == 0 /*mjJNT_FREE*/) {
        int qadr = g1c_jnt_qposadr[jntadr];
        xp[0] = q[qadr]; xp[1] = q[qadr + 1]; xp[2] = q[qadr + 2];
        xq[0] = q[qadr + 3]; xq[1] = q[qadr + 4]; xq[2] = q[qadr + 5]; xq[3] = q[qadr + 6];
        quat_normalize(xq);
    } else {
        int pid = g1c_body_parentid[i];
        const float* bp = g1c_body_pos + 3 * i;
        const float* bq = g1c_body_quat + 4 * i;
        if (pid) {
            rot_vec_quat(xp, bp, xquat + 4 * pid);
            xp[0] += xpos[3 * pid]; xp[1] += xpos[3 * pid + 1]; xp[2] += xpos[3 * pid + 2];
            quat_mul(xq, xquat + 4 * pid, bq);
        } else {
            xp[0] = bp[0]; xp[1] = bp[1]; xp[2] = bp[2];
            xq[0] = bq[0]; xq[1] = bq[1]; xq[2] = bq[2]; xq[3] = bq[3];
        }
        for (int j = 0; j < jntnum; j++) {
            int jid = jntadr + j;
            int qadr = g1c_jnt_qposadr[jid];
            float xanchor[3];
            rot_vec_quat(xanchor, g1c_jnt_pos + 3 * jid, xq);
            xanchor[0] += xp[0]; xanchor[1] += xp[1]; xanchor[2] += xp[2];

            float qloc[4];
            axis_angle_quat(qloc, g1c_jnt_axis + 3 * jid, q[qadr] - g1c_qpos0[qadr]);
            float xq2[4];
            quat_mul(xq2, xq, qloc);
            xq[0] = xq2[0]; xq[1] = xq2[1]; xq[2] = xq2[2]; xq[3] = xq2[3];

            float vec[3];
            rot_vec_quat(vec, g1c_jnt_pos + 3 * jid, xq);
            xp[0] = xanchor[0] - vec[0];
            xp[1] = xanchor[1] - vec[1];
            xp[2] = xanchor[2] - vec[2];
        }
    }
    quat_normalize(xq);
    xpos[3 * i] = xp[0]; xpos[3 * i + 1] = xp[1]; xpos[3 * i + 2] = xp[2];
    xquat[4 * i] = xq[0]; xquat[4 * i + 1] = xq[1];
    xquat[4 * i + 2] = xq[2]; xquat[4 * i + 3] = xq[3];
}

// ---------------------------------------------------------------------------
// one full smooth step on shared state S (warp-cooperative, lane in [0,32))
// ---------------------------------------------------------------------------
__device__ void smooth_forward(EnvShared* S, int lane) {
    // ---- FK (level-synchronized) ----
    if (lane == 0) {
        S->xpos[0] = S->xpos[1] = S->xpos[2] = 0.0f;
        S->xquat[0] = 1.0f; S->xquat[1] = S->xquat[2] = S->xquat[3] = 0.0f;
    }
    __syncwarp();
    for (int lv = 0; lv < G1_NLEVELS; lv++) {
        int off = g1c_level_offset[lv], cnt = g1c_level_offset[lv + 1] - off;
        if (lane < cnt) fk_body(g1c_level_bodies[off + lane], S->qpos, S->xpos, S->xquat);
        __syncwarp();
    }

    // ---- xipos + root-subtree COM (single tree: com = sum(m_i*xipos_i)/M) ----
    float part[3] = {0.0f, 0.0f, 0.0f};
    if (lane >= 1 && lane < G1_NBODY) {
        int i = lane;
        rot_vec_quat(S->xipos + 3 * i, g1c_body_ipos + 3 * i, S->xquat + 4 * i);
        S->xipos[3*i]   += S->xpos[3*i];
        S->xipos[3*i+1] += S->xpos[3*i+1];
        S->xipos[3*i+2] += S->xpos[3*i+2];
        float mass = g1c_body_mass[i];
        part[0] = mass * S->xipos[3*i];
        part[1] = mass * S->xipos[3*i+1];
        part[2] = mass * S->xipos[3*i+2];
    }
    for (int o = 16; o > 0; o >>= 1) {
        part[0] += __shfl_down_sync(0xffffffff, part[0], o);
        part[1] += __shfl_down_sync(0xffffffff, part[1], o);
        part[2] += __shfl_down_sync(0xffffffff, part[2], o);
    }
    if (lane == 0) {
        float inv = 1.0f / g1c_body_subtreemass[1];
        S->com[0] = part[0] * inv; S->com[1] = part[1] * inv; S->com[2] = part[2] * inv;
    }
    __syncwarp();

    // ---- cinert: body inertia in the com-centered frame ----
    if (lane >= 1 && lane < G1_NBODY) {
        int i = lane;
        float iq[4], mat[9], dif[3];
        quat_mul(iq, S->xquat + 4 * i, g1c_body_iquat + 4 * i);
        quat2mat(mat, iq);
        dif[0] = S->xipos[3*i]   - S->com[0];
        dif[1] = S->xipos[3*i+1] - S->com[1];
        dif[2] = S->xipos[3*i+2] - S->com[2];
        inert_com(S->cinert + 10 * i, g1c_body_inertia + 3 * i, mat, dif,
                  g1c_body_mass[i]);
    }

    // ---- cdof ----
    if (lane == 0) {
        // free joint (joint 0, body 1): 3 translations then 3 rotations
        int da = g1c_jnt_dofadr[0];
        float mat1[9];
        quat2mat(mat1, S->xquat + 4);
        for (int k = 0; k < 18; k++) S->cdof[6 * da + k] = 0.0f;
        S->cdof[6*(da+0) + 3] = 1.0f;
        S->cdof[6*(da+1) + 4] = 1.0f;
        S->cdof[6*(da+2) + 5] = 1.0f;
        float off[3] = {S->com[0] - S->xpos[3], S->com[1] - S->xpos[4],
                        S->com[2] - S->xpos[5]};  // xanchor_free = xpos[body 1]
        for (int k = 0; k < 3; k++) {
            float axis[3] = {mat1[k], mat1[k + 3], mat1[k + 6]};  // column k
            float* cd = S->cdof + 6 * (da + 3 + k);
            cd[0] = axis[0]; cd[1] = axis[1]; cd[2] = axis[2];
            cross3(cd + 3, axis, off);
        }
    } else if (lane < G1_NJNT) {
        // hinges: axis invariant under own-axis rotation -> use final body quat
        int j = lane;
        int b = g1c_jnt_bodyid[j];
        int da = g1c_jnt_dofadr[j];
        float axis[3], anchor[3];
        rot_vec_quat(axis, g1c_jnt_axis + 3 * j, S->xquat + 4 * b);
        rot_vec_quat(anchor, g1c_jnt_pos + 3 * j, S->xquat + 4 * b);
        float off[3] = {S->com[0] - (anchor[0] + S->xpos[3*b]),
                        S->com[1] - (anchor[1] + S->xpos[3*b+1]),
                        S->com[2] - (anchor[2] + S->xpos[3*b+2])};
        float* cd = S->cdof + 6 * da;
        cd[0] = axis[0]; cd[1] = axis[1]; cd[2] = axis[2];
        cross3(cd + 3, axis, off);
    }
    __syncwarp();

    // ---- CRB: crb = cinert, backward accumulate (lanes = the 10 components) ----
    for (int k = lane; k < G1_NBODY * 10; k += 32) S->crb[k] = S->cinert[k];
    __syncwarp();
    if (lane < 10) {
        for (int i = G1_NBODY - 1; i >= 1; i--) {
            int p = g1c_body_parentid[i];
            if (p > 0) S->crb[10 * p + lane] += S->crb[10 * i + lane];
        }
    }
    __syncwarp();

    // ---- qM (legacy layout: row i = [M(i,i), M(i,parent), ...] at dof_Madr) ----
    for (int i = lane; i < G1_NV; i += 32) {
        float buf[6];
        mul_inert_vec(buf, S->crb + 10 * g1c_dof_bodyid[i], S->cdof + 6 * i);
        int adr = g1c_dof_Madr[i];
        S->qM[adr++] = g1c_dof_armature[i] + dot6(S->cdof + 6 * i, buf);
        for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j]) {
            S->qM[adr++] = dot6(S->cdof + 6 * j, buf);
        }
    }
    __syncwarp();

    // ---- LDL^T factor (mj_factorI_legacy; k serial, ancestors lane-parallel) ----
    for (int k = lane; k < G1_NM; k += 32) S->qLD[k] = S->qM[k];
    __syncwarp();
    for (int k = G1_NV - 1; k >= 0; k--) {
        int Madr_kk = g1c_dof_Madr[k];
        int nanc = (k < G1_NV - 1 ? g1c_dof_Madr[k + 1] : G1_NM) - Madr_kk - 1;
        int i = (lane < nanc) ? g1c_dof_chain[k * G1_MAX_CHAIN + lane] : -1;
        // phase A: read-only tmp per ancestor
        float tmp = 0.0f;
        if (i >= 0) tmp = S->qLD[Madr_kk + 1 + lane] / S->qLD[Madr_kk];
        __syncwarp();
        // phase B: row_i -= rowk[from i's position] * tmp (rows i disjoint, row k untouched)
        if (i >= 0) {
            int cnt = g1c_dof_Madr[i + 1] - g1c_dof_Madr[i];  // ancestors => i < nv-1
            for (int c = 0; c < cnt; c++)
                S->qLD[g1c_dof_Madr[i] + c] -= S->qLD[Madr_kk + 1 + lane + c] * tmp;
        }
        __syncwarp();
        // phase C: scale row k entries
        if (i >= 0) S->qLD[Madr_kk + 1 + lane] = tmp;
        __syncwarp();
    }
    for (int i = lane; i < G1_NV; i += 32)
        S->qLDiagInv[i] = 1.0f / S->qLD[g1c_dof_Madr[i]];
    __syncwarp();

    // ---- comVel: cvel + cdof_dot (level-synchronized forward pass) ----
    if (lane == 0) for (int k = 0; k < 6; k++) S->cvel[k] = 0.0f;
    __syncwarp();
    for (int lv = 0; lv < G1_NLEVELS; lv++) {
        int off = g1c_level_offset[lv], cnt = g1c_level_offset[lv + 1] - off;
        if (lane < cnt) {
            int i = g1c_level_bodies[off + lane];
            int bda = g1c_body_dofadr[i], dofnum = g1c_body_dofnum[i];
            float v[6];
            for (int k = 0; k < 6; k++) v[k] = S->cvel[6 * g1c_body_parentid[i] + k];
            if (dofnum == 6) {  // free joint body
                for (int k = 0; k < 3; k++)
                    for (int c = 0; c < 6; c++)
                        v[c] += S->cdof[6 * (bda + k) + c] * S->qvel[bda + k];
                for (int k = 0; k < 3; k++) {
                    for (int c = 0; c < 6; c++) S->cdof_dot[6*(bda+k) + c] = 0.0f;
                    cross_motion(S->cdof_dot + 6 * (bda + 3 + k), v,
                                 S->cdof + 6 * (bda + 3 + k));
                }
                for (int k = 0; k < 3; k++)
                    for (int c = 0; c < 6; c++)
                        v[c] += S->cdof[6 * (bda + 3 + k) + c] * S->qvel[bda + 3 + k];
            } else {
                for (int k = 0; k < dofnum; k++) {
                    cross_motion(S->cdof_dot + 6 * (bda + k), v, S->cdof + 6 * (bda + k));
                    for (int c = 0; c < 6; c++)
                        v[c] += S->cdof[6 * (bda + k) + c] * S->qvel[bda + k];
                }
            }
            for (int c = 0; c < 6; c++) S->cvel[6 * i + c] = v[c];
        }
        __syncwarp();
    }

    // ---- RNE (flg_acc=0): cacc forward, cfrc (into cvel), backward, project ----
    if (lane == 0) {
        for (int k = 0; k < 5; k++) S->cacc[k] = 0.0f;
        S->cacc[5] = -(G1_GRAVITY_Z);   // world cacc = -gravity
    }
    __syncwarp();
    for (int lv = 0; lv < G1_NLEVELS; lv++) {
        int off = g1c_level_offset[lv], cnt = g1c_level_offset[lv + 1] - off;
        if (lane < cnt) {
            int i = g1c_level_bodies[off + lane];
            int bda = g1c_body_dofadr[i], dofnum = g1c_body_dofnum[i];
            float a[6];
            for (int c = 0; c < 6; c++) {
                a[c] = S->cacc[6 * g1c_body_parentid[i] + c];
                for (int k = 0; k < dofnum; k++)
                    a[c] += S->cdof_dot[6 * (bda + k) + c] * S->qvel[bda + k];
                S->cacc[6 * i + c] = a[c];
            }
            // cfrc_body = cinert*cacc + cvel x* (cinert*cvel)  (overwrites cvel)
            float f1[6], iv[6], f2[6];
            mul_inert_vec(f1, S->cinert + 10 * i, a);
            mul_inert_vec(iv, S->cinert + 10 * i, S->cvel + 6 * i);
            cross_force(f2, S->cvel + 6 * i, iv);
            for (int c = 0; c < 6; c++) S->cvel[6 * i + c] = f1[c] + f2[c];
        }
        __syncwarp();
    }
    if (lane < 6) {  // backward accumulate cfrc to parents (component-parallel)
        for (int i = G1_NBODY - 1; i >= 1; i--) {
            int p = g1c_body_parentid[i];
            if (p) S->cvel[6 * p + lane] += S->cvel[6 * i + lane];
        }
    }
    __syncwarp();
    for (int i = lane; i < G1_NV; i += 32)
        S->qfrc_bias[i] = dot6(S->cdof + 6 * i, S->cvel + 6 * g1c_dof_bodyid[i]);
    __syncwarp();

    // ---- passive + bias fusion ----
    for (int i = lane; i < G1_NV; i += 32) {
        // sim2real plant: leg dofs (6..17) are damped ONLY by the PD controller (kd),
        // matching the proven unitree_rl_gym pipeline (zero mechanical joint damping on
        // actuated legs). -DG1_LEGACY_DAMPING restores the original baked damping (the
        // plant the <=v3 / sub-60 records trained on). See docs/sim2real.md (D1).
        float dmp = g1c_dof_damping[i];
#ifndef G1_LEGACY_DAMPING
        if (i >= 6 && i < 18) dmp = 0.0f;
#endif
        S->qfrc_passive[i] = -dmp * S->qvel[i];
        S->qfrc_smooth[i] = S->qfrc_passive[i] - S->qfrc_bias[i];
    }
    __syncwarp();

    // ---- PD actuation (gear=1 joint transmission; one actuator per hinge) ----
    if (lane < G1_NU) {
        int a = lane;
        int j = g1c_act_jntid[a];
        int padr = g1c_jnt_qposadr[j], dadr = g1c_jnt_dofadr[j];
        float c = S->ctrl[a];
        float lo = g1c_act_ctrlrange[2 * a], hi = g1c_act_ctrlrange[2 * a + 1];
        c = c < lo ? lo : (c > hi ? hi : c);
        float force = g1c_act_gain0[a] * c + g1c_act_bias1[a] * S->qpos[padr]
                      + g1c_act_bias2[a] * S->qvel[dadr];
        if (g1c_act_forcelimited[a]) {   // per-actuator clamp (Go2 motors ~±24)
            float flo = g1c_act_forcerange[2 * a], fhi = g1c_act_forcerange[2 * a + 1];
            force = force < flo ? flo : (force > fhi ? fhi : force);
        }
        if (g1c_jnt_actfrclimited[j]) {
            float flo = g1c_jnt_actfrcrange[2 * j], fhi = g1c_jnt_actfrcrange[2 * j + 1];
            force = force < flo ? flo : (force > fhi ? fhi : force);
        }
        S->act_force[a] = force;        // = actuator_force (gear 1)
        S->qfrc_smooth[dadr] += force;  // dofs unique per actuator
    }
    __syncwarp();

    // ---- qacc_smooth = LDL solve (mj_solveLD_legacy; lane 0 serial v1) ----
    if (lane == 0) {
        float* x = S->qacc;
        for (int i = 0; i < G1_NV; i++) x[i] = S->qfrc_smooth[i];
        for (int i = G1_NV - 1; i >= 0; i--) {
            if (x[i] != 0.0f) {
                int adr = g1c_dof_Madr[i] + 1;
                for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j])
                    x[j] -= S->qLD[adr++] * x[i];
            }
        }
        for (int i = 0; i < G1_NV; i++) x[i] *= S->qLDiagInv[i];
        for (int i = 0; i < G1_NV; i++) {
            int adr = g1c_dof_Madr[i] + 1;
            for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j])
                x[i] -= S->qLD[adr++] * x[j];
        }
    }
    __syncwarp();

}

// advance state with the given qacc (explicit Euler + quatIntegrate;
// EULERDAMP disabled in the wall model)
__device__ void euler_advance(EnvShared* S, int lane, const float* qacc) {
    // ---- explicit Euler (EULERDAMP disabled in the wall model) ----
    for (int i = lane; i < G1_NV; i += 32)
        S->qvel[i] += G1_DT * qacc[i];
    __syncwarp();
    if (lane == 0) {
        // free joint: semi-implicit position update + local-frame quat integrate
        S->qpos[0] += G1_DT * S->qvel[0];
        S->qpos[1] += G1_DT * S->qvel[1];
        S->qpos[2] += G1_DT * S->qvel[2];
        quat_integrate(S->qpos + 3, S->qvel + 3, G1_DT);
    } else if (lane < G1_NJNT) {
        int j = lane;
        S->qpos[g1c_jnt_qposadr[j]] += G1_DT * S->qvel[g1c_jnt_dofadr[j]];
    }
    __syncwarp();
}

// full smooth-only step (Workstream B semantics)
__device__ __forceinline__ void smooth_step(EnvShared* S, int lane) {
    smooth_forward(S, lane);
    euler_advance(S, lane, S->qacc);
}
