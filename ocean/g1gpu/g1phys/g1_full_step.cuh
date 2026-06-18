// g1_full_step.cuh — FULL G1 physics step device engine (Gate-C validated;
// D1-optimized: 3.40M steps/s H100 @32768). Shared by contact.cu (physics
// validation/bench) and env.cu (episode machinery + GATE-D).
// Define WARPS_PER_BLOCK and include g1_step.cuh deps before this header.
#pragma once

#include "g1_step.cuh"

// fixed-shape efc bounds — emitted per-robot by gen_robot.py (G1: 20/44/106;
// Go2: 23/92/108). Computed from the contact candidate set + condim.
#define NCON_MAX G1_NCON_MAX
#define NCROW_MAX G1_NCROW_MAX
#define NEFC_MAX G1_NEFC_MAX

// solver options (the wall; mirrored from the mjb in gen, hardcoded here)
#ifndef SOL_ITER
#define SOL_ITER 3
#endif
#ifndef SOL_LS_ITER
#define SOL_LS_ITER 5
#endif
#define SOL_TOL 1e-8f
#define SOL_LS_TOL 0.01f
// pyramidal-cone R scale = model opt.impratio; emitted per-robot by gen_robot
// (G1: 1.0, Go2: 100.0). Hardcoding 1.0 silently broke any robot with
// impratio != 1 (the Go2 quadruped: D came out 100x too small -> wrong solve).
#ifndef SOL_IMPRATIO
#ifdef G1_IMPRATIO
#define SOL_IMPRATIO G1_IMPRATIO
#else
#define SOL_IMPRATIO 1.0f
#endif
#endif

__device__ float g1c_meaninertia;   // m->stat.meaninertia (set by host)
__device__ float g1c_dof_invweight0[G1_NV];
__device__ float g1c_body_invweight0_t[G1_NBODY];  // translation component
__device__ float g1c_dof_solref[G1_NV * 2];
__device__ float g1c_dof_solimp[G1_NV * 5];
__device__ float g1c_jnt_solref[G1_NJNT * 2];
__device__ float g1c_jnt_solimp[G1_NJNT * 5];
__device__ float g1c_jnt_range[G1_NJNT * 2];
__device__ int   g1c_jnt_limited[G1_NJNT];

// row types
#define ROW_FRICTION 0
#define ROW_LIMIT 1
#define ROW_CONTACT 2
// constraint states
#define ST_QUAD 0
#define ST_SAT 1
#define ST_LINNEG 2
#define ST_LINPOS 3

// ---------------------------------------------------------------------------
// constraint-phase shared state (per env)
// ---------------------------------------------------------------------------
struct ConShared {
    // contacts from narrowphase
    int ncon;
    int con_pair[NCON_MAX];
    float con_dist[NCON_MAX];
    float con_pos[NCON_MAX * 3];
    float con_norm[NCON_MAX * 3];
    // contact-row Jacobians (dense over nv)
    int ncrow;
    float cJ[NCROW_MAX * G1_NV];
    // efc rows
    int nefc, nf, nl;
    int row_type[NEFC_MAX];
    int row_dof[NEFC_MAX];     // friction/limit: dof; contact: index into cJ
    float row_sign[NEFC_MAX];  // limit: -side; else unused
    float pos[NEFC_MAX];       // efc_pos (margin = 0 for every G1 row)
    float D[NEFC_MAX], R[NEFC_MAX], aref[NEFC_MAX];
    float jaref[NEFC_MAX], jv[NEFC_MAX], force[NEFC_MAX];
    int state[NEFC_MAX], oldstate[NEFC_MAX];
    // solver vectors
    float Ma[G1_NV], grad[G1_NV], Mgrad[G1_NV], search[G1_NV], Mv[G1_NV];
    float qacc_s[G1_NV];       // saved qacc_smooth (S->qacc gets overwritten)
    float gauss, cost;
    float quadg0, quadg1, quadg2;
    int boxbox_flag;           // box-box bounding spheres overlapped (unhandled)
};

// dense Hessian overlay: cinert..cdof_dot = 310+310+210+186+186+210 = 1412 >= 1225
__device__ __forceinline__ float* hessian_ptr(EnvShared* S) {
    return S->cinert;
}

// ---------------------------------------------------------------------------
// device narrowphase (3.9.0-exact; mirrors scripts/oracle_contact.py)
// ---------------------------------------------------------------------------
__device__ void geom_pose(const float* xpos, const float* xquat, int g,
                          float gpos[3], float gmat[9]) {
    int b = g1c_geom_bodyid[g];
    rot_vec_quat(gpos, g1c_geom_pos + 3 * g, xquat + 4 * b);
    gpos[0] += xpos[3 * b]; gpos[1] += xpos[3 * b + 1]; gpos[2] += xpos[3 * b + 2];
    float q[4];
    quat_mul(q, xquat + 4 * b, g1c_geom_quat + 4 * g);
    quat2mat(gmat, q);
}

__device__ __forceinline__ void make_frame(float fr[9], const float n[3]) {
    // mju_makeFrame with tangent undefined
    fr[0] = n[0]; fr[1] = n[1]; fr[2] = n[2];
    float nn = sqrtf(fr[0]*fr[0] + fr[1]*fr[1] + fr[2]*fr[2]);
    fr[0] /= nn; fr[1] /= nn; fr[2] /= nn;
    float y[3] = {0.0f, 0.0f, 0.0f};
    if (fr[1] < 0.5f && fr[1] > -0.5f) y[1] = 1.0f; else y[2] = 1.0f;
    float dot = fr[0]*y[0] + fr[1]*y[1] + fr[2]*y[2];
    y[0] -= fr[0]*dot; y[1] -= fr[1]*dot; y[2] -= fr[2]*dot;
    float yn = sqrtf(y[0]*y[0] + y[1]*y[1] + y[2]*y[2]);
    fr[3] = y[0]/yn; fr[4] = y[1]/yn; fr[5] = y[2]/yn;
    cross3(fr + 6, fr, fr + 3);
}

// plane(g1) x box(g2): writes up to 4 contacts at out slots, returns count
__device__ int plane_box_np(const float* xpos, const float* xquat, int g1, int g2,
                            float* dist, float* pos, float* norm_out) {
    float p1[3], m1[9], p2[3], m2[9];
    geom_pose(xpos, xquat, g1, p1, m1);
    geom_pose(xpos, xquat, g2, p2, m2);
    const float* sz = g1c_geom_size + 3 * g2;
    float norm[3] = {m1[2], m1[5], m1[8]};
    float dif[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    float d0 = dif[0]*norm[0] + dif[1]*norm[1] + dif[2]*norm[2];
    int cnt = 0;
    for (int i = 0; i < 8 && cnt < 4; i++) {
        float v[3] = {(i&1) ? sz[0] : -sz[0], (i&2) ? sz[1] : -sz[1], (i&4) ? sz[2] : -sz[2]};
        float c[3] = {m2[0]*v[0] + m2[1]*v[1] + m2[2]*v[2],
                      m2[3]*v[0] + m2[4]*v[1] + m2[5]*v[2],
                      m2[6]*v[0] + m2[7]*v[1] + m2[8]*v[2]};
        float ldist = norm[0]*c[0] + norm[1]*c[1] + norm[2]*c[2];
        if (d0 + ldist > 0.0f || ldist > 0.0f) continue;   // margin = 0
        float cd = d0 + ldist;
        if (cd >= 0.0f) continue;                           // includemargin = 0
        dist[cnt] = cd;
        pos[3*cnt+0] = c[0] + p2[0] - norm[0]*cd*0.5f;
        pos[3*cnt+1] = c[1] + p2[1] - norm[1]*cd*0.5f;
        pos[3*cnt+2] = c[2] + p2[2] - norm[2]*cd*0.5f;
        norm_out[3*cnt+0] = norm[0]; norm_out[3*cnt+1] = norm[1]; norm_out[3*cnt+2] = norm[2];
        cnt++;
    }
    return cnt;
}

// plane(g1) x sphere(g2): one contact (mirrors oracle_contact.plane_sphere).
__device__ int plane_sphere_np(const float* xpos, const float* xquat, int g1, int g2,
                               float* dist, float* pos, float* norm_out) {
    float p1[3], m1[9], p2[3], m2[9];
    geom_pose(xpos, xquat, g1, p1, m1);
    geom_pose(xpos, xquat, g2, p2, m2);
    float r = g1c_geom_size[3 * g2];                  // sphere radius
    float norm[3] = {m1[2], m1[5], m1[8]};            // plane z-axis
    float dif[3] = {p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]};
    float cd = dif[0]*norm[0] + dif[1]*norm[1] + dif[2]*norm[2] - r;
    if (cd >= 0.0f) return 0;                          // includemargin = 0
    dist[0] = cd;
    pos[0] = p2[0] - norm[0]*(r + 0.5f*cd);
    pos[1] = p2[1] - norm[1]*(r + 0.5f*cd);
    pos[2] = p2[2] - norm[2]*(r + 0.5f*cd);
    norm_out[0] = norm[0]; norm_out[1] = norm[1]; norm_out[2] = norm[2];
    return 1;
}

// sphere-sphere core used by capsule-capsule (radii r1, r2 at points c1, c2)
__device__ int sphere_sphere_np(const float c1[3], float r1, const float c2[3], float r2,
                                float* dist, float* pos, float* norm_out) {
    float dif[3] = {c1[0]-c2[0], c1[1]-c2[1], c1[2]-c2[2]};
    float d2 = dif[0]*dif[0] + dif[1]*dif[1] + dif[2]*dif[2];
    float mind = r1 + r2;                                   // margin = 0
    if (d2 > mind*mind) return 0;
    float len = sqrtf(d2);
    float cd = len - r1 - r2;
    if (cd >= 0.0f) return 0;                               // includemargin = 0
    float n[3] = {c2[0]-c1[0], c2[1]-c1[1], c2[2]-c1[2]};
    if (len >= 1e-15f) { n[0] /= len; n[1] /= len; n[2] /= len; }
    else { n[0] = 1.0f; n[1] = n[2] = 0.0f; }               // degenerate (never hits)
    dist[0] = cd;
    pos[0] = c1[0] + n[0]*(r1 + cd*0.5f);
    pos[1] = c1[1] + n[1]*(r1 + cd*0.5f);
    pos[2] = c1[2] + n[2]*(r1 + cd*0.5f);
    norm_out[0] = n[0]; norm_out[1] = n[1]; norm_out[2] = n[2];
    return 1;
}

__device__ int capsule_capsule_np(const float* xpos, const float* xquat, int g1, int g2,
                                  float* dist, float* pos, float* norm_out) {
    float p1[3], m1[9], p2[3], m2[9];
    geom_pose(xpos, xquat, g1, p1, m1);
    geom_pose(xpos, xquat, g2, p2, m2);
    const float* s1 = g1c_geom_size + 3 * g1;
    const float* s2 = g1c_geom_size + 3 * g2;
    float a1[3] = {m1[2]*s1[1], m1[5]*s1[1], m1[8]*s1[1]};
    float a2[3] = {m2[2]*s2[1], m2[5]*s2[1], m2[8]*s2[1]};
    float dif[3] = {p1[0]-p2[0], p1[1]-p2[1], p1[2]-p2[2]};
    float ma = a1[0]*a1[0] + a1[1]*a1[1] + a1[2]*a1[2];
    float mb = -(a1[0]*a2[0] + a1[1]*a2[1] + a1[2]*a2[2]);
    float mc = a2[0]*a2[0] + a2[1]*a2[1] + a2[2]*a2[2];
    float u = -(a1[0]*dif[0] + a1[1]*dif[1] + a1[2]*dif[2]);
    float v = a2[0]*dif[0] + a2[1]*dif[1] + a2[2]*dif[2];
    float det = ma*mc - mb*mb;
    if (fabsf(det) >= 1e-15f) {
        float x1 = (mc*u - mb*v) / det;
        float x2 = (ma*v - mb*u) / det;
        if (x1 > 1.0f) { x1 = 1.0f; x2 = (v - mb) / mc; }
        else if (x1 < -1.0f) { x1 = -1.0f; x2 = (v + mb) / mc; }
        if (x2 > 1.0f) { x2 = 1.0f; x1 = fminf(fmaxf((u - mb)/ma, -1.0f), 1.0f); }
        else if (x2 < -1.0f) { x2 = -1.0f; x1 = fminf(fmaxf((u + mb)/ma, -1.0f), 1.0f); }
        float c1[3] = {p1[0]+a1[0]*x1, p1[1]+a1[1]*x1, p1[2]+a1[2]*x1};
        float c2[3] = {p2[0]+a2[0]*x2, p2[1]+a2[1]*x2, p2[2]+a2[2]*x2};
        return sphere_sphere_np(c1, s1[0], c2, s2[0], dist, pos, norm_out);
    }
    // parallel axes: up to two endpoint contacts (rare; mirrors C exactly)
    int n = 0;
    for (int pass = 0; pass < 4 && n < 2; pass++) {
        float c1[3], c2[3];
        if (pass < 2) {
            float x1 = pass == 0 ? 1.0f : -1.0f;
            float x2 = fminf(fmaxf((v - mb*x1)/mc, -1.0f), 1.0f);
            c1[0] = p1[0]+a1[0]*x1; c1[1] = p1[1]+a1[1]*x1; c1[2] = p1[2]+a1[2]*x1;
            c2[0] = p2[0]+a2[0]*x2; c2[1] = p2[1]+a2[1]*x2; c2[2] = p2[2]+a2[2]*x2;
        } else {
            if (pass == 2 && n >= 2) break;
            float x2 = pass == 2 ? 1.0f : -1.0f;
            float x1 = fminf(fmaxf((u - mb*x2)/ma, -1.0f), 1.0f);
            c1[0] = p1[0]+a1[0]*x1; c1[1] = p1[1]+a1[1]*x1; c1[2] = p1[2]+a1[2]*x1;
            c2[0] = p2[0]+a2[0]*x2; c2[1] = p2[1]+a2[1]*x2; c2[2] = p2[2]+a2[2]*x2;
        }
        n += sphere_sphere_np(c1, s1[0], c2, s2[0], dist + n, pos + 3*n, norm_out + 3*n);
    }
    return n;
}

// ---------------------------------------------------------------------------
// impedance (getimpedance, 3.9.0; margin = 0 for all G1 rows)
// ---------------------------------------------------------------------------
__device__ float impedance(const float* solimp, float pos) {
    float i0 = fminf(0.9999f, fmaxf(0.0001f, solimp[0]));
    float i1 = fminf(0.9999f, fmaxf(0.0001f, solimp[1]));
    float width = fmaxf(0.0f, solimp[2]);
    float mid = fminf(0.9999f, fmaxf(0.0001f, solimp[3]));
    float pw = fmaxf(1.0f, solimp[4]);
    if (i0 == i1 || width <= 1e-15f) return 0.5f * (i0 + i1);
    float x = fabsf(pos / width);
    if (x >= 1.0f) return i1;
    if (x <= 0.0f) return i0;
    float y;
    if (pw == 1.0f) y = x;
    else if (x <= mid) y = powf(x, pw) / powf(mid, pw - 1.0f);
    else y = 1.0f - powf(1.0f - x, pw) / powf(1.0f - mid, pw - 1.0f);
    return i0 + y * (i1 - i0);
}

__device__ __forceinline__ void kbip_from(const float* solref_in, const float* solimp,
                                          float imp, int is_friction,
                                          float* K, float* B) {
    float r0 = solref_in[0], r1 = solref_in[1];
    if (r0 > 0.0f) r0 = fmaxf(r0, 2.0f * G1_DT);  // refsafe
    float i1 = fminf(0.9999f, fmaxf(0.0001f, solimp[1]));
    if (is_friction) *K = 0.0f;
    else if (r0 > 0.0f) *K = 1.0f / fmaxf(1e-15f, i1*i1 * r0*r0 * r1*r1);
    else *K = -r0 / fmaxf(1e-15f, i1*i1);
    if (r1 > 0.0f) *B = 2.0f / fmaxf(1e-15f, i1 * r0);
    else *B = -r1 / fmaxf(1e-15f, i1);
}

// ---------------------------------------------------------------------------
// J ops with the hybrid row structure (friction/limit implicit, contact dense)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float row_dot(ConShared* C, int r, const float* v) {
    if (C->row_type[r] == ROW_FRICTION) return v[C->row_dof[r]];
    if (C->row_type[r] == ROW_LIMIT) return C->row_sign[r] * v[C->row_dof[r]];
    const float* J = C->cJ + C->row_dof[r] * G1_NV;
    float s = 0.0f;
    for (int i = 0; i < G1_NV; i++) s += J[i] * v[i];
    return s;
}

// qfrc[i] += sum_r J[r][i] * force[r], lanes parallel over i
__device__ void jt_force(ConShared* C, const float* force, float* qfrc, int lane) {
    for (int i = lane; i < G1_NV; i += 32) {
        float s = 0.0f;
        for (int r = 0; r < C->nefc; r++) {
            float f = force[r];
            if (f == 0.0f) continue;
            if (C->row_type[r] == ROW_FRICTION) { if (C->row_dof[r] == i) s += f; }
            else if (C->row_type[r] == ROW_LIMIT) { if (C->row_dof[r] == i) s += C->row_sign[r] * f; }
            else s += C->cJ[C->row_dof[r] * G1_NV + i] * f;
        }
        qfrc[i] = s;
    }
}

// sparse symmetric M*v (legacy dof_Madr layout), lane-parallel and
// conflict-free: row j = diag + own ancestors (row j) + descendants
// (gathered via the codegen g1c_dof_desc* lists)
__device__ void mul_M_vec(EnvShared* S, const float* v, float* out, int lane) {
    for (int j = lane; j < G1_NV; j += 32) {
        int adr = g1c_dof_Madr[j];
        float s = S->qM[adr++] * v[j];
        for (int a = g1c_dof_parentid[j]; a >= 0; a = g1c_dof_parentid[a])
            s += S->qM[adr++] * v[a];
        for (int d = g1c_dof_descoff[j]; d < g1c_dof_descoff[j + 1]; d++)
            s += S->qM[g1c_dof_desc_adr[d]] * v[g1c_dof_desc_i[d]];
        out[j] = s;
    }
    __syncwarp();
}

// ---------------------------------------------------------------------------
// constraint update: forces/states/cost(jar) (constraintUpdate_impl)
// lane-parallel over rows; returns cost via shfl reduction (all lanes)
// ---------------------------------------------------------------------------
__device__ float constraint_update(ConShared* C, const float* jar, int lane) {
    float cost = 0.0f;
    for (int r = lane; r < C->nefc; r += 32) {
        float f = -C->D[r] * jar[r];
        int st = ST_QUAD;
        if (C->row_type[r] == ROW_FRICTION) {
            float fl = g1c_dof_frictionloss[C->row_dof[r]], Rf = C->R[r] * fl;
            if (jar[r] <= -Rf) { cost += -0.5f*Rf*fl - fl*jar[r]; f = fl; st = ST_LINNEG; }
            else if (jar[r] >= Rf) { cost += -0.5f*Rf*fl + fl*jar[r]; f = -fl; st = ST_LINPOS; }
            else cost += 0.5f * C->D[r] * jar[r] * jar[r];
        } else {
            if (jar[r] >= 0.0f) { f = 0.0f; st = ST_SAT; }
            else cost += 0.5f * C->D[r] * jar[r] * jar[r];
        }
        C->force[r] = f;
        C->state[r] = st;
    }
    for (int o = 16; o > 0; o >>= 1) cost += __shfl_xor_sync(0xffffffff, cost, o);
    return cost;
}

// ---------------------------------------------------------------------------
// linesearch eval (PrimalEval): returns cost,d0,d1 on all lanes
// ---------------------------------------------------------------------------
__device__ void primal_eval(ConShared* C, float alpha, float* cost, float* d0, float* d1) {
    float t0 = 0.0f, t1 = 0.0f, t2 = 0.0f;
    for (int r = threadIdx.x % 32; r < C->nefc; r += 32) {
        float ja = C->jaref[r], jvr = C->jv[r], Dr = C->D[r];
        if (C->row_type[r] == ROW_FRICTION) {
            float x = ja + alpha * jvr;
            float fl = g1c_dof_frictionloss[C->row_dof[r]], Rf = C->R[r] * fl;
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
    t0 += C->quadg0; t1 += C->quadg1; t2 += C->quadg2;
    *cost = alpha*alpha*t2 + alpha*t1 + t0;
    *d0 = 2.0f*alpha*t2 + t1;
    *d1 = fmaxf(2.0f*t2, 1e-15f);
}

// ---------------------------------------------------------------------------
// stage profiling (env 0, lane 0 only; ~zero overhead elsewhere)
// sections: 0 smooth | 1 assemble | 2 warmstart/init | 3 H build |
//           4 cholesky factor+solve | 5 linesearch | 6 newton misc | 7 euler
// ---------------------------------------------------------------------------
__device__ unsigned long long g_prof[8];
#define PROF_START(en) long long _pt = ((en) && lane == 0) ? clock64() : 0
#define PROF_MARK(en, slot) do { if ((en) && lane == 0) { \
    long long _now = clock64(); g_prof[slot] += (unsigned long long)(_now - _pt); \
    _pt = _now; } } while (0)
#define PROF_RESET(en) do { if ((en) && lane == 0) _pt = clock64(); } while (0)

// ---------------------------------------------------------------------------
// dense Hessian build + in-place Cholesky (H = M + sum_{QUADRATIC} D J'J)
// ---------------------------------------------------------------------------
__device__ void build_factor_H(EnvShared* S, ConShared* C, float* H, int lane) {
    const int nv = G1_NV;
    for (int k = lane; k < nv * nv; k += 32) H[k] = 0.0f;
    __syncwarp();
    // M from sparse: lanes over rows; mirror element (j,i) is uniquely owned
    // by row i, so the parallel fill is conflict-free
    for (int i = lane; i < nv; i += 32) {
        int adr = g1c_dof_Madr[i];
        H[i * nv + i] = S->qM[adr++];
        for (int j = g1c_dof_parentid[i]; j >= 0; j = g1c_dof_parentid[j]) {
            float v = S->qM[adr++];
            H[i * nv + j] = v;
            H[j * nv + i] = v;
        }
    }
    __syncwarp();
    // friction/limit quadratic rows: diagonal adds (atomic: a friction and a
    // limit row can share a dof)
    for (int r = lane; r < C->nefc; r += 32) {
        if (C->state[r] != ST_QUAD || C->row_type[r] == ROW_CONTACT) continue;
        atomicAdd(&H[C->row_dof[r] * nv + C->row_dof[r]], C->D[r]);  // sign^2 = 1
    }
    __syncwarp();
    // contact rows: serial over rows, lanes own H rows (no cross-lane hazard)
    for (int r = 0; r < C->nefc; r++) {
        if (C->state[r] != ST_QUAD || C->row_type[r] != ROW_CONTACT) continue;
        float Dr = C->D[r];
        const float* J = C->cJ + C->row_dof[r] * G1_NV;
        for (int i = lane; i < nv; i += 32) {
            float Ji = J[i];
            if (Ji == 0.0f) continue;
            float DJi = Dr * Ji;
            for (int j = 0; j < nv; j++) H[i * nv + j] += DJi * J[j];
        }
    }
    __syncwarp();
    // Cholesky in place (lower), right-looking, lanes own trailing rows.
    // (A flattened constant-table variant was 3x SLOWER: divergent constant
    // memory reads serialize 32-way. Keep row mapping.)
    for (int k = 0; k < nv; k++) {
        if (lane == 0) H[k * nv + k] = sqrtf(H[k * nv + k]);
        __syncwarp();
        float invd = 1.0f / H[k * nv + k];
        for (int i = k + 1 + lane; i < nv; i += 32) H[i * nv + k] *= invd;
        __syncwarp();
        for (int i = k + 1 + lane; i < nv; i += 32) {
            float Lik = H[i * nv + k];
            for (int j = k + 1; j <= i; j++) H[i * nv + j] -= Lik * H[j * nv + k];
        }
        __syncwarp();
    }
}

// rank-1 Cholesky update/downdate (mju_cholUpdate, engine_util_solve.c:96),
// k-serial with lane-parallel row updates. x is destroyed. Returns rank.
__device__ int chol_rank1(float* L, float* x, int lane, int flg_plus, int kstart) {
    const int nv = G1_NV;
    int rank = nv;
    for (int k = kstart; k < nv; k++) {
        float xk = x[k];
        if (xk == 0.0f) continue;
        float Lkk = L[k * nv + k];
        float tmp = Lkk * Lkk + (flg_plus ? xk * xk : -xk * xk);
        if (tmp < 1e-15f) { tmp = 1e-15f; rank--; }
        float r = sqrtf(tmp);
        float c = r / Lkk, cinv = 1.0f / c, s = xk / Lkk;
        if (lane == 0) L[k * nv + k] = r;
        for (int i = k + 1 + lane; i < nv; i += 32) {
            float m = L[i * nv + k];
            float mn = flg_plus ? (m + s * x[i]) * cinv : (m - s * x[i]) * cinv;
            L[i * nv + k] = mn;
            x[i] = c * x[i] - s * mn;
        }
        __syncwarp();
    }
    return rank;
}

// solve L L' y = b (y in C->Mgrad) via column sweeps: pure parallel AXPYs,
// no reductions — each j-step is one division + one element per lane
__device__ void chol_solve_par(ConShared* C, const float* L, const float* b, int lane) {
    const int nv = G1_NV;
    float* x = C->Mgrad;
    for (int i = lane; i < nv; i += 32) x[i] = b[i];
    __syncwarp();
    // forward: x <- L^-1 x, column sweep
    for (int j = 0; j < nv; j++) {
        if (lane == 0) x[j] /= L[j * nv + j];
        __syncwarp();
        float xj = x[j];
        for (int i = j + 1 + lane; i < nv; i += 32) x[i] -= L[i * nv + j] * xj;
        __syncwarp();
    }
    // backward: x <- L^-T x, row sweep on L (column of L')
    for (int j = nv - 1; j >= 0; j--) {
        if (lane == 0) x[j] /= L[j * nv + j];
        __syncwarp();
        float xj = x[j];
        for (int i = lane; i < j; i += 32) x[i] -= L[j * nv + i] * xj;
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// full constraint solve given smooth solution in S->qacc (= qacc_smooth)
// and warm-start in ws[nv]. Leaves final qacc in S->qacc, forces in C.
// ---------------------------------------------------------------------------
__device__ void newton_solve(EnvShared* S, ConShared* C, const float* ws, int lane,
                             int prof_en = 0, int trace = -1) {
    int nv = G1_NV;
    float* H = hessian_ptr(S);
    PROF_START(prof_en);

    // save qacc_smooth
    for (int i = lane; i < nv; i += 32) C->qacc_s[i] = S->qacc[i];
    __syncwarp();

    if (C->nefc == 0) return;  // qacc = qacc_smooth

    // ---- warm-start selection ----
    for (int r = lane; r < C->nefc; r += 32)
        C->jaref[r] = row_dot(C, r, ws) - C->aref[r];
    __syncwarp();
    float cost_ws = constraint_update(C, C->jaref, lane);
    mul_M_vec(S, ws, C->Ma, lane);
    float g = 0.0f;
    for (int i = lane; i < nv; i += 32)
        g += 0.5f * (C->Ma[i] - S->qfrc_smooth[i]) * (ws[i] - C->qacc_s[i]);
    for (int o = 16; o > 0; o >>= 1) g += __shfl_xor_sync(0xffffffff, g, o);
    cost_ws += g;
    for (int r = lane; r < C->nefc; r += 32)
        C->jv[r] = row_dot(C, r, C->qacc_s) - C->aref[r];  // jv as scratch: efc_b
    __syncwarp();
    float cost_sm = constraint_update(C, C->jv, lane);
    int use_ws = cost_ws <= cost_sm;
    for (int i = lane; i < nv; i += 32) S->qacc[i] = use_ws ? ws[i] : C->qacc_s[i];
    __syncwarp();

    // ---- Newton init ----
    mul_M_vec(S, S->qacc, C->Ma, lane);
    for (int r = lane; r < C->nefc; r += 32)
        C->jaref[r] = row_dot(C, r, S->qacc) - C->aref[r];
    __syncwarp();
    float cost = constraint_update(C, C->jaref, lane);
    jt_force(C, C->force, C->grad, lane);  // grad temp = qfrc_constraint
    __syncwarp();
    float gauss = 0.0f;
    for (int i = lane; i < nv; i += 32)
        gauss += 0.5f * (C->Ma[i] - S->qfrc_smooth[i]) * (S->qacc[i] - C->qacc_s[i]);
    for (int o = 16; o > 0; o >>= 1) gauss += __shfl_xor_sync(0xffffffff, gauss, o);
    cost += gauss;
    C->gauss = gauss;

    float scale = 1.0f / (g1c_meaninertia * (float)nv);
    PROF_MARK(prof_en, 2);

    // factor ONCE per step; the loop maintains it with rank-1 updates on
    // active-set changes (MuJoCo's HessianIncremental design)
    build_factor_H(S, C, H, lane);
    PROF_MARK(prof_en, 3);

    for (int iter = 0; iter < SOL_ITER; iter++) {
        // grad = Ma - qfrc_smooth - qfrc_constraint  (qfrc_constraint in grad)
        for (int i = lane; i < nv; i += 32)
            C->grad[i] = C->Ma[i] - S->qfrc_smooth[i] - C->grad[i];
        __syncwarp();

        // Mgrad = (L L')^-1 grad (lane-parallel substitutions)
        chol_solve_par(C, H, C->grad, lane);
        PROF_MARK(prof_en, 4);

        for (int i = lane; i < nv; i += 32) C->search[i] = -C->Mgrad[i];
        __syncwarp();

        // ---- linesearch (PrimalSearch, exact transcription) ----
        float snorm = 0.0f;
        for (int i = lane; i < nv; i += 32) snorm += C->search[i] * C->search[i];
        for (int o = 16; o > 0; o >>= 1) snorm += __shfl_xor_sync(0xffffffff, snorm, o);
        snorm = sqrtf(snorm);
        if (snorm < 1e-15f) {
            // restore qfrc_constraint into grad (it was overwritten by the gradient)
            jt_force(C, C->force, C->grad, lane);
            __syncwarp();
            break;
        }
        float gtol = SOL_TOL * SOL_LS_TOL * snorm / scale;

        mul_M_vec(S, C->search, C->Mv, lane);
        for (int r = lane; r < C->nefc; r += 32) C->jv[r] = row_dot(C, r, C->search);
        __syncwarp();
        // quadratics
        float qg1 = 0.0f, qg2 = 0.0f;
        for (int i = lane; i < nv; i += 32) {
            qg1 += C->search[i] * C->Ma[i] - S->qfrc_smooth[i] * C->search[i];
            qg2 += 0.5f * C->search[i] * C->Mv[i];
        }
        for (int o = 16; o > 0; o >>= 1) {
            qg1 += __shfl_xor_sync(0xffffffff, qg1, o);
            qg2 += __shfl_xor_sync(0xffffffff, qg2, o);
        }
        if (lane == 0) { C->quadg0 = C->gauss; C->quadg1 = qg1; C->quadg2 = qg2; }
        __syncwarp();

        // bracketing on uniform warp execution (all lanes run the same scalars)
        float a_p0 = 0.0f, c_p0, d0_p0, d1_p0;
        primal_eval(C, a_p0, &c_p0, &d0_p0, &d1_p0);
        int ls_iter = 1;
        float a_p1 = a_p0 - d0_p0 / d1_p0, c_p1, d0_p1, d1_p1;
        primal_eval(C, a_p1, &c_p1, &d0_p1, &d1_p1);
        ls_iter++;
        float alpha;
        int done = 0;
        if (fabsf(d0_p1) < gtol) { alpha = a_p1; done = 1; }
        if (!done) {
            int dir = d0_p1 < 0.0f ? 1 : -1;
            float a_p2 = a_p0, c_p2 = c_p0, d0_p2 = d0_p0, d1_p2 = d1_p0;
            while (d0_p1 * dir <= -gtol && ls_iter < SOL_LS_ITER) {
                a_p2 = a_p1; c_p2 = c_p1; d0_p2 = d0_p1; d1_p2 = d1_p1;
                a_p1 = a_p1 - d0_p1 / d1_p1;
                primal_eval(C, a_p1, &c_p1, &d0_p1, &d1_p1);
                ls_iter++;
                if (fabsf(d0_p1) < gtol) { alpha = a_p1; done = 1; break; }
            }
            if (!done && ls_iter >= SOL_LS_ITER) { alpha = a_p1; done = 1; }
            if (!done) {
                // bracketed search
                float a_n1, c_n1, d0_n1, d1_n1;   // p1next
                float a_n2 = a_p1, c_n2 = c_p1, d0_n2 = d0_p1, d1_n2 = d1_p1;  // p2next
                a_n1 = a_p1 - d0_p1 / d1_p1;
                primal_eval(C, a_n1, &c_n1, &d0_n1, &d1_n1);
                ls_iter++;
                while (!done && ls_iter < SOL_LS_ITER) {
                    float a_m = 0.5f * (a_p1 + a_p2), c_m, d0_m, d1_m;
                    primal_eval(C, a_m, &c_m, &d0_m, &d1_m);
                    ls_iter++;
                    // candidates: n1, n2, mid
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
                        primal_eval(C, a_n1, &c_n1, &d0_n1, &d1_n1);
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
                        primal_eval(C, a_n2, &c_n2, &d0_n2, &d1_n2);
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
        PROF_MARK(prof_en, 5);
        if (alpha == 0.0f) {
            // restore qfrc_constraint into grad before exiting
            jt_force(C, C->force, C->grad, lane);
            __syncwarp();
            break;
        }

        // ---- move ----
        for (int i = lane; i < nv; i += 32) {
            S->qacc[i] += alpha * C->search[i];
            C->Ma[i] += alpha * C->Mv[i];
        }
        for (int r = lane; r < C->nefc; r += 32) C->jaref[r] += alpha * C->jv[r];
        __syncwarp();

        float oldcost = cost;
        for (int r = lane; r < C->nefc; r += 32) C->oldstate[r] = C->state[r];
        __syncwarp();
        cost = constraint_update(C, C->jaref, lane);
        jt_force(C, C->force, C->grad, lane);  // qfrc_constraint into grad
        __syncwarp();

        // maintain the factorization: rank-1 update/downdate per state change
        // (QUADRATIC rows enter/leave H). ADAPTIVE: heavy active-set churn
        // makes a storm of rank-1s costlier than one rebuild — count first.
        if (iter + 1 < SOL_ITER) {  // last iteration never solves again
            int nchange = 0;
            for (int r = lane; r < C->nefc; r += 32)
                nchange += (C->oldstate[r] == ST_QUAD) != (C->state[r] == ST_QUAD);
            for (int o = 16; o > 0; o >>= 1)
                nchange += __shfl_xor_sync(0xffffffff, nchange, o);
            if (nchange > 6) {
                build_factor_H(S, C, H, lane);
            } else if (nchange) {
                int need_full = 0;
                for (int r = 0; r < C->nefc && !need_full; r++) {
                    int was = C->oldstate[r] == ST_QUAD, is = C->state[r] == ST_QUAD;
                    if (was == is) continue;
                    float sD = sqrtf(C->D[r]);
                    int kstart;
                    if (C->row_type[r] != ROW_CONTACT) {
                        kstart = C->row_dof[r];
                        for (int i = lane; i < nv; i += 32) C->Mv[i] = 0.0f;
                        __syncwarp();
                        if (lane == 0) C->Mv[kstart] = sD;  // |sign| = 1
                    } else {
                        kstart = 0;
                        const float* J = C->cJ + C->row_dof[r] * G1_NV;
                        for (int i = lane; i < nv; i += 32) C->Mv[i] = sD * J[i];
                    }
                    __syncwarp();
                    if (chol_rank1(H, C->Mv, lane, is, kstart) < nv) need_full = 1;
                }
                if (need_full) build_factor_H(S, C, H, lane);
            }
        }

        gauss = 0.0f;
        for (int i = lane; i < nv; i += 32)
            gauss += 0.5f * (C->Ma[i] - S->qfrc_smooth[i]) * (S->qacc[i] - C->qacc_s[i]);
        for (int o = 16; o > 0; o >>= 1) gauss += __shfl_xor_sync(0xffffffff, gauss, o);
        cost += gauss;
        C->gauss = gauss;

        // termination (grad computed at top of next iteration; compute norm here)
        float gn = 0.0f;
        for (int i = lane; i < nv; i += 32) {
            float gi = C->Ma[i] - S->qfrc_smooth[i] - C->grad[i];
            gn += gi * gi;
        }
        for (int o = 16; o > 0; o >>= 1) gn += __shfl_xor_sync(0xffffffff, gn, o);
        float improvement = scale * (oldcost - cost);
        float gradient = scale * sqrtf(gn);
        if (trace >= 0 && lane == 0) {
            int nact = 0;
            for (int r = 0; r < C->nefc; r++) nact += (C->state[r] == ST_QUAD);
            printf("TRACE iter=%d alpha=%.5g snorm=%.4g cost=%.6g oldcost=%.6g impr=%.3e grad=%.3e nact=%d/%d qacc3=%.3f qacc6=%.3f\n",
                   iter, alpha, snorm, cost, oldcost, improvement, gradient, nact, C->nefc, S->qacc[3], S->qacc[6]);
        }
        PROF_MARK(prof_en, 6);
        if (improvement < SOL_TOL || gradient < SOL_TOL) break;
    }
    // C->grad holds qfrc_constraint of the final state
}

// ---------------------------------------------------------------------------
// efc assembly after smooth_forward (narrowphase + rows + impedance + aref)
// ---------------------------------------------------------------------------
__device__ void assemble_constraints(EnvShared* S, ConShared* C, int lane) {
    // ---- narrowphase: lane 0 runs all 5 pairs serially (tiny) ----
    if (lane == 0) {
        C->boxbox_flag = 0;
        int n = 0;
        for (int p = 0; p < G1_NPAIR; p++) {
            int g1 = g1c_pair_geom1[p], g2 = g1c_pair_geom2[p];
            int t1 = g1c_geom_type[g1], t2 = g1c_geom_type[g2];
            // per-pair capacity (plane-box 4, capsule-capsule 2): skip ONLY
            // this pair if it cannot fit — never abort the whole loop
            int need = (t1 == 0 && t2 == 6) ? 4 : (t1 == 0 && t2 == 2) ? 1 : 2;
            if (n + need > NCON_MAX) continue;
            int cnt = 0;
            if (t1 == 0 /*PLANE*/ && t2 == 6 /*BOX*/) {
                cnt = plane_box_np(S->xpos, S->xquat, g1, g2, C->con_dist + n, C->con_pos + 3*n,
                                   C->con_norm + 3*n);
            } else if (t1 == 0 /*PLANE*/ && t2 == 2 /*SPHERE*/) {
                cnt = plane_sphere_np(S->xpos, S->xquat, g1, g2, C->con_dist + n, C->con_pos + 3*n,
                                      C->con_norm + 3*n);
            } else if (t1 == 3 /*CAPSULE*/ && t2 == 3) {
                cnt = capsule_capsule_np(S->xpos, S->xquat, g1, g2, C->con_dist + n, C->con_pos + 3*n,
                                         C->con_norm + 3*n);
            } else if (t1 == 6 && t2 == 6) {
                // box-box not transcribed: bounding-sphere detect-and-flag
                float p1[3], m1[9], p2[3], m2[9];
                geom_pose(S->xpos, S->xquat, g1, p1, m1);
                geom_pose(S->xpos, S->xquat, g2, p2, m2);
                const float* s1 = g1c_geom_size + 3 * g1;
                const float* s2 = g1c_geom_size + 3 * g2;
                float rb1 = sqrtf(s1[0]*s1[0] + s1[1]*s1[1] + s1[2]*s1[2]);
                float rb2 = sqrtf(s2[0]*s2[0] + s2[1]*s2[1] + s2[2]*s2[2]);
                float dx = p1[0]-p2[0], dy = p1[1]-p2[1], dz = p1[2]-p2[2];
                if (sqrtf(dx*dx + dy*dy + dz*dz) < rb1 + rb2) C->boxbox_flag = 1;
            }
            for (int c = 0; c < cnt; c++) C->con_pair[n + c] = p;
            n += cnt;
        }
        C->ncon = n;
    }
    __syncwarp();

    // ---- efc rows ----
    if (lane == 0) {
        int r = 0;
        // friction rows (dofs with frictionloss; G1: all 29 hinges = dofs 6..34)
        for (int i = 0; i < G1_NV; i++) {
            if (g1c_dof_frictionloss[i] == 0.0f) continue;
            C->row_type[r] = ROW_FRICTION;
            C->row_dof[r] = i;
            C->row_sign[r] = 1.0f;
            C->pos[r] = 0.0f;
            r++;
        }
        C->nf = r;
        // joint limits
        for (int j = 0; j < G1_NJNT; j++) {
            if (!g1c_jnt_limited[j]) continue;
            float value = S->qpos[g1c_jnt_qposadr[j]];
            for (int side = -1; side <= 1; side += 2) {
                float lim = g1c_jnt_range[2*j + (side + 1)/2];
                float dist = side * (lim - value);
                if (dist < 0.0f) {  // jnt_margin = 0
                    C->row_type[r] = ROW_LIMIT;
                    C->row_dof[r] = g1c_jnt_dofadr[j];
                    C->row_sign[r] = -(float)side;
                    C->pos[r] = dist;
                    // stash joint id for solref lookup in the impedance pass
                    C->state[r] = j;
                    r++;
                }
            }
        }
        C->nl = r - C->nf;
        C->nefc = r;  // contacts appended below (needs jacobians)
    }
    __syncwarp();

    // ---- contact Jacobians: lanes cooperate per contact (3 rows x nv) ----
    // cJ row layout per contact c (condim 3): rows 4c..4c+3 pyramidal;
    // we first build the frame-rotated difference rows in cJ scratch.
    int ncon = C->ncon;
    int crow = 0;
    for (int c = 0; c < ncon; c++) {
        int p = C->con_pair[c];
        int g1 = g1c_pair_geom1[p], g2 = g1c_pair_geom2[p];
        int b1 = g1c_geom_bodyid[g1], b2 = g1c_geom_bodyid[g2];
        int condim = g1c_pair_dim[p];
        // jacdifp (3 x nv) into shared scratch = rows crow..crow+2 of cJ (temp)
        float* jd = C->cJ + crow * G1_NV;
        for (int k = lane; k < 3 * G1_NV; k += 32) jd[k] = 0.0f;
        __syncwarp();
        if (lane == 0) {
            // both bodies share ancestor dofs (free joint/torso chain), so the
            // accumulation must be serialized: body1 (-) then body2 (+)
            const float* point = C->con_pos + 3 * c;
            float off[3] = {point[0] - S->com[0], point[1] - S->com[1],
                            point[2] - S->com[2]};
            for (int side = 0; side < 2; side++) {
                int body = side == 0 ? b1 : b2;
                float sgn = side == 0 ? -1.0f : 1.0f;
                if (body == 0) continue;
                int i = g1c_body_dofadr[body] + g1c_body_dofnum[body] - 1;
                while (i >= 0) {
                    const float* cd = S->cdof + 6 * i;
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
        // rotate to contact frame: rows = fr (3x3) @ jd (3xnv), in place via temp regs
        float fr[9];
        make_frame(fr, C->con_norm + 3 * c);
        for (int i = lane; i < G1_NV; i += 32) {
            float v0 = jd[0 * G1_NV + i], v1 = jd[1 * G1_NV + i], v2 = jd[2 * G1_NV + i];
            jd[0 * G1_NV + i] = fr[0]*v0 + fr[1]*v1 + fr[2]*v2;
            jd[1 * G1_NV + i] = fr[3]*v0 + fr[4]*v1 + fr[5]*v2;
            jd[2 * G1_NV + i] = fr[6]*v0 + fr[7]*v1 + fr[8]*v2;
        }
        __syncwarp();
        if (condim == 1) {
            // single normal row already in place at crow
            if (lane == 0) {
                int r = C->nefc;
                C->row_type[r] = ROW_CONTACT;
                C->row_dof[r] = crow;
                C->row_sign[r] = 1.0f;
                C->pos[r] = C->con_dist[c];
                C->state[r] = -1 - c;   // stash contact index (negative-coded)
                C->nefc = r + 1;
            }
            crow += 1;  // rows crow+1, crow+2 are scratch, reused next contact
            __syncwarp();
            // compact: next contact's scratch starts at crow (overwrites old scratch)
        } else {
            // pyramidal: build 4 rows j0 +- mu_k*jk at crow..crow+3.
            // need rows beyond scratch: compute into rows crow..crow+3 using
            // the 3 scratch rows (in place safe: read rows 0..2, write 0..3
            // would clobber; use reverse order with temp registers per lane)
            float mu0 = g1c_pair_friction[5 * p + 0];
            float mu1 = g1c_pair_friction[5 * p + 1];
            for (int i = lane; i < G1_NV; i += 32) {
                float j0 = jd[0 * G1_NV + i];
                float j1 = jd[1 * G1_NV + i];
                float j2 = jd[2 * G1_NV + i];
                jd[0 * G1_NV + i] = j0 + mu0 * j1;
                jd[1 * G1_NV + i] = j0 - mu0 * j1;
                jd[2 * G1_NV + i] = j0 + mu1 * j2;
                jd[3 * G1_NV + i] = j0 - mu1 * j2;
            }
            __syncwarp();
            if (lane == 0) {
                for (int k = 0; k < 4; k++) {
                    int r = C->nefc + k;
                    C->row_type[r] = ROW_CONTACT;
                    C->row_dof[r] = crow + k;
                    C->row_sign[r] = 1.0f;
                    C->pos[r] = C->con_dist[c];
                    C->state[r] = -1 - c;
                }
                C->nefc += 4;
            }
            crow += 4;
            __syncwarp();
        }
    }
    if (lane == 0) C->ncrow = crow;
    __syncwarp();

    // ---- impedance -> R, D, aref ----
    for (int r = lane; r < C->nefc; r += 32) {
        const float* solref;
        const float* solimp;
        float dA;
        int is_fric = 0;
        if (C->row_type[r] == ROW_FRICTION) {
            int dof = C->row_dof[r];
            solref = g1c_dof_solref + 2 * dof;
            solimp = g1c_dof_solimp + 5 * dof;
            dA = g1c_dof_invweight0[dof];
            is_fric = 1;
        } else if (C->row_type[r] == ROW_LIMIT) {
            int j = C->state[r];  // stashed joint id
            solref = g1c_jnt_solref + 2 * j;
            solimp = g1c_jnt_solimp + 5 * j;
            dA = g1c_dof_invweight0[C->row_dof[r]];
        } else {
            int c = -1 - C->state[r];  // stashed contact index
            int p = C->con_pair[c];
            int g1 = g1c_pair_geom1[p], g2 = g1c_pair_geom2[p];
            solref = g1c_pair_solref + 2 * p;
            solimp = g1c_pair_solimp + 5 * p;
            float tran = g1c_body_invweight0_t[g1c_geom_bodyid[g1]]
                       + g1c_body_invweight0_t[g1c_geom_bodyid[g2]];
            if (g1c_pair_dim[p] == 1) {
                dA = tran;
            } else {
                // pyramidal pair row k: dA = tran + mu_k^2 * tran (k<2 translational)
                int k_in = (C->row_dof[r] - (C->row_dof[r] / 4) * 4) / 2;  // 0 or 1
                float mu = g1c_pair_friction[5 * p + k_in];
                dA = tran + mu * mu * tran;
            }
        }
        float imp = impedance(solimp, C->pos[r]);
        float Rr = fmaxf(1e-15f, (1.0f - imp) * dA / imp);
        float K, B;
        kbip_from(solref, solimp, imp, is_fric, &K, &B);
        C->R[r] = Rr;
        // aref needs vel; fill later (needs final R for nothing; aref uses K,B,imp)
        C->aref[r] = -B * row_dot(C, r, S->qvel) - K * imp * C->pos[r];
    }
    __syncwarp();

    // pyramidal R adjustment: Rpy = 2*mu_reg^2*R0, mu_reg = mu0/sqrt(impratio)
    if (lane == 0) {
        for (int r = C->nf + C->nl; r < C->nefc; ) {
            int c = -1 - C->state[r];
            int p = C->con_pair[c];
            int condim = g1c_pair_dim[p];
            if (condim > 1) {
                float R0 = C->R[r];
                float R1 = R0 / SOL_IMPRATIO;
                float mu = g1c_pair_friction[5 * p] * sqrtf(R1 / R0);
                float Rpy = 2.0f * mu * mu * R0;
                for (int k = 0; k < 4; k++) C->R[r + k] = Rpy;
                r += 4;
            } else {
                r += 1;
            }
        }
    }
    __syncwarp();
    for (int r = lane; r < C->nefc; r += 32) C->D[r] = 1.0f / C->R[r];
    __syncwarp();
}

// ---------------------------------------------------------------------------
// one FULL physics step (assumes S->qpos/qvel/ctrl set; ws = warm-start accel)
// ---------------------------------------------------------------------------
__device__ void full_step(EnvShared* S, ConShared* C, const float* ws, int lane) {
    smooth_forward(S, lane);                 // leaves qacc_smooth in S->qacc
    assemble_constraints(S, C, lane);
    newton_solve(S, C, ws, lane);            // leaves qacc in S->qacc
    euler_advance(S, lane, S->qacc);
}


// host copies for near-threshold classification in the validate compare
static float h_jrange[G1_NJNT * 2];
static int h_jlimited[G1_NJNT];
static short h_jqposadr[G1_NJNT];  // generated topology is int16

static int load_solver_consts(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("solver consts"); return 0; }
    float meaninertia;
    static float dof_iw[G1_NV], body_iw[G1_NBODY];
    static float dsolref[G1_NV * 2], dsolimp[G1_NV * 5];
    static float jsolref[G1_NJNT * 2], jsolimp[G1_NJNT * 5];
    float* jrange = h_jrange;
    int* jlimited = h_jlimited;
    if (fread(&meaninertia, 4, 1, f) != 1) return 0;
    if (fread(dof_iw, 4, G1_NV, f) != G1_NV) return 0;
    if (fread(body_iw, 4, G1_NBODY, f) != G1_NBODY) return 0;
    if (fread(dsolref, 4, G1_NV * 2, f) != G1_NV * 2) return 0;
    if (fread(dsolimp, 4, G1_NV * 5, f) != G1_NV * 5) return 0;
    if (fread(jsolref, 4, G1_NJNT * 2, f) != G1_NJNT * 2) return 0;
    if (fread(jsolimp, 4, G1_NJNT * 5, f) != G1_NJNT * 5) return 0;
    if (fread(jrange, 4, G1_NJNT * 2, f) != G1_NJNT * 2) return 0;
    if (fread(jlimited, 4, G1_NJNT, f) != G1_NJNT) return 0;
    fclose(f);
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_meaninertia, &meaninertia, 4));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_dof_invweight0, dof_iw, sizeof(dof_iw)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_body_invweight0_t, body_iw, sizeof(body_iw)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_dof_solref, dsolref, sizeof(dsolref)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_dof_solimp, dsolimp, sizeof(dsolimp)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_jnt_solref, jsolref, sizeof(jsolref)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_jnt_solimp, jsolimp, sizeof(jsolimp)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_jnt_range, jrange, sizeof(h_jrange)));
    CUDA_CHECK(cudaMemcpyToSymbol(g1c_jnt_limited, jlimited, sizeof(h_jlimited)));
    CUDA_CHECK(cudaMemcpyFromSymbol(h_jqposadr, g1c_jnt_qposadr, sizeof(h_jqposadr)));
    return 1;
}

// (requires tests/traj_format.h included by the translation unit)
// near-threshold classification: fp32 cannot be expected to reproduce the fp64
// active set when a contact or joint limit sits within eps of activation —
// those steps are excluded from the strict compare (A1 tolerance protocol)
// and counted separately. The rollout bounded-divergence check covers them.
#define NEAR_DIST 2e-6
static int near_threshold(const TrajStep* st) {
    for (int c = 0; c < st->ncon; c++)
        if (fabs(st->con[c].dist) < NEAR_DIST) return 1;
    for (int j = 0; j < G1_NJNT; j++) {
        if (!h_jlimited[j]) continue;
        double q = st->qpos[h_jqposadr[j]];
        if (fabs(q - h_jrange[2*j]) < NEAR_DIST ||
            fabs(q - h_jrange[2*j+1]) < NEAR_DIST) return 1;
    }
    return 0;
}


