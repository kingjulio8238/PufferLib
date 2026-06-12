// g1phys fp32 device math (mirrors mju_* semantics used by kinematics1).
#pragma once

__device__ __forceinline__ void quat_mul(float r[4], const float a[4], const float b[4]) {
    r[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3];
    r[1] = a[0]*b[1] + a[1]*b[0] + a[2]*b[3] - a[3]*b[2];
    r[2] = a[0]*b[2] - a[1]*b[3] + a[2]*b[0] + a[3]*b[1];
    r[3] = a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + a[3]*b[0];
}

__device__ __forceinline__ void quat_normalize(float q[4]) {
    float n = sqrtf(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]);
    float inv = (n > 1e-15f) ? 1.0f / n : 1.0f;
    q[0] *= inv; q[1] *= inv; q[2] *= inv; q[3] *= inv;
}

// rotate vector by quaternion (mju_rotVecQuat): r = q * v * q^-1
__device__ __forceinline__ void rot_vec_quat(float r[3], const float v[3], const float q[4]) {
    // r = v + 2w*(u x v) + 2*(u x (u x v)),  u = q.xyz, w = q.w
    float ux = q[1], uy = q[2], uz = q[3], w = q[0];
    float c1x = uy*v[2] - uz*v[1];
    float c1y = uz*v[0] - ux*v[2];
    float c1z = ux*v[1] - uy*v[0];
    float c2x = uy*c1z - uz*c1y;
    float c2y = uz*c1x - ux*c1z;
    float c2z = ux*c1y - uy*c1x;
    r[0] = v[0] + 2.0f*(w*c1x + c2x);
    r[1] = v[1] + 2.0f*(w*c1y + c2y);
    r[2] = v[2] + 2.0f*(w*c1z + c2z);
}

__device__ __forceinline__ void axis_angle_quat(float q[4], const float ax[3], float angle) {
    float h = 0.5f * angle;
    float s = sinf(h);
    q[0] = cosf(h);
    q[1] = ax[0] * s; q[2] = ax[1] * s; q[3] = ax[2] * s;
}

// quaternion -> 3x3 rotation matrix, row-major (mju_quat2Mat)
__device__ __forceinline__ void quat2mat(float m[9], const float q[4]) {
    float w = q[0], x = q[1], y = q[2], z = q[3];
    m[0] = 1.0f - 2.0f*(y*y + z*z); m[1] = 2.0f*(x*y - w*z); m[2] = 2.0f*(x*z + w*y);
    m[3] = 2.0f*(x*y + w*z); m[4] = 1.0f - 2.0f*(x*x + z*z); m[5] = 2.0f*(y*z - w*x);
    m[6] = 2.0f*(x*z - w*y); m[7] = 2.0f*(y*z + w*x); m[8] = 1.0f - 2.0f*(x*x + y*y);
}

__device__ __forceinline__ void cross3(float r[3], const float a[3], const float b[3]) {
    r[0] = a[1]*b[2] - a[2]*b[1];
    r[1] = a[2]*b[0] - a[0]*b[2];
    r[2] = a[0]*b[1] - a[1]*b[0];
}

__device__ __forceinline__ float dot6(const float* a, const float* b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4] + a[5]*b[5];
}

// mju_crossMotion: res = vel x v  (spatial motion vectors, [rot(3); trans(3)])
__device__ __forceinline__ void cross_motion(float res[6], const float vel[6], const float v[6]) {
    res[0] = -vel[2]*v[1] + vel[1]*v[2];
    res[1] =  vel[2]*v[0] - vel[0]*v[2];
    res[2] = -vel[1]*v[0] + vel[0]*v[1];
    res[3] = -vel[2]*v[4] + vel[1]*v[5] - vel[5]*v[1] + vel[4]*v[2];
    res[4] =  vel[2]*v[3] - vel[0]*v[5] + vel[5]*v[0] - vel[3]*v[2];
    res[5] = -vel[1]*v[3] + vel[0]*v[4] - vel[4]*v[0] + vel[3]*v[1];
}

// mju_crossForce: res = vel x* f  (spatial force vectors)
__device__ __forceinline__ void cross_force(float res[6], const float vel[6], const float f[6]) {
    res[0] = -vel[2]*f[1] + vel[1]*f[2] - vel[5]*f[4] + vel[4]*f[5];
    res[1] =  vel[2]*f[0] - vel[0]*f[2] + vel[5]*f[3] - vel[3]*f[5];
    res[2] = -vel[1]*f[0] + vel[0]*f[1] - vel[4]*f[3] + vel[3]*f[4];
    res[3] = -vel[2]*f[4] + vel[1]*f[5];
    res[4] =  vel[2]*f[3] - vel[0]*f[5];
    res[5] = -vel[1]*f[3] + vel[0]*f[4];
}

// mju_mulInertVec: spatial force = I(10) * motion(6)
__device__ __forceinline__ void mul_inert_vec(float res[6], const float* i, const float v[6]) {
    res[0] = i[0]*v[0] + i[3]*v[1] + i[4]*v[2] - i[8]*v[4] + i[7]*v[5];
    res[1] = i[3]*v[0] + i[1]*v[1] + i[5]*v[2] + i[8]*v[3] - i[6]*v[5];
    res[2] = i[4]*v[0] + i[5]*v[1] + i[2]*v[2] - i[7]*v[3] + i[6]*v[4];
    res[3] = i[8]*v[1] - i[7]*v[2] + i[9]*v[3];
    res[4] = i[6]*v[2] - i[8]*v[0] + i[9]*v[4];
    res[5] = i[7]*v[0] - i[6]*v[1] + i[9]*v[5];
}

// mju_inertCom: 10-vec [Ixx Iyy Izz Ixy Ixz Iyz; mass*dif(3); mass]
__device__ __forceinline__ void inert_com(float* res, const float inert[3],
                                          const float mat[9], const float dif[3],
                                          float mass) {
    float tmp[9] = {mat[0]*inert[0], mat[3]*inert[0], mat[6]*inert[0],
                    mat[1]*inert[1], mat[4]*inert[1], mat[7]*inert[1],
                    mat[2]*inert[2], mat[5]*inert[2], mat[8]*inert[2]};
    res[0] = mat[0]*tmp[0] + mat[1]*tmp[3] + mat[2]*tmp[6] + mass*(dif[1]*dif[1] + dif[2]*dif[2]);
    res[1] = mat[3]*tmp[1] + mat[4]*tmp[4] + mat[5]*tmp[7] + mass*(dif[0]*dif[0] + dif[2]*dif[2]);
    res[2] = mat[6]*tmp[2] + mat[7]*tmp[5] + mat[8]*tmp[8] + mass*(dif[0]*dif[0] + dif[1]*dif[1]);
    res[3] = mat[0]*tmp[1] + mat[1]*tmp[4] + mat[2]*tmp[7] - mass*dif[0]*dif[1];
    res[4] = mat[0]*tmp[2] + mat[1]*tmp[5] + mat[2]*tmp[8] - mass*dif[0]*dif[2];
    res[5] = mat[3]*tmp[2] + mat[4]*tmp[5] + mat[5]*tmp[8] - mass*dif[1]*dif[2];
    res[6] = mass*dif[0]; res[7] = mass*dif[1]; res[8] = mass*dif[2];
    res[9] = mass;
}

// mju_quatIntegrate: quat <- normalize(quat) * axisangle(vel/|vel|, scale*|vel|)
__device__ __forceinline__ void quat_integrate(float quat[4], const float vel[3], float scale) {
    float tmp[3] = {vel[0], vel[1], vel[2]};
    float n = sqrtf(tmp[0]*tmp[0] + tmp[1]*tmp[1] + tmp[2]*tmp[2]);
    if (n > 1e-15f) { tmp[0] /= n; tmp[1] /= n; tmp[2] /= n; }
    float qrot[4];
    axis_angle_quat(qrot, tmp, scale * n);
    quat_normalize(quat);
    float out[4];
    quat_mul(out, quat, qrot);
    quat[0] = out[0]; quat[1] = out[1]; quat[2] = out[2]; quat[3] = out[3];
}
